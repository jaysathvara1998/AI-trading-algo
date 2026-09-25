import pandas as pd
import numpy as np
import pytz
from algo_vpin_v2.data_feed import DhanDataFeed
from algo_vpin_v2.macro_features import MacroFeatureEngine
from algo_vpin_v2.vpin import VPINCalculator
from algo_vpin_v2.garch_engine import GARCHEngine, DirectionalSignal
from algo_vpin_v2.ensemble_brain import EnsembleBrain

def generate_day_by_day_report():
    feed = DhanDataFeed()
    hist = feed.fetch_historical_bars(days=5)
    
    # Ensure proper IST timezone parsing
    if 'timestamp' in hist.columns:
        hist['timestamp'] = pd.to_datetime(hist['timestamp'])
        if hist['timestamp'].dt.tz is None:
            hist['timestamp'] = hist['timestamp'].dt.tz_localize('Asia/Kolkata')
        else:
            hist['timestamp'] = hist['timestamp'].dt.tz_convert('Asia/Kolkata')
    
    macro = MacroFeatureEngine()
    vpin_calc = VPINCalculator()
    garch = GARCHEngine()
    brain = EnsembleBrain()

    trades = []
    pos = {'side': 'FLAT', 'contract': '', 'entry_p': 0.0, 'sl': 0.0, 'high_p': 0.0, 'entry_spot': 0.0, 'entry_time': None}

    for idx, row in hist.iterrows():
        p = float(row['close'])
        h = float(row.get('high', row['close']))
        l = float(row.get('low', row['close']))
        v = float(row['volume'])
        ts = row.get('timestamp')

        mst = macro.update_1min_bar(p, h, l)
        vres = vpin_calc.process_bar(p, v, ts)
        gres = garch.add_bar(p, ts)

        p_delta = p - (hist.iloc[idx-1]['close'] if idx > 0 else p)
        sigma = gres.sigma_next if gres else 0.001
        mu = gres.mu_next if gres else 0.0

        if idx < 150:
            continue

        dec = brain.evaluate_signal(gres.signal if gres else DirectionalSignal.HOLD, p_delta, sigma, mu, vres.vpin, mst)

        # Active position management
        if pos['side'] != 'FLAT':
            spot_change = (p - pos['entry_spot']) if pos['side'] == 'BUY' else (pos['entry_spot'] - p)
            curr_opt = max(5.0, pos['entry_p'] + spot_change * 0.55)
            pos['high_p'] = max(pos['high_p'], curr_opt)
            gain_pct = (pos['high_p'] - pos['entry_p']) / pos['entry_p']

            # Trailing SL: Move to breakeven at +20%
            if gain_pct >= 0.20 and pos['sl'] < pos['entry_p']:
                pos['sl'] = pos['entry_p']

            # Beyond +30%: Continuous Trailing at Peak - 15%
            if gain_pct >= 0.30:
                dynamic_trail = round(pos['high_p'] * 0.85, 2)
                if dynamic_trail > pos['sl']:
                    pos['sl'] = dynamic_trail

            # SL / Trailing SL Exit
            if curr_opt <= pos['sl']:
                pnl = (pos['sl'] - pos['entry_p']) * 65
                ret_pct = ((pos['sl'] - pos['entry_p']) / pos['entry_p']) * 100
                trades.append({
                    'Date': ts.strftime('%d-%b-%Y'),
                    'Contract': pos['contract'],
                    'Entry Time': pos['entry_time'].strftime('%H:%M:%S'),
                    'Exit Time': ts.strftime('%H:%M:%S'),
                    'Spot @ Entry': f"{pos['entry_spot']:,.2f}",
                    'Spot @ Exit': f"{p:,.2f}",
                    'Entry Prem': f"INR {pos['entry_p']:.2f}",
                    'Peak Prem': f"INR {pos['high_p']:.2f} (+{gain_pct*100:.1f}%)",
                    'Exit Prem': f"INR {pos['sl']:.2f}",
                    'PnL (INR)': pnl,
                    'Return %': f"{ret_pct:+.1f}%",
                    'Exit Reason': 'TRAILING_SL_EXIT' if pos['sl'] >= pos['entry_p'] else 'INITIAL_SL_HIT'
                })
                pos = {'side': 'FLAT', 'contract': '', 'entry_p': 0.0, 'sl': 0.0, 'high_p': 0.0, 'entry_spot': 0.0, 'entry_time': None}

            # Emergency Model Reversal Exit
            elif not dec.is_vetoed and dec.final_action != DirectionalSignal.HOLD and (
                (pos['side'] == 'BUY' and dec.final_action == DirectionalSignal.SELL) or
                (pos['side'] == 'SELL' and dec.final_action == DirectionalSignal.BUY)
            ):
                pnl = (curr_opt - pos['entry_p']) * 65
                ret_pct = ((curr_opt - pos['entry_p']) / pos['entry_p']) * 100
                trades.append({
                    'Date': ts.strftime('%d-%b-%Y'),
                    'Contract': pos['contract'],
                    'Entry Time': pos['entry_time'].strftime('%H:%M:%S'),
                    'Exit Time': ts.strftime('%H:%M:%S'),
                    'Spot @ Entry': f"{pos['entry_spot']:,.2f}",
                    'Spot @ Exit': f"{p:,.2f}",
                    'Entry Prem': f"INR {pos['entry_p']:.2f}",
                    'Peak Prem': f"INR {pos['high_p']:.2f} (+{gain_pct*100:.1f}%)",
                    'Exit Prem': f"INR {curr_opt:.2f}",
                    'PnL (INR)': pnl,
                    'Return %': f"{ret_pct:+.1f}%",
                    'Exit Reason': 'EMERGENCY_REVERSAL'
                })
                pos = {'side': 'FLAT', 'contract': '', 'entry_p': 0.0, 'sl': 0.0, 'high_p': 0.0, 'entry_spot': 0.0, 'entry_time': None}

        # Entry logic
        if pos['side'] == 'FLAT' and not dec.is_vetoed and dec.final_action in (DirectionalSignal.BUY, DirectionalSignal.SELL):
            atm_strike = round(p / 50) * 50
            opt_type = "CALL" if dec.final_action == DirectionalSignal.BUY else "PUT"
            contract_sym = f"NIFTY {atm_strike} {opt_type}"
            entry_opt = round(max(50.0, p * 0.0045), 2)
            sl_opt = round(entry_opt * (1.0 - 0.20), 2)
            pos = {
                'side': 'BUY' if dec.final_action == DirectionalSignal.BUY else 'SELL',
                'contract': contract_sym,
                'entry_p': entry_opt,
                'sl': sl_opt,
                'high_p': entry_opt,
                'entry_spot': p,
                'entry_time': ts
            }

    tdf = pd.DataFrame(trades)
    print("=" * 125)
    print("                      DAY-BY-DAY TRADE VERIFICATION REPORT (NIFTY OPTIONS)")
    print("=" * 125)
    
    for date, group in tdf.groupby('Date', sort=False):
        day_pnl = group['PnL (INR)'].sum()
        pnl_badge = f"+INR {day_pnl:,.2f}" if day_pnl >= 0 else f"-INR {abs(day_pnl):,.2f}"
        print(f"\n[DATE: {date}] | Total Trades: {len(group)} | Day Net P&L: {pnl_badge}")
        print("-" * 125)
        for idx, t in group.reset_index(drop=True).iterrows():
            t_pnl = f"+INR {t['PnL (INR)']:,.2f}" if t['PnL (INR)'] >= 0 else f"-INR {abs(t['PnL (INR)']):,.2f}"
            print(f"  Trade #{idx+1:02d} | {t['Entry Time']} -> {t['Exit Time']} | {t['Contract']:20s} | Spot: {t['Spot @ Entry']} -> {t['Spot @ Exit']} | Entry: {t['Entry Prem']} | Peak: {t['Peak Prem']} | Exit: {t['Exit Prem']} | PnL: {t_pnl:14s} ({t['Return %']:6s}) | {t['Exit Reason']}")

    print("\n" + "=" * 125)

if __name__ == '__main__':
    generate_day_by_day_report()
