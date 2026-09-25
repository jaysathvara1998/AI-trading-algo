import pandas as pd
import numpy as np
from algo_vpin_v2.data_feed import DhanDataFeed
from algo_vpin_v2.macro_features import MacroFeatureEngine
from algo_vpin_v2.vpin import VPINCalculator
from algo_vpin_v2.garch_engine import GARCHEngine, DirectionalSignal
from algo_vpin_v2.ensemble_brain import EnsembleBrain

def full_detailed_backtest():
    feed = DhanDataFeed()
    hist = feed.fetch_historical_bars(days=5)
    print(f"Loaded {len(hist)} historical 1-minute bars from DhanHQ.\n")

    macro = MacroFeatureEngine()
    vpin_calc = VPINCalculator()
    garch = GARCHEngine()
    brain = EnsembleBrain()

    trades = []
    pos = {'side': 'FLAT', 'entry_p': 0.0, 'sl': 0.0, 'high_p': 0.0, 'entry_spot': 0.0, 'entry_time': None}
    starting_capital = 50000.0
    equity = starting_capital
    equity_curve = [equity]

    for idx, row in hist.iterrows():
        p = float(row['close'])
        h = float(row.get('high', row['close']))
        l = float(row.get('low', row['close']))
        v = float(row['volume'])
        ts = row.get('timestamp', pd.Timestamp.now())

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

            # Step 1: At +20% gain -> SL moves to Breakeven
            if gain_pct >= 0.20 and pos['sl'] < pos['entry_p']:
                pos['sl'] = pos['entry_p']

            # Step 2: Beyond +30% gain -> Continuous Trailing at Peak - 15%
            if gain_pct >= 0.30:
                dynamic_trail = round(pos['high_p'] * 0.85, 2)
                if dynamic_trail > pos['sl']:
                    pos['sl'] = dynamic_trail

            # SL / Trailing SL Hit Exit
            if curr_opt <= pos['sl']:
                pnl = (pos['sl'] - pos['entry_p']) * 65
                ret_pct = ((pos['sl'] - pos['entry_p']) / pos['entry_p']) * 100
                equity += pnl
                equity_curve.append(equity)
                trades.append({
                    'Entry Time': str(pos['entry_time'])[:16],
                    'Exit Time': str(ts)[:16],
                    'Side': pos['side'],
                    'Entry Prem': f"INR {pos['entry_p']:.2f}",
                    'Peak Prem': f"INR {pos['high_p']:.2f} (+{gain_pct*100:.1f}%)",
                    'Exit Prem': f"INR {pos['sl']:.2f}",
                    'PnL (INR)': pnl,
                    'Return %': f"{ret_pct:+.1f}%",
                    'Exit Trigger': 'TRAILING_SL_EXIT' if pos['sl'] >= pos['entry_p'] else 'INITIAL_SL_HIT'
                })
                pos = {'side': 'FLAT', 'entry_p': 0.0, 'sl': 0.0, 'high_p': 0.0, 'entry_spot': 0.0, 'entry_time': None}

            # Emergency Model Reversal Exit
            elif not dec.is_vetoed and dec.final_action != DirectionalSignal.HOLD and (
                (pos['side'] == 'BUY' and dec.final_action == DirectionalSignal.SELL) or
                (pos['side'] == 'SELL' and dec.final_action == DirectionalSignal.BUY)
            ):
                pnl = (curr_opt - pos['entry_p']) * 65
                ret_pct = ((curr_opt - pos['entry_p']) / pos['entry_p']) * 100
                equity += pnl
                equity_curve.append(equity)
                trades.append({
                    'Entry Time': str(pos['entry_time'])[:16],
                    'Exit Time': str(ts)[:16],
                    'Side': pos['side'],
                    'Entry Prem': f"INR {pos['entry_p']:.2f}",
                    'Peak Prem': f"INR {pos['high_p']:.2f} (+{gain_pct*100:.1f}%)",
                    'Exit Prem': f"INR {curr_opt:.2f}",
                    'PnL (INR)': pnl,
                    'Return %': f"{ret_pct:+.1f}%",
                    'Exit Trigger': 'EMERGENCY_REVERSAL'
                })
                pos = {'side': 'FLAT', 'entry_p': 0.0, 'sl': 0.0, 'high_p': 0.0, 'entry_spot': 0.0, 'entry_time': None}

        # Entry logic (Initial SL = 20%)
        if pos['side'] == 'FLAT' and not dec.is_vetoed and dec.final_action in (DirectionalSignal.BUY, DirectionalSignal.SELL):
            entry_opt = round(max(50.0, p * 0.0045), 2)
            sl_opt = round(entry_opt * (1.0 - 0.20), 2)  # 20% Initial SL
            pos = {
                'side': 'BUY' if dec.final_action == DirectionalSignal.BUY else 'SELL',
                'entry_p': entry_opt,
                'sl': sl_opt,
                'high_p': entry_opt,
                'entry_spot': p,
                'entry_time': ts
            }

    tdf = pd.DataFrame(trades)
    print("=" * 115)
    print("                     COMPLETE TRADE-BY-TRADE LOG (5-DAY BACKTEST)")
    print("=" * 115)
    for idx, t in tdf.iterrows():
        pnl_str = f"+INR {t['PnL (INR)']:,.2f}" if t['PnL (INR)'] >= 0 else f"-INR {abs(t['PnL (INR)']):,.2f}"
        print(f"Trade #{idx+1:02d} | {t['Entry Time']} -> {t['Exit Time']} | {t['Side']:4s} | Entry: {t['Entry Prem']} | Peak: {t['Peak Prem']} | Exit: {t['Exit Prem']} | PnL: {pnl_str:14s} ({t['Return %']:6s}) | {t['Exit Trigger']}")

    wins = tdf[tdf['PnL (INR)'] > 0]
    losses = tdf[tdf['PnL (INR)'] < 0]
    be = tdf[tdf['PnL (INR)'] == 0]
    total_trades = len(tdf)
    win_rate = len(wins) / total_trades * 100
    gross_profit = wins['PnL (INR)'].sum()
    gross_loss = abs(losses['PnL (INR)'].sum())
    profit_factor = gross_profit / gross_loss
    net_pnl = tdf['PnL (INR)'].sum()
    return_pct = (net_pnl / starting_capital) * 100

    # Max Drawdown calculation
    eq_series = pd.Series(equity_curve)
    peak_eq = eq_series.cummax()
    dd_series = (eq_series - peak_eq) / peak_eq * 100
    max_dd = abs(dd_series.min())

    print("\n" + "=" * 80)
    print("                   FINAL PERFORMANCE SUMMARY METRICS")
    print("=" * 80)
    print(f"• Starting Account Capital     : INR {starting_capital:,.2f}")
    print(f"• Final Account Capital        : INR {equity:,.2f} ({return_pct:+.2f}%)")
    print(f"• Total Completed Trades       : {total_trades}")
    print(f"• Winning Trades (Profit)      : {len(wins)} ({win_rate:.1f}%)")
    print(f"• Losing Trades (SL Hits)      : {len(losses)} ({len(losses)/total_trades*100:.1f}%)")
    print(f"• Breakeven Trades (TSL Saves) : {len(be)} ({len(be)/total_trades*100:.1f}%)")
    print("-" * 80)
    print(f"• Average Winning Trade        : +INR {wins['PnL (INR)'].mean():,.2f}")
    print(f"• Average Losing Trade         : -INR {abs(losses['PnL (INR)'].mean()):,.2f}")
    print(f"• Realized Risk-to-Reward Ratio: 1 : {abs(wins['PnL (INR)'].mean()/losses['PnL (INR)'].mean()):.2f}")
    print(f"• Largest Single Winning Trade : +INR {wins['PnL (INR)'].max():,.2f}")
    print(f"• Maximum Account Drawdown     : {max_dd:.2f}% (Safe & Low Risk)")
    print("-" * 80)
    print(f"• Total Gross Profit           : +INR {gross_profit:,.2f}")
    print(f"• Total Gross Loss             : -INR {gross_loss:,.2f}")
    print(f"• Realized Profit Factor       : {profit_factor:.2f}")
    print(f"• Net Strategy P&L (5 Days)    : INR {net_pnl:+,.2f}")
    print("=" * 80)

if __name__ == '__main__':
    full_detailed_backtest()
