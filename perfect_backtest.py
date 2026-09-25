import pandas as pd
import numpy as np
from algo_vpin_v2.data_feed import DhanDataFeed
from algo_vpin_v2.macro_features import MacroFeatureEngine
from algo_vpin_v2.vpin import VPINCalculator
from algo_vpin_v2.garch_engine import GARCHEngine, DirectionalSignal
from algo_vpin_v2.ensemble_brain import EnsembleBrain

def perfect_institutional_backtest():
    feed = DhanDataFeed()
    hist = feed.fetch_historical_bars(days=5)

    # Convert UTC timestamps accurately to IST (Indian Standard Time: UTC+5:30)
    if 'timestamp' in hist.columns:
        hist['timestamp'] = pd.to_datetime(hist['timestamp'])
        if hist['timestamp'].dt.tz is None:
            # Tradehull UTC -> convert to IST
            hist['timestamp'] = hist['timestamp'].dt.tz_localize('UTC').dt.tz_convert('Asia/Kolkata')
        else:
            hist['timestamp'] = hist['timestamp'].dt.tz_convert('Asia/Kolkata')

    macro = MacroFeatureEngine()
    vpin_calc = VPINCalculator()
    garch = GARCHEngine()
    brain = EnsembleBrain()

    trades = []
    pos = {'side': 'FLAT', 'contract': '', 'entry_p': 0.0, 'sl': 0.0, 'high_p': 0.0, 'entry_spot': 0.0, 'entry_time': None}
    cooldown_until = None
    trades_today = 0
    current_day = None

    for idx, row in hist.iterrows():
        p = float(row['close'])
        h = float(row.get('high', row['close']))
        l = float(row.get('low', row['close']))
        v = float(row['volume'])
        ts = row.get('timestamp')

        # Reset daily trade counter on new calendar day
        trade_date = ts.strftime('%d-%b-%Y')
        if trade_date != current_day:
            current_day = trade_date
            trades_today = 0
            cooldown_until = None

        # Filter active Indian market session: 09:15 to 15:24 IST
        is_market_hours = (ts.hour == 9 and ts.minute >= 15) or (10 <= ts.hour < 15) or (ts.hour == 15 and ts.minute <= 24)

        mst = macro.update_1min_bar(p, h, l)
        vres = vpin_calc.process_bar(p, v, ts)
        gres = garch.add_bar(p, ts)

        p_delta = p - (hist.iloc[idx-1]['close'] if idx > 0 else p)
        sigma = gres.sigma_next if gres else 0.001
        mu = gres.mu_next if gres else 0.0

        if idx < 150:
            continue

        dec = brain.evaluate_signal(gres.signal if gres else DirectionalSignal.HOLD, p_delta, sigma, mu, vres.vpin, mst)

        # 1. Check open position exit
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

            # Exit triggers: SL hit, Trailing SL hit, or Intraday 15:24 Square-off
            is_squareoff = (ts.hour == 15 and ts.minute >= 24)
            if curr_opt <= pos['sl'] or is_squareoff:
                exit_price = pos['sl'] if not is_squareoff else curr_opt
                pnl = (exit_price - pos['entry_p']) * 65
                ret_pct = ((exit_price - pos['entry_p']) / pos['entry_p']) * 100
                exit_reason = 'INTRADAY_SQUAREOFF' if is_squareoff else ('TRAILING_SL_EXIT' if exit_price >= pos['entry_p'] else 'INITIAL_SL_HIT')

                trades.append({
                    'Date': trade_date,
                    'Contract': pos['contract'],
                    'Entry Time': pos['entry_time'].strftime('%H:%M'),
                    'Exit Time': ts.strftime('%H:%M'),
                    'Spot In': f"{pos['entry_spot']:,.2f}",
                    'Spot Out': f"{p:,.2f}",
                    'Entry Prem': f"INR {pos['entry_p']:.2f}",
                    'Peak Prem': f"INR {pos['high_p']:.2f} (+{gain_pct*100:.1f}%)",
                    'Exit Prem': f"INR {exit_price:.2f}",
                    'PnL (INR)': pnl,
                    'Return %': f"{ret_pct:+.1f}%",
                    'Exit Reason': exit_reason
                })

                # If SL hit, apply 10-minute cooldown to avoid chop traps
                if exit_price < pos['entry_p']:
                    cooldown_until = ts + pd.Timedelta(minutes=10)

                pos = {'side': 'FLAT', 'contract': '', 'entry_p': 0.0, 'sl': 0.0, 'high_p': 0.0, 'entry_spot': 0.0, 'entry_time': None}

            # Emergency Model Reversal Exit
            elif not dec.is_vetoed and dec.final_action != DirectionalSignal.HOLD and (
                (pos['side'] == 'BUY' and dec.final_action == DirectionalSignal.SELL) or
                (pos['side'] == 'SELL' and dec.final_action == DirectionalSignal.BUY)
            ):
                pnl = (curr_opt - pos['entry_p']) * 65
                ret_pct = ((curr_opt - pos['entry_p']) / pos['entry_p']) * 100
                trades.append({
                    'Date': trade_date,
                    'Contract': pos['contract'],
                    'Entry Time': pos['entry_time'].strftime('%H:%M'),
                    'Exit Time': ts.strftime('%H:%M'),
                    'Spot In': f"{pos['entry_spot']:,.2f}",
                    'Spot Out': f"{p:,.2f}",
                    'Entry Prem': f"INR {pos['entry_p']:.2f}",
                    'Peak Prem': f"INR {pos['high_p']:.2f} (+{gain_pct*100:.1f}%)",
                    'Exit Prem': f"INR {curr_opt:.2f}",
                    'PnL (INR)': pnl,
                    'Return %': f"{ret_pct:+.1f}%",
                    'Exit Reason': 'EMERGENCY_REVERSAL'
                })
                pos = {'side': 'FLAT', 'contract': '', 'entry_p': 0.0, 'sl': 0.0, 'high_p': 0.0, 'entry_spot': 0.0, 'entry_time': None}

        # 2. Entry Logic (Precision Quality Rules)
        in_cooldown = cooldown_until is not None and ts < cooldown_until
        can_trade_today = trades_today < 4

        if (pos['side'] == 'FLAT' and is_market_hours and not in_cooldown and can_trade_today and
            not dec.is_vetoed and dec.final_action in (DirectionalSignal.BUY, DirectionalSignal.SELL) and
            dec.xgb_confidence >= 0.55):

            atm_strike = round(p / 50) * 50
            opt_type = "CALL" if dec.final_action == DirectionalSignal.BUY else "PUT"
            contract_sym = f"NIFTY {atm_strike} {opt_type}"
            entry_opt = round(max(50.0, p * 0.0045), 2)
            sl_opt = round(entry_opt * (1.0 - 0.20), 2)  # 20% Initial SL

            pos = {
                'side': 'BUY' if dec.final_action == DirectionalSignal.BUY else 'SELL',
                'contract': contract_sym,
                'entry_p': entry_opt,
                'sl': sl_opt,
                'high_p': entry_opt,
                'entry_spot': p,
                'entry_time': ts
            }
            trades_today += 1

    tdf = pd.DataFrame(trades)
    print("=" * 130)
    print("               INSTITUTIONAL PRECISION ENGINE - DAY-BY-DAY TRADE REPORT (IST TIME)")
    print("=" * 130)

    for date, group in tdf.groupby('Date', sort=False):
        day_pnl = group['PnL (INR)'].sum()
        pnl_badge = f"+INR {day_pnl:,.2f}" if day_pnl >= 0 else f"-INR {abs(day_pnl):,.2f}"
        win_count = len(group[group['PnL (INR)'] > 0])
        print(f"\n[DATE: {date}] | Trades: {len(group)} (Wins: {win_count}) | Net Day P&L: {pnl_badge}")
        print("-" * 130)
        for idx, t in group.reset_index(drop=True).iterrows():
            t_pnl = f"+INR {t['PnL (INR)']:,.2f}" if t['PnL (INR)'] >= 0 else f"-INR {abs(t['PnL (INR)']):,.2f}"
            print(f"  Trade #{idx+1:02d} | {t['Entry Time']} -> {t['Exit Time']} IST | {t['Contract']:20s} | Spot: {t['Spot In']} -> {t['Spot Out']} | Entry: {t['Entry Prem']} | Peak: {t['Peak Prem']} | Exit: {t['Exit Prem']} | PnL: {t_pnl:14s} ({t['Return %']:6s}) | {t['Exit Reason']}")

    wins = tdf[tdf['PnL (INR)'] > 0]
    losses = tdf[tdf['PnL (INR)'] < 0]
    be = tdf[tdf['PnL (INR)'] == 0]
    total_trades = len(tdf)
    win_rate = len(wins) / total_trades * 100
    gross_profit = wins['PnL (INR)'].sum()
    gross_loss = abs(losses['PnL (INR)'].sum())
    profit_factor = gross_profit / gross_loss
    net_pnl = tdf['PnL (INR)'].sum()

    print("\n" + "=" * 80)
    print("                   PERFECTED STRATEGY PERFORMANCE")
    print("=" * 80)
    print(f"• Total Trades Executed       : {total_trades}")
    print(f"• Winning Trades (Profit)     : {len(wins)} ({win_rate:.1f}%)")
    print(f"• Losing Trades (20% SL)      : {len(losses)} ({len(losses)/total_trades*100:.1f}%)")
    print(f"• Breakeven Saves (0% Loss)   : {len(be)} ({len(be)/total_trades*100:.1f}%)")
    print("-" * 80)
    print(f"• Average Winning Trade       : +INR {wins['PnL (INR)'].mean():,.2f}")
    print(f"• Average Losing Trade        : -INR {abs(losses['PnL (INR)'].mean()):,.2f}")
    print(f"• Realized Risk-to-Reward     : 1 : {abs(wins['PnL (INR)'].mean()/losses['PnL (INR)'].mean()):.2f}")
    print(f"• Largest Winning Trade       : +INR {wins['PnL (INR)'].max():,.2f}")
    print("-" * 80)
    print(f"• Total Gross Profit          : +INR {gross_profit:,.2f}")
    print(f"• Total Gross Loss            : -INR {gross_loss:,.2f}")
    print(f"• Realized Profit Factor      : {profit_factor:.2f}")
    print(f"• Net Strategy P&L (5 Days)   : INR {net_pnl:+,.2f} (+{net_pnl/50000*100:.1f}% on 50k)")
    print("=" * 80)

if __name__ == '__main__':
    perfect_institutional_backtest()
