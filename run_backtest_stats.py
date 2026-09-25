import pandas as pd
import numpy as np
from algo_vpin_v2.data_feed import DhanDataFeed
from algo_vpin_v2.macro_features import MacroFeatureEngine
from algo_vpin_v2.vpin import VPINCalculator
from algo_vpin_v2.garch_engine import GARCHEngine, DirectionalSignal
from algo_vpin_v2.ensemble_brain import EnsembleBrain

def run_backtest():
    feed = DhanDataFeed()
    hist = feed.fetch_historical_bars(days=5)
    print(f"Loaded {len(hist)} historical 1-minute bars.")

    macro = MacroFeatureEngine()
    vpin_calc = VPINCalculator()
    garch = GARCHEngine()
    brain = EnsembleBrain()

    trades = []
    pos = {'side': 'FLAT', 'entry_p': 0.0, 'sl': 0.0, 'tp': 0.0, 'high_p': 0.0, 'entry_spot': 0.0}

    brain.train_models()

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

        dec = brain.evaluate(gres.signal if gres else DirectionalSignal.HOLD, p_delta, sigma, mu, vres.vpin, mst)

        # Check exit on active position
        if pos['side'] != 'FLAT':
            spot_change = (p - pos['entry_spot']) if pos['side'] == 'BUY' else (pos['entry_spot'] - p)
            curr_opt = max(5.0, pos['entry_p'] + spot_change * 0.55)
            pos['high_p'] = max(pos['high_p'], curr_opt)

            # Trailing Stop-Loss
            if pos['high_p'] >= pos['entry_p'] * 1.20 and pos['sl'] < pos['entry_p']:
                pos['sl'] = pos['entry_p']
            if pos['high_p'] >= pos['entry_p'] * 1.40 and pos['sl'] < pos['entry_p'] * 1.20:
                pos['sl'] = pos['entry_p'] * 1.20

            if curr_opt <= pos['sl']:
                pnl = (pos['sl'] - pos['entry_p']) * 65
                trades.append({'pnl': pnl, 'reason': 'STOP_LOSS / TRAILING_SL'})
                pos = {'side': 'FLAT', 'entry_p': 0.0, 'sl': 0.0, 'tp': 0.0, 'high_p': 0.0, 'entry_spot': 0.0}
            elif curr_opt >= pos['tp']:
                pnl = (pos['tp'] - pos['entry_p']) * 65
                trades.append({'pnl': pnl, 'reason': 'TAKE_PROFIT_HIT'})
                pos = {'side': 'FLAT', 'entry_p': 0.0, 'sl': 0.0, 'tp': 0.0, 'high_p': 0.0, 'entry_spot': 0.0}
            elif not dec.is_vetoed and dec.final_action != DirectionalSignal.HOLD and (
                (pos['side'] == 'BUY' and dec.final_action == DirectionalSignal.SELL) or
                (pos['side'] == 'SELL' and dec.final_action == DirectionalSignal.BUY)
            ):
                pnl = (curr_opt - pos['entry_p']) * 65
                trades.append({'pnl': pnl, 'reason': 'EMERGENCY_REVERSAL'})
                pos = {'side': 'FLAT', 'entry_p': 0.0, 'sl': 0.0, 'tp': 0.0, 'high_p': 0.0, 'entry_spot': 0.0}

        # Check entry
        if pos['side'] == 'FLAT' and not dec.is_vetoed and dec.final_action in (DirectionalSignal.BUY, DirectionalSignal.SELL) and dec.xgb_confidence >= 0.52:
            entry_opt = round(max(50.0, p * 0.0045), 2)
            sl_opt = round(entry_opt * (1.0 - 0.35), 2)
            tp_opt = round(entry_opt * (1.0 + 0.65), 2)
            pos = {
                'side': 'BUY' if dec.final_action == DirectionalSignal.BUY else 'SELL',
                'entry_p': entry_opt,
                'sl': sl_opt,
                'tp': tp_opt,
                'high_p': entry_opt,
                'entry_spot': p
            }

    # Close any open trade at the end of simulation
    if pos['side'] != 'FLAT':
        final_p = float(hist.iloc[-1]['close'])
        spot_change = (final_p - pos['entry_spot']) if pos['side'] == 'BUY' else (pos['entry_spot'] - final_p)
        curr_opt = max(5.0, pos['entry_p'] + spot_change * 0.55)
        pnl = (curr_opt - pos['entry_p']) * 65
        trades.append({'pnl': pnl, 'reason': 'MARKET_CLOSE_CUTOFF'})

    tdf = pd.DataFrame(trades)
    if tdf.empty:
        print("No trades generated with current threshold.")
        return

    wins = tdf[tdf['pnl'] > 0]
    losses = tdf[tdf['pnl'] < 0]
    be = tdf[tdf['pnl'] == 0]
    total_trades = len(tdf)
    win_rate = (len(wins) / total_trades * 100) if total_trades > 0 else 0
    gross_win = wins['pnl'].sum() if not wins.empty else 0.0
    gross_loss = abs(losses['pnl'].sum()) if not losses.empty else 0.0
    profit_factor = (gross_win / gross_loss) if gross_loss > 0 else (99.0 if gross_win > 0 else 0.0)

    print("=" * 65)
    print("      QUANTITATIVE BACKTEST METRICS (5 DAYS / 1,875 BARS)")
    print("=" * 65)
    print(f"• Total Trades Executed       : {total_trades}")
    print(f"• Winning Trades (Targets/TSL): {len(wins)} ({win_rate:.1f}%)")
    print(f"• Losing Trades (SL Hits)     : {len(losses)} ({len(losses)/total_trades*100:.1f}%)")
    print(f"• Breakeven Exits (TSL Save)  : {len(be)} ({len(be)/total_trades*100:.1f}%)")
    print("-" * 65)
    print(f"• Average Winning Trade       : +INR {wins['pnl'].mean():,.2f}" if not wins.empty else "• Average Winning Trade: INR 0")
    print(f"• Average Losing Trade        : -INR {abs(losses['pnl'].mean()):,.2f}" if not losses.empty else "• Average Losing Trade: INR 0")
    print(f"• Realized Risk-to-Reward (RR): 1 : {abs(wins['pnl'].mean() / losses['pnl'].mean()):.2f}" if not wins.empty and not losses.empty else "")
    print("-" * 65)
    print(f"• Total Gross Profit          : +INR {gross_win:,.2f}")
    print(f"• Total Gross Loss            : -INR {gross_loss:,.2f}")
    print(f"• Realized Profit Factor      : {profit_factor:.2f}")
    print(f"• Net Strategy P&L (5 Days)   : INR {tdf['pnl'].sum():+,.2f}")
    print("=" * 65)

if __name__ == '__main__':
    run_backtest()
