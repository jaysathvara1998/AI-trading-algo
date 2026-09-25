import pandas as pd
import numpy as np
from algo_vpin_v2.data_feed import DhanDataFeed
from algo_vpin_v2.macro_features import MacroFeatureEngine
from algo_vpin_v2.vpin import VPINCalculator
from algo_vpin_v2.garch_engine import GARCHEngine, DirectionalSignal
from algo_vpin_v2.ensemble_brain import EnsembleBrain

def test_infinite_trailing_stop():
    feed = DhanDataFeed()
    hist = feed.fetch_historical_bars(days=5)

    macro = MacroFeatureEngine()
    vpin_calc = VPINCalculator()
    garch = GARCHEngine()
    brain = EnsembleBrain()

    trades = []
    pos = {'side': 'FLAT', 'entry_p': 0.0, 'sl': 0.0, 'high_p': 0.0, 'entry_spot': 0.0}

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

        # Check exit on active position
        if pos['side'] != 'FLAT':
            spot_change = (p - pos['entry_spot']) if pos['side'] == 'BUY' else (pos['entry_spot'] - p)
            curr_opt = max(5.0, pos['entry_p'] + spot_change * 0.55)
            pos['high_p'] = max(pos['high_p'], curr_opt)

            # Continuous Dynamic Trailing Stop
            gain_pct = (pos['high_p'] - pos['entry_p']) / pos['entry_p']
            
            # 1. At +20% gain -> SL moves to Breakeven
            if gain_pct >= 0.20 and pos['sl'] < pos['entry_p']:
                pos['sl'] = pos['entry_p']
            
            # 2. Above +30% gain -> Trail SL at (Peak Price - 15% pullback buffer)
            if gain_pct >= 0.30:
                dynamic_trail = round(pos['high_p'] * 0.85, 2)
                if dynamic_trail > pos['sl']:
                    pos['sl'] = dynamic_trail

            # Exit on SL / Trailing SL Hit
            if curr_opt <= pos['sl']:
                pnl = (pos['sl'] - pos['entry_p']) * 65
                ret_pct = (pos['sl'] - pos['entry_p']) / pos['entry_p'] * 100
                trades.append({'pnl': pnl, 'ret_pct': ret_pct, 'peak_gain': gain_pct * 100})
                pos = {'side': 'FLAT', 'entry_p': 0.0, 'sl': 0.0, 'high_p': 0.0, 'entry_spot': 0.0}
            
            # Emergency Model Reversal Exit
            elif not dec.is_vetoed and dec.final_action != DirectionalSignal.HOLD and (
                (pos['side'] == 'BUY' and dec.final_action == DirectionalSignal.SELL) or
                (pos['side'] == 'SELL' and dec.final_action == DirectionalSignal.BUY)
            ):
                pnl = (curr_opt - pos['entry_p']) * 65
                ret_pct = (curr_opt - pos['entry_p']) / pos['entry_p'] * 100
                trades.append({'pnl': pnl, 'ret_pct': ret_pct, 'peak_gain': gain_pct * 100})
                pos = {'side': 'FLAT', 'entry_p': 0.0, 'sl': 0.0, 'high_p': 0.0, 'entry_spot': 0.0}

        # Check entry
        if pos['side'] == 'FLAT' and not dec.is_vetoed and dec.final_action in (DirectionalSignal.BUY, DirectionalSignal.SELL):
            entry_opt = round(max(50.0, p * 0.0045), 2)
            sl_opt = round(entry_opt * (1.0 - 0.20), 2)  # Initial 20% SL
            pos = {
                'side': 'BUY' if dec.final_action == DirectionalSignal.BUY else 'SELL',
                'entry_p': entry_opt,
                'sl': sl_opt,
                'high_p': entry_opt,
                'entry_spot': p
            }

    tdf = pd.DataFrame(trades)
    wins = tdf[tdf['pnl'] > 0]
    losses = tdf[tdf['pnl'] < 0]
    be = tdf[tdf['pnl'] == 0]
    
    print("=" * 70)
    print("   INFINITE DYNAMIC TRAILING STOP BACKTEST (OPEN-ENDED RUNNERS)")
    print("=" * 70)
    print(f"• Total Completed Trades      : {len(tdf)}")
    print(f"• Big Winning Trades (Profit) : {len(wins)} ({len(wins)/len(tdf)*100:.1f}%)")
    print(f"• Losing Trades (20% SL Hits) : {len(losses)} ({len(losses)/len(tdf)*100:.1f}%)")
    print(f"• Breakeven Saves (0% Loss)   : {len(be)} ({len(be)/len(tdf)*100:.1f}%)")
    print("-" * 70)
    print(f"• Average Winning Trade       : +INR {wins['pnl'].mean():,.2f}")
    print(f"• Max Single Winner Caught    : +INR {wins['pnl'].max():,.2f} (+{wins['ret_pct'].max():.1f}%)")
    print(f"• Average Losing Trade        : -INR {abs(losses['pnl'].mean()):,.2f}")
    print(f"• Realized Risk-to-Reward     : 1 : {abs(wins['pnl'].mean()/losses['pnl'].mean()):.2f}")
    print("-" * 70)
    print(f"• Total Gross Profit          : +INR {wins['pnl'].sum():,.2f}")
    print(f"• Total Gross Loss            : -INR {abs(losses['pnl'].sum()):,.2f}")
    print(f"• Realized Profit Factor      : {abs(wins['pnl'].sum()/losses['pnl'].sum()):.2f}")
    print(f"• Net Strategy P&L (5 Days)   : INR {tdf['pnl'].sum():+,.2f}")
    print("=" * 70)

if __name__ == '__main__':
    test_infinite_trailing_stop()
