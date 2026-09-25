import pandas as pd
import numpy as np
from algo_vpin_v2.data_feed import DhanDataFeed
from algo_vpin_v2.macro_features import MacroFeatureEngine
from algo_vpin_v2.vpin import VPINCalculator
from algo_vpin_v2.garch_engine import GARCHEngine, DirectionalSignal
from algo_vpin_v2.ensemble_brain import EnsembleBrain

def run_comparative_backtest():
    feed = DhanDataFeed()
    hist = feed.fetch_historical_bars(days=5)
    print(f"Loaded {len(hist)} historical bars. Running comparative SL backtests...\n")

    sl_levels = [0.15, 0.20, 0.25, 0.30, 0.35]
    results = []

    for sl_pct in sl_levels:
        tp_pct = sl_pct * 2.0  # Strict 1:2 Risk to Reward
        macro = MacroFeatureEngine()
        vpin_calc = VPINCalculator()
        garch = GARCHEngine()
        brain = EnsembleBrain()

        trades = []
        pos = {'side': 'FLAT', 'entry_p': 0.0, 'sl': 0.0, 'tp': 0.0, 'high_p': 0.0, 'entry_spot': 0.0}

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

            # Active position exit check
            if pos['side'] != 'FLAT':
                spot_change = (p - pos['entry_spot']) if pos['side'] == 'BUY' else (pos['entry_spot'] - p)
                curr_opt = max(5.0, pos['entry_p'] + spot_change * 0.55)
                pos['high_p'] = max(pos['high_p'], curr_opt)

                # Trailing Stop-Loss: Move to breakeven at +15% profit
                if pos['high_p'] >= pos['entry_p'] * (1.0 + sl_pct * 0.75) and pos['sl'] < pos['entry_p']:
                    pos['sl'] = pos['entry_p']

                if curr_opt <= pos['sl']:
                    pnl = (pos['sl'] - pos['entry_p']) * 65
                    trades.append({'pnl': pnl, 'reason': 'STOP_LOSS / TSL'})
                    pos = {'side': 'FLAT', 'entry_p': 0.0, 'sl': 0.0, 'tp': 0.0, 'high_p': 0.0, 'entry_spot': 0.0}
                elif curr_opt >= pos['tp']:
                    pnl = (pos['tp'] - pos['entry_p']) * 65
                    trades.append({'pnl': pnl, 'reason': 'TARGET_HIT'})
                    pos = {'side': 'FLAT', 'entry_p': 0.0, 'sl': 0.0, 'tp': 0.0, 'high_p': 0.0, 'entry_spot': 0.0}
                elif not dec.is_vetoed and dec.final_action != DirectionalSignal.HOLD and (
                    (pos['side'] == 'BUY' and dec.final_action == DirectionalSignal.SELL) or
                    (pos['side'] == 'SELL' and dec.final_action == DirectionalSignal.BUY)
                ):
                    pnl = (curr_opt - pos['entry_p']) * 65
                    trades.append({'pnl': pnl, 'reason': 'REVERSAL'})
                    pos = {'side': 'FLAT', 'entry_p': 0.0, 'sl': 0.0, 'tp': 0.0, 'high_p': 0.0, 'entry_spot': 0.0}

            # Entry check
            if pos['side'] == 'FLAT' and not dec.is_vetoed and dec.final_action in (DirectionalSignal.BUY, DirectionalSignal.SELL):
                entry_opt = round(max(50.0, p * 0.0045), 2)
                sl_opt = round(entry_opt * (1.0 - sl_pct), 2)
                tp_opt = round(entry_opt * (1.0 + tp_pct), 2)
                pos = {
                    'side': 'BUY' if dec.final_action == DirectionalSignal.BUY else 'SELL',
                    'entry_p': entry_opt,
                    'sl': sl_opt,
                    'tp': tp_opt,
                    'high_p': entry_opt,
                    'entry_spot': p
                }

        tdf = pd.DataFrame(trades)
        wins = tdf[tdf['pnl'] > 0]
        losses = tdf[tdf['pnl'] < 0]
        be = tdf[tdf['pnl'] == 0]
        total_t = len(tdf)
        win_rate = (len(wins) / total_t * 100) if total_t > 0 else 0
        gross_win = wins['pnl'].sum() if not wins.empty else 0.0
        gross_loss = abs(losses['pnl'].sum()) if not losses.empty else 0.0
        profit_factor = (gross_win / gross_loss) if gross_loss > 0 else 99.0
        net_pnl = tdf['pnl'].sum() if not tdf.empty else 0.0
        avg_loss = abs(losses['pnl'].mean()) if not losses.empty else 0.0
        avg_win = wins['pnl'].mean() if not wins.empty else 0.0

        results.append({
            'SL %': f"{int(sl_pct*100)}%",
            'TP %': f"{int(tp_pct*100)}%",
            'Total Trades': total_t,
            'Wins': len(wins),
            'Losses': len(losses),
            'Breakeven': len(be),
            'Win Rate': f"{win_rate:.1f}%",
            'Avg Win': f"+INR {avg_win:,.0f}",
            'Avg Loss': f"-INR {avg_loss:,.0f}",
            'Profit Factor': f"{profit_factor:.2f}",
            'Net PnL (5-Day)': f"INR {net_pnl:+,.0f}"
        })

    res_df = pd.DataFrame(results)
    print("=" * 80)
    print("           COMPARATIVE STOP-LOSS BACKTEST MATRIX (1:2 RR)")
    print("=" * 80)
    print(res_df.to_string(index=False))
    print("=" * 80)

if __name__ == '__main__':
    run_comparative_backtest()
