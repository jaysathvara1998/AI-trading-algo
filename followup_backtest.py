"""
Comparative Backtest on 100% Real Dhan IDX_I Exchange 1-Minute NIFTY Bars
Tests SL = 20% vs 25% with Infinite Continuous Trailing Stop
"""
import pandas as pd
import numpy as np
from algo_vpin_v2.data_feed import DhanDataFeed
from algo_vpin_v2.macro_features import MacroFeatureEngine
from algo_vpin_v2.vpin import VPINCalculator
from algo_vpin_v2.garch_engine import GARCHEngine, DirectionalSignal
from algo_vpin_v2.ensemble_brain import EnsembleBrain


def add_indicators(hist: pd.DataFrame) -> pd.DataFrame:
    hist = hist.copy()
    hist['ema9']  = hist['close'].ewm(span=9,  adjust=False).mean()
    hist['ema21'] = hist['close'].ewm(span=21, adjust=False).mean()
    hist['vwap']  = (hist['close'] * hist['volume']).cumsum() / hist['volume'].cumsum().replace(0, 1)
    delta = hist['close'].diff()
    gain  = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(14).mean().replace(0, 1e-9)
    hist['rsi'] = 100 - (100 / (1 + gain / loss))
    return hist


def run_backtest_on_real_data(initial_sl_pct: float, label: str) -> list:
    feed = DhanDataFeed()
    hist = feed.fetch_historical_bars(days=6)
    hist = add_indicators(hist)

    macro      = MacroFeatureEngine()
    vpin_calc  = VPINCalculator()
    garch      = GARCHEngine()
    brain      = EnsembleBrain()

    trades = []
    pos    = {'side': 'FLAT', 'contract': '', 'entry_p': 0.0, 'sl': 0.0,
              'high_p': 0.0, 'entry_spot': 0.0, 'entry_time': None}
    cooldown_until = None

    for idx, row in hist.iterrows():
        p     = float(row['close'])
        h     = float(row.get('high', p))
        l     = float(row.get('low',  p))
        v     = float(row['volume'])
        ts    = row['timestamp']
        ema9  = float(row.get('ema9',  p))
        ema21 = float(row.get('ema21', p))
        vwap  = float(row.get('vwap',  p))
        rsi   = float(row.get('rsi',  50.0))

        mst  = macro.update_1min_bar(p, h, l)
        vres = vpin_calc.process_bar(p, v, ts)
        gres = garch.add_bar(p, ts)

        if idx < 50:
            continue

        p_delta = p - float(hist.iloc[idx-1]['close'])
        sigma   = gres.sigma_next if gres else 0.001
        mu      = gres.mu_next    if gres else 0.0
        dec     = brain.evaluate_signal(
            gres.signal if gres else DirectionalSignal.HOLD,
            p_delta, sigma, mu, vres.vpin, mst
        )

        # ── Manage open position ──────────────────────────────────────────
        if pos['side'] != 'FLAT':
            spot_change = (p - pos['entry_spot']) if pos['side'] == 'BUY' else (pos['entry_spot'] - p)
            curr_opt    = max(5.0, pos['entry_p'] + spot_change * 0.55)
            pos['high_p'] = max(pos['high_p'], curr_opt)
            gain_pct    = (pos['high_p'] - pos['entry_p']) / pos['entry_p']

            # Breakeven at (initial_sl_pct * 0.9) gain
            be_trigger = initial_sl_pct * 0.9
            if gain_pct >= be_trigger and pos['sl'] < pos['entry_p']:
                pos['sl'] = pos['entry_p']

            # Continuous trailing above +25%: trail at Peak - 12%
            if gain_pct >= 0.25:
                trail = round(pos['high_p'] * 0.88, 2)
                if trail > pos['sl']:
                    pos['sl'] = trail

            # Exit on EOD, SL or model reversal
            is_eod = (ts.hour == 15 and ts.minute >= 20)
            reversal = not dec.is_vetoed and dec.final_action != DirectionalSignal.HOLD and (
                (pos['side'] == 'BUY'  and dec.final_action == DirectionalSignal.SELL) or
                (pos['side'] == 'SELL' and dec.final_action == DirectionalSignal.BUY)
            )

            if curr_opt <= pos['sl'] or is_eod or reversal:
                exit_price = (pos['sl'] if not is_eod else curr_opt) if not reversal else curr_opt
                pnl        = (exit_price - pos['entry_p']) * 65
                ret_pct    = (exit_price - pos['entry_p']) / pos['entry_p'] * 100
                if is_eod:
                    trigger = 'EOD_SQUAREOFF'
                elif reversal:
                    trigger = 'MODEL_REVERSAL'
                elif exit_price >= pos['entry_p']:
                    trigger = 'TRAILING_SL_EXIT'
                else:
                    trigger = 'INITIAL_SL_HIT'

                trades.append({
                    'Date':        ts.strftime('%d-%b-%Y'),
                    'Contract':    pos['contract'],
                    'Entry Time':  pos['entry_time'].strftime('%H:%M:%S'),
                    'Exit Time':   ts.strftime('%H:%M:%S'),
                    'Spot Entry':  f"{pos['entry_spot']:,.2f}",
                    'Spot Exit':   f"{p:,.2f}",
                    'Entry Prem':  f"INR {pos['entry_p']:.2f}",
                    'SL Price':    f"INR {pos['entry_p'] * (1 - initial_sl_pct):.2f}",
                    'Peak Prem':   f"INR {pos['high_p']:.2f} (+{gain_pct*100:.1f}%)",
                    'Exit Prem':   f"INR {exit_price:.2f}",
                    'PnL (INR)':   pnl,
                    'Return %':    f"{ret_pct:+.1f}%",
                    'Exit Trigger': trigger
                })

                if trigger == 'INITIAL_SL_HIT':
                    cooldown_until = ts + pd.Timedelta(minutes=10)

                pos = {'side': 'FLAT', 'contract': '', 'entry_p': 0.0,
                       'sl': 0.0, 'high_p': 0.0, 'entry_spot': 0.0, 'entry_time': None}

        # ── Entry: Market Hours 09:20-15:00, Momentum-Aligned ────────────
        can_enter  = ((ts.hour == 9 and ts.minute >= 20) or (10 <= ts.hour < 15))
        in_cooldown = (cooldown_until is not None and ts < cooldown_until)

        if pos['side'] == 'FLAT' and can_enter and not in_cooldown and not dec.is_vetoed:
            bullish = (p > vwap and ema9 > ema21 and rsi > 52 and dec.final_action == DirectionalSignal.BUY)
            bearish = (p < vwap and ema9 < ema21 and rsi < 48 and dec.final_action == DirectionalSignal.SELL)

            if bullish or bearish:
                atm        = round(p / 50) * 50
                opt_type   = 'CALL' if bullish else 'PUT'
                entry_opt  = round(max(50.0, p * 0.0045), 2)
                sl_price   = round(entry_opt * (1 - initial_sl_pct), 2)

                pos = {
                    'side':       'BUY' if bullish else 'SELL',
                    'contract':   f"NIFTY {atm} {opt_type}",
                    'entry_p':    entry_opt,
                    'sl':         sl_price,
                    'high_p':     entry_opt,
                    'entry_spot': p,
                    'entry_time': ts
                }

    return trades


def print_report(trades: list, label: str, sl_pct: float):
    tdf = pd.DataFrame(trades)
    if tdf.empty:
        print(f"\n[{label}] No trades generated.")
        return

    print(f"\n{'=' * 130}")
    print(f"   {label} - INITIAL SL: {int(sl_pct*100)}% | Real Dhan Exchange 1-Min Bars")
    print(f"{'=' * 130}")

    for date, group in tdf.groupby('Date', sort=False):
        day_pnl  = group['PnL (INR)'].sum()
        pnl_str  = f"+INR {day_pnl:,.2f}" if day_pnl >= 0 else f"-INR {abs(day_pnl):,.2f}"
        wins_day = len(group[group['PnL (INR)'] > 0])
        print(f"\n  [DATE: {date}] | Trades: {len(group)} (Wins: {wins_day}) | Day P&L: {pnl_str}")
        print(f"  {'-' * 127}")
        for i, t in group.reset_index(drop=True).iterrows():
            t_pnl = f"+INR {t['PnL (INR)']:,.2f}" if t['PnL (INR)'] >= 0 else f"-INR {abs(t['PnL (INR)']):,.2f}"
            print(f"  #{i+1:02d} | {t['Entry Time']} -> {t['Exit Time']} IST | {t['Contract']:18s} | "
                  f"Spot: {t['Spot Entry']} -> {t['Spot Exit']} | "
                  f"Entry: {t['Entry Prem']} | SL@: {t['SL Price']} | "
                  f"Peak: {t['Peak Prem']} | Exit: {t['Exit Prem']} | "
                  f"PnL: {t_pnl:14s} ({t['Return %']:6s}) | {t['Exit Trigger']}")

    wins   = tdf[tdf['PnL (INR)'] > 0]
    losses = tdf[tdf['PnL (INR)'] < 0]
    be     = tdf[tdf['PnL (INR)'] == 0]
    total  = len(tdf)
    gw = wins['PnL (INR)'].sum()   if not wins.empty   else 0.0
    gl = abs(losses['PnL (INR)'].sum()) if not losses.empty else 0.0
    pf = gw / gl if gl > 0 else 99.0

    print(f"\n  {'-' * 80}")
    print(f"  Total Trades: {total}  |  Wins: {len(wins)} ({len(wins)/total*100:.1f}%)  |  "
          f"Losses: {len(losses)} ({len(losses)/total*100:.1f}%)  |  Breakeven: {len(be)}")
    print(f"  Avg Win: +INR {wins['PnL (INR)'].mean():,.2f}" if not wins.empty else "  Avg Win: INR 0", end="  |  ")
    print(f"Avg Loss: -INR {abs(losses['PnL (INR)'].mean()):,.2f}" if not losses.empty else "Avg Loss: INR 0")
    print(f"  Max Single Win: +INR {wins['PnL (INR)'].max():,.2f}" if not wins.empty else "  Max Win: 0")
    print(f"  Profit Factor: {pf:.2f}  |  Net P&L: INR {tdf['PnL (INR)'].sum():+,.2f}  |  "
          f"Return on 50k: {tdf['PnL (INR)'].sum()/50000*100:+.1f}%")


if __name__ == '__main__':
    print("Running backtest on 100% Real Dhan Exchange 1-Minute NIFTY Bars...")
    trades_20 = run_backtest_on_real_data(0.20, "SL = 20%")
    trades_25 = run_backtest_on_real_data(0.25, "SL = 25%")

    print_report(trades_20, "BACKTEST A: SL 20% | Breakeven @ +18% | Trail @ +25%", 0.20)
    print_report(trades_25, "BACKTEST B: SL 25% | Breakeven @ +22% | Trail @ +25%", 0.25)

    # Quick side-by-side summary
    tdf20 = pd.DataFrame(trades_20)
    tdf25 = pd.DataFrame(trades_25)
    print(f"\n{'=' * 80}")
    print("                  SIDE-BY-SIDE COMPARISON")
    print(f"{'=' * 80}")
    print(f"{'Metric':<35} {'SL 20%':>20} {'SL 25%':>20}")
    print(f"{'-' * 80}")
    for label, df in [("SL 20%", tdf20), ("SL 25%", tdf25)]:
        pass

    for (metric, v20, v25) in [
        ("Total Trades",       len(tdf20), len(tdf25)),
        ("Win Count",          len(tdf20[tdf20['PnL (INR)']>0]), len(tdf25[tdf25['PnL (INR)']>0])),
        ("Loss Count",         len(tdf20[tdf20['PnL (INR)']<0]), len(tdf25[tdf25['PnL (INR)']<0])),
        ("Breakeven (TSL Save)",len(tdf20[tdf20['PnL (INR)']==0]),len(tdf25[tdf25['PnL (INR)']==0])),
    ]:
        print(f"  {metric:<33} {str(v20):>20} {str(v25):>20}")

    v20_pnl = tdf20['PnL (INR)'].sum()
    v25_pnl = tdf25['PnL (INR)'].sum()
    gl20 = abs(tdf20[tdf20['PnL (INR)']<0]['PnL (INR)'].sum()) if len(tdf20[tdf20['PnL (INR)']<0])>0 else 1
    gl25 = abs(tdf25[tdf25['PnL (INR)']<0]['PnL (INR)'].sum()) if len(tdf25[tdf25['PnL (INR)']<0])>0 else 1
    gw20 = tdf20[tdf20['PnL (INR)']>0]['PnL (INR)'].sum()
    gw25 = tdf25[tdf25['PnL (INR)']>0]['PnL (INR)'].sum()
    for (metric, v20, v25) in [
        ("Net P&L (5 Days)",       f"INR {v20_pnl:+,.2f}", f"INR {v25_pnl:+,.2f}"),
        ("Profit Factor",          f"{gw20/gl20:.2f}", f"{gw25/gl25:.2f}"),
        ("Return on 50k Capital",  f"{v20_pnl/50000*100:+.1f}%", f"{v25_pnl/50000*100:+.1f}%"),
    ]:
        print(f"  {metric:<33} {str(v20):>20} {str(v25):>20}")
    print(f"{'=' * 80}")
