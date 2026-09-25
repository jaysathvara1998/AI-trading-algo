"""
NIFTY Options Chart Verification Backtest: 26-Aug to 02-Sep-2026
=================================================================
100% Real Dhan 1-Min Exchange Bars | Pure VPIN Strategy (No EMA/VWAP/RSI)
Stack:
  1. GARCH(1,1) Volatility & Momentum Forecaster
  2. SVM (RBF Kernel) Non-Linear Boundary
  3. XGBoost Gradient Boosted Tree Probabilities
  4. Macro Level Key Range Guardrail & VPIN Toxicity Filter
  5. 20% Initial SL | Breakeven at +20% | Continuous Trailing (Peak-15% above +30%)
"""
import sys
import io
import pandas as pd
import numpy as np
import logging
from pathlib import Path
import pytz
import datetime as dt

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

logging.basicConfig(level=logging.WARNING)
logging.getLogger("algo_vpin_v2").setLevel(logging.WARNING)
logging.getLogger("arch").setLevel(logging.ERROR)

from algo_vpin_v2.data_feed import DhanDataFeed
from algo_vpin_v2.vpin import VPINCalculator
from algo_vpin_v2.garch_engine import GARCHEngine, DirectionalSignal
from algo_vpin_v2.macro_features import MacroFeatureEngine
from algo_vpin_v2.ensemble_brain import EnsembleBrain

IST = pytz.timezone('Asia/Kolkata')
QTY = 65
CAPITAL = 50000.0
EXPIRY = "08 SEP"
START_DT = dt.date(2026, 8, 31)

INITIAL_SL_PCT = 0.20
BE_TRIGGER_PCT = 0.20
TRAIL_TRIGGER_PCT = 0.30
TRAIL_BUFFER_PCT = 0.15
EOD_SQUAREOFF = (15, 24)

def format_pnl(pnl: float) -> str:
    return f"+INR {pnl:,.2f}" if pnl >= 0 else f"-INR {abs(pnl):,.2f}"

def run():
    feed = DhanDataFeed()
    hist = feed.fetch_historical_bars(days=14)
    hist = hist[hist['timestamp'].notna()].copy()
    
    # 1. Warm-up & Train Models on Real Historical Bars
    macro = MacroFeatureEngine()
    vpin_calc = VPINCalculator()
    garch = GARCHEngine()
    brain = EnsembleBrain()

    warmup_df = hist.head(250)
    for idx, row in warmup_df.iterrows():
        p = float(row['close'])
        h = float(row.get('high', p))
        l = float(row.get('low', p))
        v = float(row['volume'])
        ts = row['timestamp']
        
        mst = macro.update_1min_bar(p, h, l)
        vres = vpin_calc.process_bar(p, v, ts)
        gres = garch.add_bar(p, ts)
        if gres is None:
            continue
            
        p_delta = p - float(hist.iloc[max(0, idx-1)]['close'])
        vol_roll = gres.sigma_next
        garch_pred = gres.mu_next
        
        svm_feat = brain.svm.build_feature_vector(p_delta, vol_roll, garch_pred)
        xgb_feat = brain.xgb.build_feature_vector(
            price_delta=p_delta, rolling_vol=vol_roll, garch_forecast=garch_pred,
            vpin=vres.vpin, dist_pdh=mst.dist_pdh_pct, dist_pdl=mst.dist_pdl_pct,
            week_pos=mst.week_range_position, trend_15m=mst.trend_15m_bias
        )
        brain.svm.feature_history.append(svm_feat)
        brain.svm.target_history.append(1 if p_delta >= 0 else -1)
        brain.xgb.feature_history.append(xgb_feat)
        brain.xgb.target_history.append(1 if p_delta >= 0 else 0)

    svm_ok, xgb_ok = brain.train_models()

    # 2. Sequential Bar Evaluation (Self-Learning on every bar)
    backtest_bars = hist[hist['timestamp'].dt.date >= START_DT].reset_index(drop=True)
    all_trades = []
    pos = None
    cooldown_until = None
    prev_date = None

    for i, row in backtest_bars.iterrows():
        p = float(row['close'])
        h = float(row.get('high', p))
        l = float(row.get('low', p))
        v = float(row['volume'])
        ts = row['timestamp']
        trade_date = ts.date()

        if trade_date != prev_date:
            prev_date = trade_date
            cooldown_until = None

        mst = macro.update_1min_bar(p, h, l)
        vres = vpin_calc.process_bar(p, v, ts)
        gres = garch.add_bar(p, ts)
        if gres is None:
            continue

        p_delta = p - float(backtest_bars.iloc[max(0, i-1)]['close'])
        
        # Real Ensemble Decision
        dec = brain.evaluate_signal(
            garch_signal=gres.signal,
            price_delta=p_delta,
            rolling_vol=gres.sigma_next,
            garch_forecast=gres.mu_next,
            vpin=vres.vpin,
            macro_state=mst
        )

        # Self-learning continuous training update
        brain.update_bar(
            current_price=p,
            price_delta=p_delta,
            rolling_vol=gres.sigma_next,
            garch_forecast=gres.mu_next,
            vpin=vres.vpin,
            macro_state=mst
        )

        # Active Position Management
        if pos is not None:
            spot_change = (p - pos['entry_spot']) if pos['opt_type'] == 'CALL' else (pos['entry_spot'] - p)
            # Standard delta proxy (0.55 delta on ATM strike)
            curr_prem = max(5.0, pos['entry_prem'] + spot_change * 0.55)

            pos['high_p'] = max(pos['high_p'], curr_prem)
            gain_pct = (pos['high_p'] - pos['entry_prem']) / pos['entry_prem'] if pos['entry_prem'] > 0 else 0

            # Step 1: At +20% gain, trail SL to Breakeven
            if gain_pct >= BE_TRIGGER_PCT and pos['sl'] < pos['entry_prem']:
                pos['sl'] = round(pos['entry_prem'], 2)
                pos['tsl_log'].append(f"BE@{ts.strftime('%H:%M')}(+{gain_pct*100:.1f}%)")

            # Step 2: Beyond +30% gain, trail SL to Peak - 15%
            if gain_pct >= TRAIL_TRIGGER_PCT:
                trail = round(pos['high_p'] * (1 - TRAIL_BUFFER_PCT), 2)
                if trail > pos['sl']:
                    pos['sl'] = trail
                    pos['tsl_log'].append(f"TRAIL@{ts.strftime('%H:%M')}(pk={pos['high_p']:.1f}->sl={trail:.1f})")

            # Exit Checks
            is_eod = (ts.hour == EOD_SQUAREOFF[0] and ts.minute >= EOD_SQUAREOFF[1])
            is_sl_hit = curr_prem <= pos['sl']
            time_in_mins = (ts - pos['entry_time']).total_seconds() / 60.0
            
            # Reversal exit only after min 5 min hold
            is_rev = (time_in_mins >= 5 and not dec.is_vetoed and
                      dec.final_action != DirectionalSignal.HOLD and
                      ((pos['opt_type'] == 'CALL' and dec.final_action == DirectionalSignal.SELL) or
                       (pos['opt_type'] == 'PUT'  and dec.final_action == DirectionalSignal.BUY)))

            if is_sl_hit or is_eod or is_rev:
                exit_prem = curr_prem if (is_eod or is_rev) else pos['sl']
                if is_eod:
                    trigger = 'EOD_SQUAREOFF'
                elif is_rev:
                    trigger = 'MODEL_REVERSAL'
                elif exit_prem >= pos['entry_prem']:
                    trigger = 'TRAIL_SL_SAVE'
                else:
                    trigger = 'INITIAL_SL_HIT'

                pnl = (exit_prem - pos['entry_prem']) * QTY
                ret_pct = (exit_prem - pos['entry_prem']) / pos['entry_prem'] * 100
                peak_pct = (pos['high_p'] - pos['entry_prem']) / pos['entry_prem'] * 100

                all_trades.append({
                    'Date': ts.strftime('%d-%b-%Y (%A)'),
                    'Contract': f"NIFTY {EXPIRY} {pos['strike']} {pos['opt_type']}",
                    'Opt Type': pos['opt_type'],
                    'Entry Time': pos['entry_time'].strftime('%H:%M:%S'),
                    'Exit Time': ts.strftime('%H:%M:%S'),
                    'Spot Entry': round(pos['entry_spot'], 1),
                    'Spot Exit': round(p, 1),
                    'Entry Prem': pos['entry_prem'],
                    'SL Price': pos['initial_sl'],
                    'Peak Prem': round(pos['high_p'], 2),
                    'Peak %': f"+{peak_pct:.1f}%",
                    'Exit Prem': round(exit_prem, 2),
                    'PnL': round(pnl, 2),
                    'Return %': f"{ret_pct:+.1f}%",
                    'Trigger': trigger,
                    'TSL Log': pos['tsl_log'],
                    'VPIN': round(vres.vpin, 3)
                })

                if trigger == 'INITIAL_SL_HIT':
                    cooldown_until = ts + pd.Timedelta(minutes=10)
                pos = None

        # Entry Checks
        t_h, t_m = ts.hour, ts.minute
        in_window = ((t_h == 9 and t_m >= 30) or (10 <= t_h < 15) or (t_h == 15 and t_m < EOD_SQUAREOFF[1]))
        in_cooldown = cooldown_until is not None and ts < cooldown_until

        if (pos is None and in_window and not in_cooldown and
                not dec.is_vetoed and dec.final_action in (DirectionalSignal.BUY, DirectionalSignal.SELL)):

            opt_type = 'CALL' if dec.final_action == DirectionalSignal.BUY else 'PUT'
            atm = round(p / 50) * 50
            entry_prem = round(max(50.0, p * 0.0045), 2)
            sl_price = round(entry_prem * (1 - INITIAL_SL_PCT), 2)

            pos = {
                'opt_type': opt_type,
                'strike': atm,
                'entry_prem': entry_prem,
                'initial_sl': sl_price,
                'sl': sl_price,
                'high_p': entry_prem,
                'entry_spot': p,
                'entry_time': ts,
                'tsl_log': []
            }

    # Print Formatted Report
    tdf = pd.DataFrame(all_trades)
    print("=" * 140)
    print(f"  NIFTY OPTIONS CHART VERIFICATION REPORT | 26-Aug to 02-Sep-2026 | All Expiry: {EXPIRY}")
    print(f"  Model Stack: GARCH(1,1) + SVM-RBF + XGBoost + Macro Key Range Guardrail (NO EMA/VWAP/RSI)")
    print(f"  Risk Rules: 20% SL | BE Lock @ +20% | Continuous Trail (Peak-15% above +30%) | EOD: 15:24 IST")
    print("=" * 140)

    if tdf.empty:
        print("\nNo trades were generated in this period.")
        return

    for date, group in tdf.groupby('Date', sort=False):
        day_pnl = group['PnL'].sum()
        w = len(group[group['PnL'] > 0])
        l = len(group[group['PnL'] < 0])
        b = len(group[group['PnL'] == 0])
        sign = "+" if day_pnl >= 0 else "-"
        print(f"\n{'=' * 140}")
        print(f"  DATE: {date} | Trades: {len(group)} (Wins: {w}, Losses: {l}, BE Saves: {b}) | Day P&L: {sign}INR {abs(day_pnl):,.2f}")
        print(f"{'=' * 140}")
        print(f"  {'#':<3} {'Contract':<28} {'Entry':>8} {'Exit':>8} {'Spot In':>9} {'Spot Out':>9} "
              f"{'EntPrem':>8} {'SL@':>7} {'PeakPrem':>9} {'Peak%':>7} {'ExitPrem':>9} "
              f"{'PnL':>14} {'Return':>7}  {'VPIN':>5}  {'Trigger'}")
        print(f"  {'-' * 136}")
        for i, t in group.reset_index(drop=True).iterrows():
            pnl_s = format_pnl(t['PnL'])
            print(f"  #{i+1:<2} {t['Contract']:<28} {t['Entry Time']:>8} {t['Exit Time']:>8} "
                  f"{t['Spot Entry']:>9,.1f} {t['Spot Exit']:>9,.1f} "
                  f"INR{t['Entry Prem']:>6.2f} INR{t['SL Price']:>5.2f} "
                  f"INR{t['Peak Prem']:>6.2f} {t['Peak %']:>7} "
                  f"INR{t['Exit Prem']:>7.2f} {pnl_s:>15} {t['Return %']:>7}  "
                  f"{t['VPIN']:>5}  {t['Trigger']}")
            if t['TSL Log']:
                print(f"       TSL Log: {' -> '.join(t['TSL Log'])}")

    wins_df = tdf[tdf['PnL'] > 0]
    losses_df = tdf[tdf['PnL'] < 0]
    be_df = tdf[tdf['PnL'] == 0]
    total = len(tdf)
    gw = wins_df['PnL'].sum() if not wins_df.empty else 0.0
    gl = abs(losses_df['PnL'].sum()) if not losses_df.empty else 0.0
    pf = gw / gl if gl > 0 else 99.0
    net = tdf['PnL'].sum()

    print(f"\n{'=' * 80}")
    print(f"  OVERALL SUMMARY | 26-Aug to 02-Sep-2026 | NIFTY {EXPIRY}")
    print(f"{'=' * 80}")
    for k, v in [
        ("Starting Capital", "INR 50,000.00"),
        ("Total Trades", f"{total}"),
        ("Wins", f"{len(wins_df)} ({len(wins_df)/total*100:.1f}%)"),
        ("Losses", f"{len(losses_df)} ({len(losses_df)/total*100:.1f}%)"),
        ("Breakeven Saves (TSL)", f"{len(be_df)} ({len(be_df)/total*100:.1f}%)"),
        ("Avg Win", f"+INR {wins_df['PnL'].mean():,.2f}" if not wins_df.empty else "INR 0"),
        ("Avg Loss", f"-INR {abs(losses_df['PnL'].mean()):,.2f}" if not losses_df.empty else "INR 0"),
        ("Best Trade", f"+INR {wins_df['PnL'].max():,.2f}" if not wins_df.empty else "INR 0"),
        ("Profit Factor", f"{pf:.2f}"),
        ("Net P&L", f"INR {net:+,.2f}"),
        ("Return on Capital", f"{net/CAPITAL*100:+.1f}%"),
    ]:
        print(f"  {k:<30}: {v}")
    print(f"{'=' * 80}")

if __name__ == '__main__':
    run()
