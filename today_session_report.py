"""
Today's Complete Session Report (02-Sep-2026)
Runs VPIN Engine on today's actual 1-minute exchange candles from 09:15 to current time.
"""
import sys, io
import pandas as pd
import numpy as np
import datetime as dt
import pytz

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from algo_vpin_v2.data_feed import DhanDataFeed
from algo_vpin_v2.vpin import VPINCalculator
from algo_vpin_v2.garch_engine import GARCHEngine, DirectionalSignal
from algo_vpin_v2.macro_features import MacroFeatureEngine
from algo_vpin_v2.ensemble_brain import EnsembleBrain

IST = pytz.timezone('Asia/Kolkata')
QTY = 65
CAPITAL = 50000.0
EXPIRY = "08 SEP"

INITIAL_SL_PCT = 0.20
BE_TRIGGER_PCT = 0.20
TRAIL_TRIGGER_PCT = 0.30
TRAIL_BUFFER_PCT = 0.15

feed = DhanDataFeed()
hist = feed.fetch_historical_bars(days=6)
hist = hist[hist['timestamp'].notna()].copy()

# 1. Warm-up
macro = MacroFeatureEngine()
vpin_calc = VPINCalculator()
garch = GARCHEngine()
brain = EnsembleBrain()

warmup_df = hist[hist['timestamp'].dt.date < dt.date(2026, 9, 2)]
for idx, row in warmup_df.iterrows():
    p = float(row['close'])
    h = float(row.get('high', p))
    l = float(row.get('low', p))
    v = float(row['volume'])
    ts = row['timestamp']
    mst = macro.update_1min_bar(p, h, l)
    vres = vpin_calc.process_bar(p, v, ts)
    gres = garch.add_bar(p, ts)
    if gres is None: continue
    p_delta = p - float(hist.iloc[max(0, idx-1)]['close'])
    svm_feat = brain.svm.build_feature_vector(p_delta, gres.sigma_next, gres.mu_next)
    xgb_feat = brain.xgb.build_feature_vector(
        price_delta=p_delta, rolling_vol=gres.sigma_next, garch_forecast=gres.mu_next,
        vpin=vres.vpin, dist_pdh=mst.dist_pdh_pct, dist_pdl=mst.dist_pdl_pct,
        week_pos=mst.week_range_position, trend_15m=mst.trend_15m_bias
    )
    brain.svm.feature_history.append(svm_feat)
    brain.svm.target_history.append(1 if p_delta >= 0 else -1)
    brain.xgb.feature_history.append(xgb_feat)
    brain.xgb.target_history.append(1 if p_delta >= 0 else 0)

brain.train_models()

# 2. Today's Bars
today_bars = hist[hist['timestamp'].dt.date == dt.date(2026, 9, 2)].reset_index(drop=True)
print(f"Loaded {len(today_bars)} real exchange bars for today (02-Sep-2026): {today_bars.iloc[0]['timestamp'].strftime('%H:%M')} to {today_bars.iloc[-1]['timestamp'].strftime('%H:%M')} IST")
print(f"NIFTY Range Today: Open={today_bars.iloc[0]['open']:,.1f}, Low={today_bars['low'].min():,.1f}, High={today_bars['high'].max():,.1f}, Current={today_bars.iloc[-1]['close']:,.1f}\n")

trades = []
pos = None
cooldown_until = None

for i, row in today_bars.iterrows():
    p = float(row['close'])
    h = float(row.get('high', p))
    l = float(row.get('low', p))
    v = float(row['volume'])
    ts = row['timestamp']

    mst = macro.update_1min_bar(p, h, l)
    vres = vpin_calc.process_bar(p, v, ts)
    gres = garch.add_bar(p, ts)
    if gres is None: continue

    p_delta = p - float(today_bars.iloc[max(0, i-1)]['close'])
    dec = brain.evaluate_signal(gres.signal, p_delta, gres.sigma_next, gres.mu_next, vres.vpin, mst)
    brain.update_bar(p, p_delta, gres.sigma_next, gres.mu_next, vres.vpin, mst)

    if pos is not None:
        spot_chg = (p - pos['entry_spot']) if pos['opt_type'] == 'CALL' else (pos['entry_spot'] - p)
        curr_prem = max(5.0, pos['entry_prem'] + spot_chg * 0.55)
        pos['high_p'] = max(pos['high_p'], curr_prem)
        gain_pct = (pos['high_p'] - pos['entry_prem']) / pos['entry_prem'] if pos['entry_prem'] > 0 else 0

        if gain_pct >= BE_TRIGGER_PCT and pos['sl'] < pos['entry_prem']:
            pos['sl'] = round(pos['entry_prem'], 2)
            pos['tsl_log'].append(f"BE@{ts.strftime('%H:%M')}(+{gain_pct*100:.1f}%)")

        if gain_pct >= TRAIL_TRIGGER_PCT:
            trail = round(pos['high_p'] * (1 - TRAIL_BUFFER_PCT), 2)
            if trail > pos['sl']:
                pos['sl'] = trail
                pos['tsl_log'].append(f"TRAIL@{ts.strftime('%H:%M')}(pk={pos['high_p']:.1f}->sl={trail:.1f})")

        is_eod = (ts.hour == 15 and ts.minute >= 24)
        is_sl_hit = curr_prem <= pos['sl']
        time_in = (ts - pos['entry_time']).total_seconds() / 60.0
        is_rev = (time_in >= 5 and not dec.is_vetoed and dec.final_action != DirectionalSignal.HOLD and
                  ((pos['opt_type'] == 'CALL' and dec.final_action == DirectionalSignal.SELL) or
                   (pos['opt_type'] == 'PUT' and dec.final_action == DirectionalSignal.BUY)))

        if is_sl_hit or is_eod or is_rev:
            exit_prem = curr_prem if (is_eod or is_rev) else pos['sl']
            trigger = 'EOD_SQUAREOFF' if is_eod else ('MODEL_REVERSAL' if is_rev else ('TRAIL_SL_SAVE' if exit_prem >= pos['entry_prem'] else 'INITIAL_SL_HIT'))
            pnl = (exit_prem - pos['entry_prem']) * QTY
            ret_pct = (exit_prem - pos['entry_prem']) / pos['entry_prem'] * 100
            trades.append({
                'Contract': f"NIFTY {EXPIRY} {pos['strike']} {pos['opt_type']}",
                'Entry Time': pos['entry_time'].strftime('%H:%M:%S'),
                'Exit Time': ts.strftime('%H:%M:%S'),
                'Spot In': round(pos['entry_spot'], 1),
                'Spot Out': round(p, 1),
                'Entry Prem': pos['entry_prem'],
                'SL Price': pos['initial_sl'],
                'Peak Prem': round(pos['high_p'], 2),
                'Peak %': f"+{(pos['high_p']-pos['entry_prem'])/pos['entry_prem']*100:.1f}%",
                'Exit Prem': round(exit_prem, 2),
                'PnL': round(pnl, 2),
                'Return': f"{ret_pct:+.1f}%",
                'Trigger': trigger,
                'TSL Log': pos['tsl_log'],
                'VPIN': round(vres.vpin, 3)
            })
            if trigger == 'INITIAL_SL_HIT':
                cooldown_until = ts + pd.Timedelta(minutes=10)
            pos = None

    t_h, t_m = ts.hour, ts.minute
    in_window = ((t_h == 9 and t_m >= 30) or (10 <= t_h < 15))
    in_cooldown = cooldown_until is not None and ts < cooldown_until

    if pos is None and in_window and not in_cooldown and not dec.is_vetoed and dec.final_action in (DirectionalSignal.BUY, DirectionalSignal.SELL):
        opt_type = 'CALL' if dec.final_action == DirectionalSignal.BUY else 'PUT'
        atm = round(p / 50) * 50
        entry_prem = round(max(50.0, p * 0.0045), 2)
        sl_price = round(entry_prem * (1 - INITIAL_SL_PCT), 2)
        pos = {
            'opt_type': opt_type, 'strike': atm, 'entry_prem': entry_prem,
            'initial_sl': sl_price, 'sl': sl_price, 'high_p': entry_prem,
            'entry_spot': p, 'entry_time': ts, 'tsl_log': []
        }

tdf = pd.DataFrame(trades)
print("=" * 135)
print("  TODAY'S TRADE LOG (02-Sep-2026) | NIFTY 08 SEP EXPIRY | 1 Lot = 65 Qty")
print("=" * 135)

if tdf.empty and pos is None:
    print("No trades closed or open today.")
elif not tdf.empty:
    for i, t in tdf.iterrows():
        pnl_s = f"+INR {t['PnL']:,.2f}" if t['PnL'] >= 0 else f"-INR {abs(t['PnL']):,.2f}"
        print(f"  #{i+1:<2} {t['Contract']:<26} {t['Entry Time']:>8} -> {t['Exit Time']:>8} | "
              f"Spot: {t['Spot In']:>8,.1f} -> {t['Spot Out']:>8,.1f} | "
              f"Entry: INR{t['Entry Prem']:>6.2f} | SL: INR{t['SL Price']:>5.2f} | "
              f"Peak: INR{t['Peak Prem']:>6.2f} ({t['Peak %']:>6}) | Exit: INR{t['Exit Prem']:>6.2f} | "
              f"PnL: {pnl_s:>12} ({t['Return']:>6}) | {t['Trigger']}")
        if t['TSL Log']:
            print(f"       TSL Trail: {' -> '.join(t['TSL Log'])}")

if pos is not None:
    spot_chg = (today_bars.iloc[-1]['close'] - pos['entry_spot']) if pos['opt_type'] == 'CALL' else (pos['entry_spot'] - today_bars.iloc[-1]['close'])
    curr_prem = max(5.0, pos['entry_prem'] + spot_chg * 0.55)
    unrealized = (curr_prem - pos['entry_prem']) * QTY
    print(f"\n  ACTIVE OPEN POSITION: {pos['strike']} {pos['opt_type']} entered at {pos['entry_time'].strftime('%H:%M:%S')} @ INR {pos['entry_prem']:.2f}")
    print(f"  Current Option Premium: INR {curr_prem:.2f} | Unrealized P&L: INR {unrealized:+,.2f}")

tot_pnl = tdf['PnL'].sum() if not tdf.empty else 0.0
print("=" * 135)
print(f"  Total Closed Trades Today : {len(tdf)}")
print(f"  Total Realized P&L Today  : INR {tot_pnl:+,.2f}")
print("=" * 135)
