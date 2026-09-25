"""
Debug: What signals is the VPIN stack generating day by day?
Shows every signal that passed GARCH stage but was vetoed, and what passed everything.
"""
import sys, io
import pandas as pd
import numpy as np
from algo_vpin_v2.data_feed import DhanDataFeed
from algo_vpin_v2.macro_features import MacroFeatureEngine
from algo_vpin_v2.vpin import VPINCalculator
from algo_vpin_v2.garch_engine import GARCHEngine, DirectionalSignal
from algo_vpin_v2.ensemble_brain import EnsembleBrain

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

def add_indicators(df):
    df = df.copy()
    df['ema9']  = df['close'].ewm(span=9,  adjust=False).mean()
    df['ema21'] = df['close'].ewm(span=21, adjust=False).mean()
    df['date_key'] = df['timestamp'].dt.date
    df['vwap'] = df.groupby('date_key', group_keys=False).apply(
        lambda g: (g['close'] * g['volume']).cumsum() / g['volume'].cumsum()
    )
    delta = df['close'].diff()
    gain  = delta.where(delta > 0, 0).rolling(14).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(14).mean().replace(0, 1e-9)
    df['rsi'] = 100 - (100 / (1 + gain / loss))
    return df

feed  = DhanDataFeed()
hist  = feed.fetch_historical_bars(days=9)
hist  = add_indicators(hist)

macro     = MacroFeatureEngine()
vpin_calc = VPINCalculator()
garch     = GARCHEngine()
brain     = EnsembleBrain()

print(f"Loaded {len(hist)} bars\n")
print(f"{'Date':<12} {'Time':>8} {'Price':>8} {'EMA9>21':>8} {'P>VWAP':>7} {'RSI':>6} {'GARCH':>6} {'VPIN':>6} {'Vetoed':>7} {'Action':>8} {'Reason'}")
print("-" * 115)

import datetime as dt
start_date = dt.date(2026, 8, 26)
prev_date = None

for idx, row in hist.iterrows():
    p    = float(row['close'])
    h    = float(row.get('high', p))
    l    = float(row.get('low',  p))
    v    = float(row['volume'])
    ts   = row['timestamp']
    ema9  = float(row.get('ema9', p))
    ema21 = float(row.get('ema21', p))
    vwap  = float(row.get('vwap', p))
    rsi   = float(row.get('rsi', 50.0))

    mst  = macro.update_1min_bar(p, h, l)
    vres = vpin_calc.process_bar(p, v, ts)
    gres = garch.add_bar(p, ts)

    if idx < 50 or ts.date() < start_date:
        continue

    # Only show market hours
    if not ((ts.hour == 9 and ts.minute >= 30) or (10 <= ts.hour < 15)):
        continue

    p_delta = p - float(hist.iloc[idx-1]['close'])
    sigma   = gres.sigma_next if gres else 0.001
    mu      = gres.mu_next    if gres else 0.0
    gsig    = gres.signal if gres else DirectionalSignal.HOLD
    dec     = brain.evaluate_signal(gsig, p_delta, sigma, mu, vres.vpin, mst)

    # Only print rows where GARCH fired a non-HOLD or momentum aligned
    ema_align  = ema9 > ema21
    vwap_align = p > vwap
    bullish    = ema_align and vwap_align and rsi > 52
    bearish    = not ema_align and not vwap_align and rsi < 48

    if gsig != DirectionalSignal.HOLD or bullish or bearish:
        trade_date = ts.date()
        if trade_date != prev_date:
            prev_date = trade_date
            print(f"\n--- {ts.strftime('%d-%b-%Y (%A)')} ---")

        row_str = (f"{ts.strftime('%d-%b'):>12} {ts.strftime('%H:%M'):>8} "
                   f"{p:>8.1f} {str(ema9>ema21):>8} {str(p>vwap):>7} {rsi:>6.1f} "
                   f"{gsig.name[:4]:>6} {vres.vpin:>6.3f} {str(dec.is_vetoed):>7} "
                   f"{dec.final_action.name[:4]:>8}  {dec.reason[:55]}")
        print(row_str)
