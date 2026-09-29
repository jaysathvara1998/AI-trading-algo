"""
Context-first price-action study on 3y of 1-min NIFTY/SENSEX: higher-timeframe (15-min) zones and structure,
lower-timeframe (5-min) confirmation. Measures forward spot moves after each event type; no costs, no strategy.

Events:
  A. Supply/demand zone (15-min): base of 1-3 small candles followed by an impulsive departure (>= 1.5 x ATR15,
     closing beyond the base). First retest (price enters the zone) -> forward move, and "reaction" = reaches
     +1 ATR15 in the zone's direction before a 15-min close through the distal line.
  B. Pullback in a 15-min trend: HH+HL structure, price returns to within 0.25 ATR15 of the last higher low
     (mirror for downtrend), and a 5-min candle closes back in the trend direction -> forward move.
  C. Make-or-break level (last higher low in an uptrend / last lower high in a downtrend): 5-min close through it
     -> continuation? vs. hold (touch, no close through) -> bounce?
  D. Trendline break + retest on 15-min swings (2 swing lows rising / 2 swing highs falling): close beyond by
     0.25 ATR15, retest within 6 x 15-min bars without closing back, 5-min confirmation -> forward move.
  E. Sweep of previous-day high/low AGAINST the 15-min trend, close back inside -> reversal move.
Run: PA_DATA_TAG=3y .venv/bin/python research/zone_study.py
"""
import sys, os
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("PA_DATA_TAG", "3y")
from pa_study import load, tstat

def agg(g, k):
    o = g.open.values[::k]; c = g.close.values[k-1::k]; n = min(len(o), len(c))
    h = np.array([g.high.values[i*k:(i+1)*k].max() for i in range(n)]); l = np.array([g.low.values[i*k:(i+1)*k].min() for i in range(n)])
    return o[:n], h, l, c[:n]

def report(name, recs):
    r = pd.DataFrame(recs)
    if r.empty or len(r) < 20:
        print(f"  {name:52s} n={len(r)} (too few)"); return r
    line = f"  {name:52s} n={len(r):5d} | +30m {r.f30.mean():+6.1f} (hit {100*(r.f30>0).mean():.0f}%, t={tstat(r.f30):+5.2f}) | +60m {r.f60.mean():+6.1f} (t={tstat(r.f60):+5.2f})"
    if "react" in r: line += f" | reaches +1 ATR before invalidation: {100*r.react.mean():.0f}%"
    print(line)
    if "year" in r:
        print("      by year (+60m mean):", {int(y): round(float(x.f60.mean()), 1) for y, x in r.groupby("year")})
    return r

def study(sym):
    df = load(sym); days = [g.reset_index(drop=True) for _, g in df.groupby("date", sort=True)]; dates = [g.date.iloc[0] for g in days]
    print(f"\n===== {sym.upper()} {len(days)} sessions =====")
    A, B, C_break, C_hold, D, E = [], [], [], [], [], []
    for j, g in enumerate(days):
        c1 = g.close.values; h1 = g.high.values; l1 = g.low.values
        o15, h15, l15, c15 = agg(g, 15); n15 = len(c15)
        o5, h5, l5, c5 = agg(g, 5)
        year = pd.Timestamp(g.timestamp.iloc[0]).year
        rng15 = h15 - l15
        def atr15(k): return max(1e-6, float(np.median(rng15[max(0, k-8):k+1])))
        def fwd(m, d, k): return (c1[min(m + k, 374)] - c1[m]) * d
        # ---- 15-min swings (1/1 fractal, confirmed 1 bar later)
        sh = [i for i in range(1, n15-1) if h15[i] > h15[i-1] and h15[i] >= h15[i+1]]
        sl = [i for i in range(1, n15-1) if l15[i] < l15[i-1] and l15[i] <= l15[i+1]]
        # ================= A. supply/demand zones =================
        zones = []   # (kind, proximal, distal, born_bar15, dir)
        for k in range(2, n15 - 1):
            a = atr15(k)
            # base: 1-3 candles with range <= 0.7 ATR ending at k-1; departure candle k impulsive
            for blen in (1, 2, 3):
                b0, b1 = k - blen, k - 1
                if b0 < 0: continue
                if any(rng15[x] > 0.7 * a for x in range(b0, b1 + 1)): continue
                bh, bl = h15[b0:b1+1].max(), l15[b0:b1+1].min()
                if rng15[k] >= 1.5 * a and c15[k] > bh + 0.5 * a:      # rally out -> demand zone
                    zones.append(("demand", bh, bl, k, +1)); break
                if rng15[k] >= 1.5 * a and c15[k] < bl - 0.5 * a:      # drop out -> supply zone
                    zones.append(("supply", bl, bh, k, -1)); break
        for kind, prox, dist, born, d in zones:
            a = atr15(born); height = abs(prox - dist)
            # first retest after the departure candle: 1-min price enters the zone
            start = (born + 1) * 15 + 15   # skip the departure bar and the next 15 min
            for m in range(start, 375 - 5):
                touched = (l1[m] <= prox) if d > 0 else (h1[m] >= prox)
                if not touched: continue
                # invalidation: a later 15-min close through the distal line
                k15 = m // 15; react = False; inval = None
                for kk in range(k15 + 1, n15):
                    if (c15[kk] < dist) if d > 0 else (c15[kk] > dist): inval = kk; break
                    if (h15[kk] >= prox + a) if d > 0 else (l15[kk] <= prox - a): react = True; break
                A.append({"kind": kind, "f30": fwd(m, d, 30), "f60": fwd(m, d, 60), "react": react, "year": year, "fresh": True, "height_atr": height / a, "minute": m})
                break
        # ================= B/C. pullback to last higher low / make-or-break =================
        for k in range(6, n15 - 4):
            hs = [x for x in sh if x <= k - 1][-2:]; ls = [x for x in sl if x <= k - 1][-2:]
            if len(hs) < 2 or len(ls) < 2: continue
            up = h15[hs[1]] > h15[hs[0]] and l15[ls[1]] > l15[ls[0]]; dn = h15[hs[1]] < h15[hs[0]] and l15[ls[1]] < l15[ls[0]]
            if not (up or dn): continue
            d = 1 if up else -1; a = atr15(k)
            mob = l15[ls[1]] if up else h15[hs[1]]          # make-or-break level = last higher low / lower high
            # scan the 5-min candles inside 15-min bar k for a pullback touch + confirmation, or a close through
            for q in range(k * 3, min(k * 3 + 3, len(c5))):
                m = q * 5 + 4
                near = (l5[q] <= mob + 0.25 * a and c5[q] > mob) if up else (h5[q] >= mob - 0.25 * a and c5[q] < mob)
                through = (c5[q] < mob - 0.1 * a) if up else (c5[q] > mob + 0.1 * a)
                if through:
                    C_break.append({"f30": fwd(m, -d, 30), "f60": fwd(m, -d, 60), "year": year}); break
                if near and q + 1 < len(c5):
                    # 5-min confirmation: next candle closes in trend direction beyond this candle's high/low
                    conf = (c5[q+1] > h5[q]) if up else (c5[q+1] < l5[q])
                    m2 = (q + 1) * 5 + 4
                    if conf:
                        B.append({"f30": fwd(m2, d, 30), "f60": fwd(m2, d, 60), "year": year, "minute": m2})
                    C_hold.append({"f30": fwd(m2, d, 30), "f60": fwd(m2, d, 60), "year": year, "confirmed": conf})
                    break
        # ================= D. 15-min trendline break + retest =================
        for k in range(6, n15 - 3):
            a = atr15(k)
            for pts, d in (([x for x in sl if x <= k - 1][-2:], -1), ([x for x in sh if x <= k - 1][-2:], +1)):
                if len(pts) < 2 or pts[1] - pts[0] < 2: continue
                x1, x2 = pts; y1, y2 = (l15[x1], l15[x2]) if d < 0 else (h15[x1], h15[x2]); slope = (y2 - y1) / (x2 - x1)
                if (d < 0 and slope <= 0) or (d > 0 and slope >= 0): continue
                line = lambda kk: y2 + slope * (kk - x2)
                broke = (c15[k] < line(k) - 0.25 * a and c15[k-1] >= line(k-1) - 0.25 * a) if d < 0 else (c15[k] > line(k) + 0.25 * a and c15[k-1] <= line(k-1) + 0.25 * a)
                if not broke: continue
                # retest within 6 bars, no close back through, then a 5-min confirmation close in break direction
                for kk in range(k + 1, min(k + 7, n15)):
                    if (c15[kk] > line(kk)) if d < 0 else (c15[kk] < line(kk)): break
                    near = (h15[kk] >= line(kk) - 0.25 * a) if d < 0 else (l15[kk] <= line(kk) + 0.25 * a)
                    if not near: continue
                    for q in range(kk * 3, min(kk * 3 + 3, len(c5) - 1)):
                        conf = (c5[q+1] < l5[q]) if d < 0 else (c5[q+1] > h5[q])
                        if conf:
                            m2 = (q + 1) * 5 + 4; D.append({"f30": fwd(m2, d, 30), "f60": fwd(m2, d, 60), "year": year}); break
                    break
        # ================= E. PDH/PDL sweep against the 15-min trend =================
        if j > 0 and (dates[j] - dates[j-1]).days <= 5:
            p = days[j-1]; pdh, pdl = p.high.max(), p.low.min()
            for q in range(6, len(c5) - 13):
                k = q // 3
                hs = [x for x in sh if x <= k - 1][-2:]; ls = [x for x in sl if x <= k - 1][-2:]
                if len(hs) < 2 or len(ls) < 2: continue
                up = h15[hs[1]] > h15[hs[0]] and l15[ls[1]] > l15[ls[0]]; dn = h15[hs[1]] < h15[hs[0]] and l15[ls[1]] < l15[ls[0]]
                if up and l5[q] < pdl and c5[q] > pdl:        # swept PDL in an uptrend, closed back above -> buy
                    m = q * 5 + 4; E.append({"f30": fwd(m, 1, 30), "f60": fwd(m, 1, 60), "year": year}); break
                if dn and h5[q] > pdh and c5[q] < pdh:        # swept PDH in a downtrend, closed back below -> sell
                    m = q * 5 + 4; E.append({"f30": fwd(m, -1, 30), "f60": fwd(m, -1, 60), "year": year}); break
    print("A. FRESH 15-min supply/demand zone, first retest, move in the zone's direction:")
    r = report("all fresh zones", A)
    if len(r):
        for kind, x in r.groupby("kind"): print(f"      {kind:8s} n={len(x):4d} +30m {x.f30.mean():+6.1f} (t={tstat(x.f30):+5.2f}) +60m {x.f60.mean():+6.1f} (t={tstat(x.f60):+5.2f}) reaction {100*x.react.mean():.0f}%")
        for lab, m in (("retest before 11:00", r.minute < 105), ("retest 11:00-14:30", (r.minute >= 105) & (r.minute < 315))):
            x = r[m]; print(f"      {lab:20s} n={len(x):4d} +60m {x.f60.mean():+6.1f} (t={tstat(x.f60):+5.2f}) reaction {100*x.react.mean():.0f}%")
    print("B. PULLBACK to last higher low / lower high in a 15-min trend + 5-min confirmation close, move with trend:")
    report("pullback + confirmation", B)
    print("C. MAKE-OR-BREAK level (last HL / LH):")
    report("5-min close THROUGH it -> move in break direction", C_break)
    r = report("touch and HOLD -> move with trend", C_hold)
    print("D. 15-min TRENDLINE break + retest + 5-min confirmation, move in break direction:")
    report("break-retest-confirm", D)
    print("E. PDH/PDL SWEEP against the 15-min trend, 5-min close back inside, move with trend:")
    report("counter-trend sweep reclaimed", E)
    med = float(np.median([np.median(np.abs(g.close.values[60:] - g.close.values[:-60])) for g in days]))
    print(f"  (reference: median |60-min move| = {med:.1f} pts; a round trip costs ~1.5 pts NIFTY / ~5 pts SENSEX)")

for s in ("nifty", "sensex"):
    study(s)
