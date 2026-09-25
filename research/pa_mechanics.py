"""How trendline breaks, failed breaks (trapped 'weak hands') and liquidity sweeps actually behave on 3y of 1-min NIFTY/SENSEX.
Measures forward moves (spot pts) after each event type; no strategy, no costs. Run: PA_DATA_TAG=3y .venv/bin/python research/pa_mechanics.py"""
import sys, os
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pa_study import load, tstat
os.environ.setdefault("PA_DATA_TAG", "3y")

def c5(g):
    o = g.open.values[::5]; c = g.close.values[4::5]; n = min(len(o), len(c))
    h = np.array([g.high.values[i*5:(i+1)*5].max() for i in range(n)]); l = np.array([g.low.values[i*5:(i+1)*5].min() for i in range(n)])
    return o[:n], h, l, c[:n]

def window(i5):  # 5-min bar index -> session window
    m = i5 * 5
    return "09:15-09:45" if m < 30 else "09:45-11:00" if m < 105 else "11:00-14:30" if m < 315 else "14:30-15:15"

def report(name, recs, key="f6"):
    r = pd.DataFrame(recs)
    if r.empty: print(f"  {name}: none"); return
    line = f"  {name:44s} n={len(r):5d} | +30m {r.f6.mean():+6.1f} pts (hit {100*(r.f6>0).mean():.0f}%, t={tstat(r.f6):+5.2f}) | +60m {r.f12.mean():+6.1f} (t={tstat(r.f12):+5.2f})"
    if "fail" in r: line += f" | fails(3 bars) {100*r.fail.mean():.0f}%"
    print(line)
    if "win" in r:
        for w, x in r.groupby("win"):
            print(f"      {w:12s} n={len(x):5d} +30m {x.f6.mean():+6.1f} (t={tstat(x.f6):+5.2f})  +60m {x.f12.mean():+6.1f} (t={tstat(x.f12):+5.2f})" + (f"  fails {100*x.fail.mean():.0f}%" if "fail" in x else ""))

for sym in ("nifty", "sensex"):
    df = load(sym); days = [g.reset_index(drop=True) for _, g in df.groupby("date", sort=True)]; dates = [g.date.iloc[0] for g in days]
    noise30 = float(np.median([np.median(np.abs(g.close.values[30:] - g.close.values[:-30])) for g in days]))
    print(f"\n===== {sym.upper()} {len(days)} sessions | median |30-min move| = {noise30:.1f} pts =====")
    TL, TLF, BRK, TRAP, SWP, HOLD, ORF, ORH_ = [], [], [], [], [], [], [], []
    for j, g in enumerate(days):
        o, h, l, c = c5(g); n = len(c)
        if n < 70: continue
        atr = np.median(h - l)
        # fractal swings (2 left / 2 right) on 5-min
        sh = [i for i in range(2, n - 2) if h[i] == h[i-2:i+3].max()]; sl = [i for i in range(2, n - 2) if l[i] == l[i-2:i+3].min()]
        def fwd(i, d, k): return (c[min(i + k, n - 1)] - c[i]) * d
        # ---- A. trendline breaks: line through the two most recent confirmed swing lows (rising) / highs (falling)
        last_tl_bar = -10
        for i in range(10, n - 13):
            lows = [x for x in sl if x <= i - 2][-2:]; highs = [x for x in sh if x <= i - 2][-2:]
            for pts, d in ((lows, -1), (highs, +1)):   # break of rising support -> d=-1 ; break of falling resistance -> d=+1
                if len(pts) < 2: continue
                (x1, x2) = pts; y1, y2 = (l[x1], l[x2]) if d < 0 else (h[x1], h[x2])
                if x2 - x1 < 3: continue
                slope = (y2 - y1) / (x2 - x1)
                if d < 0 and slope <= 0: continue      # need rising support
                if d > 0 and slope >= 0: continue      # need falling resistance
                line_i = y2 + slope * (i - x2); line_prev = y2 + slope * (i - 1 - x2)
                # touch count validation: a 3rd touch within 0.15 atr before i?
                touches = sum(1 for k in range(x2 + 1, i) if abs((l[k] if d < 0 else h[k]) - (y2 + slope * (k - x2))) <= 0.15 * atr)
                broke = (c[i] < line_i - 0.1 * atr and c[i - 1] >= line_prev) if d < 0 else (c[i] > line_i + 0.1 * atr and c[i - 1] <= line_prev)
                if not broke or i - last_tl_bar < 3: continue
                last_tl_bar = i
                fail = any(((c[k] > (y2 + slope * (k - x2))) if d < 0 else (c[k] < (y2 + slope * (k - x2)))) for k in range(i + 1, min(i + 4, n)))
                rec = {"f6": fwd(i, d, 6), "f12": fwd(i, d, 12), "fail": fail, "win": window(i), "touch3": touches >= 1, "day_trend_aligned": (c[i] - c[0]) * d > 0}
                TL.append(rec)
                if fail:  # failed trendline break -> trade the reversal from the failure bar
                    k = next(k for k in range(i + 1, min(i + 4, n)) if ((c[k] > (y2 + slope * (k - x2))) if d < 0 else (c[k] < (y2 + slope * (k - x2)))))
                    TLF.append({"f6": fwd(k, -d, 6), "f12": fwd(k, -d, 12), "win": window(k)})
        # ---- B. swing-level breakouts and traps (prior 12-bar high/low)
        last = -10
        for i in range(12, n - 13):
            hi, lo = h[i-12:i].max(), l[i-12:i].min()
            for d, brk in ((1, c[i] > hi), (-1, c[i] < lo)):
                if not brk or i - last < 2: continue
                last = i
                lvl = hi if d > 0 else lo
                BRK.append({"f6": fwd(i, d, 6), "f12": fwd(i, d, 12), "fail": any(((c[k] < lvl) if d > 0 else (c[k] > lvl)) for k in range(i + 1, min(i + 4, n))), "win": window(i)})
                # trap: breakout bar then a close back inside within 2 bars -> fade from that bar
                for k in range(i + 1, min(i + 3, n)):
                    if (c[k] < lvl) if d > 0 else (c[k] > lvl):
                        TRAP.append({"f6": fwd(k, -d, 6), "f12": fwd(k, -d, 12), "win": window(k), "depth_atr": (abs(c[i] - lvl)) / atr}); break
        # ---- C. liquidity sweep of previous-day high/low vs hold
        if j > 0 and (dates[j] - dates[j-1]).days <= 5:
            p = days[j-1]; pdh, pdl = p.high.max(), p.low.min()
            for lvl, d in ((pdh, +1), (pdl, -1)):
                if (c[0] > lvl) if d > 0 else (c[0] < lvl): continue
                hit = [i for i in range(1, n - 13) if ((h[i] >= lvl) if d > 0 else (l[i] <= lvl))]
                if not hit: continue
                i = hit[0]
                swept = (c[i] < lvl) if d > 0 else (c[i] > lvl)   # wick through, close back inside = sweep
                if swept: SWP.append({"f6": fwd(i, -d, 6), "f12": fwd(i, -d, 12), "win": window(i)})
                else: HOLD.append({"f6": fwd(i, d, 6), "f12": fwd(i, d, 12), "win": window(i)})
        # ---- D. opening-range (15m) break that fails within 3 bars: fade toward the other side
        orh, orl = h[:3].max(), l[:3].min()
        for i in range(3, 21):
            d = 1 if c[i] > orh else (-1 if c[i] < orl else 0)
            if not d: continue
            lvl = orh if d > 0 else orl
            for k in range(i + 1, min(i + 4, n)):
                if (c[k] < lvl) if d > 0 else (c[k] > lvl):
                    ORF.append({"f6": fwd(k, -d, 6), "f12": fwd(k, -d, 12), "win": window(k), "reach_other": any(((l[m] <= orl) if d > 0 else (h[m] >= orh)) for m in range(k + 1, min(k + 25, n)))}); break
            break
    print("A. TRENDLINE BREAK (5-min close beyond a 2-swing line by 0.1 ATR), forward move in BREAK direction:")
    report("all breaks", TL)
    r = pd.DataFrame(TL)
    for lab, m in (("with a 3rd touch (validated line)", r.touch3), ("aligned with day direction", r.day_trend_aligned), ("against day direction", ~r.day_trend_aligned)):
        x = r[m]; print(f"      {lab:34s} n={len(x):5d} +30m {x.f6.mean():+6.1f} (t={tstat(x.f6):+5.2f}) fails {100*x.fail.mean():.0f}%")
    print("   FAILED trendline break, move in REVERSAL direction from the failure bar:")
    report("failed-break reversal", TLF)
    print("B. SWING BREAKOUT (close beyond prior 12x5-min high/low), move in break direction:")
    report("all swing breakouts", BRK)
    print("   TRAP (close back inside within 2 bars), move AGAINST the breakout from the trap bar:")
    report("trap fade", TRAP)
    x = pd.DataFrame(TRAP); print(f"      shallow trap (break < 0.5 ATR) n={int((x.depth_atr<0.5).sum())} +30m {x[x.depth_atr<0.5].f6.mean():+.1f} (t={tstat(x[x.depth_atr<0.5].f6):+.2f}) | deep trap (>= 0.5 ATR) n={int((x.depth_atr>=0.5).sum())} +30m {x[x.depth_atr>=0.5].f6.mean():+.1f} (t={tstat(x[x.depth_atr>=0.5].f6):+.2f})")
    print("C. PREVIOUS-DAY HIGH/LOW: sweep (wick through, close back) -> move AGAINST; hold (close beyond) -> move WITH:")
    report("sweep -> reversal", SWP); report("hold -> continuation", HOLD)
    print("D. OPENING-RANGE break that fails within 3 bars, fade toward the other side:")
    report("OR failed-break fade", ORF)
    x = pd.DataFrame(ORF); print(f"      reaches the opposite OR edge within 2 hours: {100*x.reach_other.mean():.0f}%")
