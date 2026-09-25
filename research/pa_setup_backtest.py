"""
Implementation and backtest of research/price_action_setup.md (setups A, B, C) on the 3-year 1-minute data.

Rules are implemented as written in the specification; operationalisations that the text left open are
marked OPERATIONALISATION in comments. Nothing here is tuned.

Instruments (spec section 3): OPT72 = 0.72-delta weekly option (setup B); OPT85 = deep ITM option, delta 0.85
(setups A, C); FUT = index futures (setups A, C alternative). Costs: Dhan brokerage Rs 20/order, options STT
0.15% of sell premium, futures STT 0.02% of sell notional, exchange charges, stamp, GST, plus a spread allowance.

    .venv/bin/python research/pa_setup_backtest.py --data 3y
"""
import sys, argparse, json
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pa_study import load, tstat
from trend_range_strategy import expiry_weekday

LOT = {"nifty": 65, "sensex": 20}
ATR_FLOOR = {"nifty": 8.0, "sensex": 25.0}
MIN_RANGE = {"nifty": 60.0, "sensex": 210.0}
MIN_OR = {"nifty": 40.0, "sensex": 140.0}
ROUND = {"nifty": 100.0, "sensex": 500.0}
EXIT_MIN = 355          # flat by 15:10
WINDOWS = [(34, 104), (259, 314)]   # trigger-bar close minute: 09:49-11:00 and 13:34-14:30 (5-min closes)

INSTR = {  # premium as fraction of spot, delta, theta per minute (fraction of premium), spread per side (fraction), kind
    "OPT72": dict(prem=0.007, delta=0.72, theta=0.00025, spread=0.002, kind="opt"),
    "OPT85": dict(prem=0.025, delta=0.85, theta=0.00006, spread=0.003, kind="opt"),
    "FUT":   dict(prem=1.0,   delta=1.0,  theta=0.0,     spread=None,  kind="fut"),
}
FUT_SLIP = {"nifty": 0.5, "sensex": 2.0}

def costs(kind, buy_v, sell_v):
    if kind == "opt":
        txn = 0.00035 * (buy_v + sell_v); stt = 0.0015 * sell_v
    else:
        txn = 0.0000173 * (buy_v + sell_v); stt = 0.0002 * sell_v
    return 40 + stt + txn + 0.000001 * (buy_v + sell_v) + 0.00003 * buy_v + 0.18 * (40 + txn)

# ---------------------------------------------------------------- bars
def bars5(g):
    o = g.open.values[::5]; c = g.close.values[4::5]; n = min(len(o), len(c))
    h = np.array([g.high.values[i*5:(i+1)*5].max() for i in range(n)]); l = np.array([g.low.values[i*5:(i+1)*5].min() for i in range(n)])
    return o[:n], h, l, c[:n]

def bars15(g):
    o = g.open.values[::15]; c = g.close.values[14::15]; n = min(len(o), len(c))
    h = np.array([g.high.values[i*15:(i+1)*15].max() for i in range(n)]); l = np.array([g.low.values[i*15:(i+1)*15].min() for i in range(n)])
    return o[:n], h, l, c[:n]

def swings(h, l, upto):
    """2/2 fractal swings confirmed by bar `upto` (swing at i needs bars i+1, i+2 -> confirmed when upto >= i+2)."""
    sh = [i for i in range(2, upto - 1) if h[i] == h[i-2:i+3].max()]
    sl = [i for i in range(2, upto - 1) if l[i] == l[i-2:i+3].min()]
    return sh, sl

def atr_at(h, l, i, sym):
    return max(ATR_FLOOR[sym], float(np.median(h[:i + 1] - l[:i + 1])))

# ---------------------------------------------------------------- day context
class Day:
    def __init__(self, sym, g, prev):
        self.sym, self.g, self.prev = sym, g, prev
        self.o5, self.h5, self.l5, self.c5 = bars5(g); self.n5 = len(self.c5)
        self.o15, self.h15, self.l15, self.c15 = bars15(g)
        self.orh, self.orl = self.h5[:3].max(), self.l5[:3].min(); self.orw = self.orh - self.orl
        self.pdh = self.pdl = self.pdr = None
        if prev is not None:
            self.pdh, self.pdl = float(prev.high.max()), float(prev.low.min()); self.pdr = self.pdh - self.pdl
        self.skip = None
        d0 = pd.Timestamp(g.timestamp.iloc[0])
        if d0.weekday() == expiry_weekday(sym, d0): self.skip = "expiry"
        elif prev is not None and abs(g.open.values[0] - prev.close.iloc[-1]) / prev.close.iloc[-1] > 0.01: self.skip = "gap"
        elif self.orw < MIN_OR[sym]: self.skip = "narrow_or"
        self.kind, self.dir = self.classify()

    def classify(self):
        """Bar 18 (index 17, close 10:44). OPERATIONALISATION: 'two higher highs and higher lows' = the last three
        completed 15-min bars have strictly rising highs and lows (mirror for down)."""
        i = 17; pc = self.c5[i]
        k15 = 5  # 15-min bars 3,4,5 completed by 10:44
        hh = self.h15[k15] > self.h15[k15-1] > self.h15[k15-2] and self.l15[k15] > self.l15[k15-1] > self.l15[k15-2]
        ll = self.h15[k15] < self.h15[k15-1] < self.h15[k15-2] and self.l15[k15] < self.l15[k15-1] < self.l15[k15-2]
        if pc >= self.orh + 0.5 * self.orw and hh: return "TREND", +1
        if pc <= self.orl - 0.5 * self.orw and ll: return "TREND", -1
        if self.orl <= pc <= self.orh and self.orw >= MIN_RANGE[self.sym]: return "RANGE", 0
        return "UNDEF", 0

    def levels(self, i):
        """Levels available at 5-min bar i: PDH/PDL, OR edges, confirmed 15-min swings, round numbers near price."""
        lv = []
        if self.pdh: lv += [("PDH", self.pdh), ("PDL", self.pdl)]
        lv += [("ORH", self.orh), ("ORL", self.orl)]
        k = (i * 5 + 4) // 15   # completed 15-min bars: 0..k-1
        sh, sl = swings(self.h15, self.l15, k - 1)
        lv += [("S15H", float(self.h15[x])) for x in sh] + [("S15L", float(self.l15[x])) for x in sl]
        r = ROUND[self.sym]; p = self.c5[i]
        lv += [("RN", float(np.floor(p / r) * r)), ("RN", float(np.ceil(p / r) * r))]
        return lv

    def in_window(self, i):
        m = i * 5 + 4
        return any(a <= m <= b for a, b in WINDOWS)

# ---------------------------------------------------------------- setups
def setup_A(day):
    """Trendline break and retest, with trend. Trend days only."""
    if day.kind != "TREND": return []
    d = day.dir; h, l, c = day.h5, day.l5, day.c5; out = []
    for i in range(6, day.n5 - 14):
        atr = atr_at(h, l, i, day.sym)
        sh, sl = swings(h, l, i)
        pts = ([x for x in sh][-2:] if d > 0 else [x for x in sl][-2:])   # falling resistance in up-day / rising support in down-day
        if len(pts) < 2 or pts[1] - pts[0] < 3: continue
        x1, x2 = pts; y1, y2 = (h[x1], h[x2]) if d > 0 else (l[x1], l[x2])
        slope = (y2 - y1) / (x2 - x1)
        if (d > 0 and slope >= 0) or (d < 0 and slope <= 0): continue
        line = lambda k: y2 + slope * (k - x2)
        # validity: a third touch within 0.25 ATR after x2 that does not close through; not stale (last touch <= 20 bars ago)
        touches = [k for k in range(x2 + 1, i) if abs((h[k] if d > 0 else l[k]) - line(k)) <= 0.25 * atr and ((c[k] <= line(k)) if d > 0 else (c[k] >= line(k)))]
        if not touches or i - max(touches) > 20: continue
        # 1. break: close >= 0.25 ATR beyond
        broke = (c[i] >= line(i) + 0.25 * atr and c[i-1] < line(i-1) + 0.25 * atr) if d > 0 else (c[i] <= line(i) - 0.25 * atr and c[i-1] > line(i-1) - 0.25 * atr)
        if not broke: continue
        # 2. retest within 6 bars without closing back through; 3. trigger bar
        ret_ext = None
        for k in range(i + 1, min(i + 7, day.n5)):
            if (c[k] < line(k)) if d > 0 else (c[k] > line(k)): break   # closed back through -> setup void
            near = (l[k] <= line(k) + 0.25 * atr) if d > 0 else (h[k] >= line(k) - 0.25 * atr)
            if near: ret_ext = (l[k] if d > 0 else h[k]) if ret_ext is None else (min(ret_ext, l[k]) if d > 0 else max(ret_ext, h[k]))
            if ret_ext is None: continue
            rng = h[k] - l[k]
            strong = (c[k] >= h[k] - rng / 3 and c[k] > h[k-1]) if d > 0 else (c[k] <= l[k] + rng / 3 and c[k] < l[k-1])
            if strong and day.in_window(k):
                e = c[k]; stop = ret_ext - d * 0.1 * atr; R = abs(e - stop)
                if R <= 0: break
                # target: nearer of prior swing extreme and entry + last completed swing height; must be >= 2R
                prior = (max([h[x] for x in sh] or [e]) if d > 0 else min([l[x] for x in sl] or [e]))
                legs = [abs(h[a] - l[b]) for a, b in zip(sh, sl)] if sh and sl else []
                mm = e + d * (legs[-1] if legs else 0)
                cands = [t for t in (prior, mm) if (t - e) * d > 0]
                if not cands: break
                tgt = min(cands, key=lambda t: abs(t - e))
                if abs(tgt - e) < 2 * R: break
                out.append(dict(setup="A", bar=k, dir=d, stop=stop, target=tgt, time_bars=12, be_r=1.0, trail_r=2.0, min_r_at=None)); break
        if out: break   # one per day
    return out

def setup_B(day):
    """Deep trap reversal. Trend or range days only."""
    if day.kind == "UNDEF": return []
    h, l, c = day.h5, day.l5, day.c5; out = []; used = set()
    for i in range(6, day.n5 - 8):
        atr = atr_at(h, l, i, day.sym)
        for name, lvl in day.levels(i):
            if name == "RN": continue   # OPERATIONALISATION: round numbers are context, not trap levels (spec lists PD, OR, 15-min swings)
            for pen in (+1, -1):        # penetration direction
                deep = (c[i] >= lvl + 0.5 * atr) if pen > 0 else (c[i] <= lvl - 0.5 * atr)
                prev_inside = (c[i-1] < lvl) if pen > 0 else (c[i-1] > lvl)
                if not (deep and prev_inside): continue
                d = -pen   # trade direction = back toward the original side
                if day.kind == "TREND" and d != day.dir: continue
                if day.kind == "RANGE" and name not in ("ORH", "ORL"): continue
                key = (name, round(lvl, 1))
                if key in used: continue
                ext = c[i]
                for k in range(i + 1, min(i + 4, day.n5)):
                    ext = max(ext, h[k]) if pen > 0 else min(ext, l[k])
                    back = (c[k] < lvl) if pen > 0 else (c[k] > lvl)
                    if not back: continue
                    if not day.in_window(k): break
                    e = c[k]; stop = ext + pen * 0.1 * atr; R = abs(e - stop)
                    # target: opposite edge of the structure the level belongs to
                    if name in ("ORH", "ORL"): tgt = day.orl if d < 0 else day.orh
                    elif name in ("PDH", "PDL"): tgt = day.pdl if d < 0 else day.pdh
                    else:
                        sh, sl = swings(h, l, k); tgt = (min([l[x] for x in sl]) if d < 0 and sl else (max([h[x] for x in sh]) if d > 0 and sh else None))
                    if tgt is None or (tgt - e) * d <= 0 or abs(tgt - e) < 1.5 * R: break
                    out.append(dict(setup="B", bar=k, dir=d, stop=stop, target=tgt, time_bars=6, be_r=1.0, trail_r=2.0, min_r_at=(6, 0.5))); used.add(key); break
    return out[:2]

def setup_C(day):
    """Mid-session previous-day level break that holds three closes."""
    if day.pdh is None or day.kind == "RANGE": return []
    h, l, c = day.h5, day.l5, day.c5
    for i in range(21, 63):           # closes 11:04 .. 14:29
        atr = atr_at(h, l, i, day.sym)
        for lvl, d in ((day.pdh, +1), (day.pdl, -1)):
            if day.kind == "TREND" and d != day.dir: continue
            beyond = (c[i] >= lvl + 0.25 * atr) if d > 0 else (c[i] <= lvl - 0.25 * atr)
            before = (c[i-1] < lvl + 0.25 * atr) if d > 0 else (c[i-1] > lvl - 0.25 * atr)
            if not (beyond and before): continue
            k = i + 3
            if k >= day.n5 or k * 5 + 4 > 259: continue          # entry no later than 13:34 (>= 60 min of window left)
            held = all(((c[j] > lvl) if d > 0 else (c[j] < lvl)) for j in range(i + 1, k + 1))
            if not held: continue
            e = c[k]; stop = lvl - d * 0.1 * atr; R = abs(e - stop)
            tgt = e + d * 0.5 * day.pdr
            if R <= 0 or abs(tgt - e) < 2 * R: continue
            return [dict(setup="C", bar=k, dir=d, stop=stop, target=tgt, time_bars=None, be_r=1.0, trail_r=2.0, min_r_at=None)]
    return []

# ---------------------------------------------------------------- execution
def simulate(day, sig, instr, sym):
    """1-min execution: entry at trigger close (+spread), stop/target on 1-min extremes (adverse first),
    breakeven at +1R, trail 0.5R behind best 5-min close after +2R, time stops on 5-min closes, flat at 15:10."""
    g = day.g; h1, l1, c1 = g.high.values, g.low.values, g.close.values
    P = INSTR[instr]; lot = LOT[sym]; d = sig["dir"]; i0 = sig["bar"] * 5 + 4; e = c1[i0]
    stop, tgt = sig["stop"], sig["target"]; R = abs(e - stop)
    if P["kind"] == "opt":
        p0 = e * P["prem"]; fill_in = p0 * (1 + P["spread"])
        prem = lambda spot, mins: max(0.5, p0 + P["delta"] * d * (spot - e) - p0 * P["theta"] * mins)
    else:
        fill_in = e + d * FUT_SLIP[sym]
        prem = lambda spot, mins: spot
    be_spot = e + d * (1.5 / P["delta"] if P["kind"] == "opt" else 1.5)   # +1.5 premium points locked
    best_close = e; exit_i, reason = None, None
    for k in range(i0 + 1, EXIT_MIN + 1):
        adverse = l1[k] if d > 0 else h1[k]; fav = h1[k] if d > 0 else l1[k]
        if (adverse <= stop) if d > 0 else (adverse >= stop):
            exit_i, exit_spot, reason = k, stop, "stop"; break
        if (fav >= tgt) if d > 0 else (fav <= tgt):
            exit_i, exit_spot, reason = k, tgt, "target"; break
        if k % 5 == 4:   # 5-min close: management
            bars = (k - i0) // 5; gain_r = (c1[k] - e) * d / R
            best_close = max(best_close, c1[k]) if d > 0 else min(best_close, c1[k])
            if gain_r >= sig["be_r"]: stop = max(stop, be_spot) if d > 0 else min(stop, be_spot)
            if gain_r >= sig["trail_r"]:
                tr = best_close - d * 0.5 * R; stop = max(stop, tr) if d > 0 else min(stop, tr)
            if sig["min_r_at"] and bars >= sig["min_r_at"][0] and gain_r < sig["min_r_at"][1]:
                exit_i, exit_spot, reason = k, c1[k], "time_minr"; break
            if sig["time_bars"] and bars >= sig["time_bars"]:
                exit_i, exit_spot, reason = k, c1[k], "time"; break
    if exit_i is None: exit_i, exit_spot, reason = EXIT_MIN, c1[EXIT_MIN], "eod"
    mins = exit_i - i0
    if P["kind"] == "opt":
        p_out = prem(exit_spot, mins) * (1 - P["spread"]); gross = (p_out - fill_in) * lot; cst = costs("opt", fill_in * lot, p_out * lot)
    else:
        p_out = exit_spot - d * FUT_SLIP[sym]; gross = (p_out - fill_in) * d * lot; cst = costs("fut", e * lot, e * lot)
    return dict(setup=sig["setup"], instr=instr, dir=d, entry_min=i0, exit_min=exit_i, spot_pts=(exit_spot - e) * d, R=R, r_mult=(exit_spot - e) * d / R,
                gross=gross, costs=cst, net=gross - cst, reason=reason, hold=mins)

def run(sym, days, dates, instr_for):
    trades, kinds, skips = [], [], []
    for j, g in enumerate(days):
        prev = days[j-1] if j > 0 and (dates[j] - dates[j-1]).days <= 5 else None
        day = Day(sym, g, prev)
        if day.skip: skips.append(day.skip); continue
        kinds.append(day.kind)
        sigs = sorted(setup_A(day) + setup_B(day) + setup_C(day), key=lambda s: s["bar"])
        busy, day_net, consec = -1, 0.0, 0
        for s in sigs:
            if s["bar"] * 5 + 4 <= busy: continue
            if consec >= 2 or day_net <= -2500: break
            t = simulate(day, s, instr_for[s["setup"]], sym); t["date"] = dates[j]; t["kind"] = day.kind
            trades.append(t); busy = t["exit_min"]; day_net += t["net"]; consec = consec + 1 if t["net"] <= 0 else 0
    return pd.DataFrame(trades), pd.Series(kinds).value_counts().to_dict(), pd.Series(skips).value_counts().to_dict()

def summ(t):
    if t.empty: return "no trades"
    w = t[t.net > 0]; L = t[t.net <= 0]; cum = t.net.cumsum(); dd = (cum - cum.cummax()).min()
    pf = w.net.sum() / -L.net.sum() if len(L) and L.net.sum() < 0 else float("inf")
    yrs = t.groupby(pd.to_datetime(t.date.astype(str)).dt.year).net.sum().round(0).to_dict()
    return (f"n={len(t):4d} net {t.net.sum():+9.0f} (gross {t.gross.sum():+8.0f} costs {t.costs.sum():6.0f}) win {100*len(w)/len(t):4.1f}% PF {pf:4.2f} "
            f"net/trade {t.net.mean():+6.0f} t={tstat(t.net):+5.2f} maxDD {dd:8.0f} avgR {t.r_mult.mean():+5.2f} hold {t.hold.mean():4.0f}m | years {yrs}")

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default="3y"); ap.add_argument("--out", default=None); a = ap.parse_args()
    SPEC = {"A": "OPT85", "B": "OPT72", "C": "OPT85"}
    res = {}
    for sym in ("nifty", "sensex"):
        df = load(sym, a.data); days = [g.reset_index(drop=True) for _, g in df.groupby("date", sort=True)]; dates = [g.date.iloc[0] for g in days]
        print(f"\n===== {sym.upper()} ({len(days)} sessions) =====")
        t, kinds, skips = run(sym, days, dates, SPEC)
        print("day types:", kinds, "| skipped:", skips)
        print(f"AS SPECIFIED (A,C on OPT85; B on OPT72): {summ(t)}")
        for s in "ABC":
            x = t[t.setup == s]; print(f"   setup {s}: {summ(x)}")
            if not x.empty: print(f"            exits: {x.reason.value_counts().to_dict()} | by day type: {x.groupby('kind').net.agg(['size','sum']).round(0).to_dict('index')}")
        for instr in ("OPT72", "OPT85", "FUT"):
            tt, _, _ = run(sym, days, dates, {"A": instr, "B": instr, "C": instr})
            print(f"ALL SETUPS ON {instr:5s}: {summ(tt)}")
            for s in "ABC":
                x = tt[tt.setup == s]; print(f"   setup {s}: {summ(x)}")
        res[sym] = {"kinds": kinds, "skips": skips, "n": int(len(t)), "net": float(t.net.sum()) if len(t) else 0.0}
        if a.out: t.to_csv(f"{a.out}_{sym}_trades.csv", index=False)
    if a.out: json.dump(res, open(f"{a.out}_summary.json", "w"), indent=1, default=str)

if __name__ == "__main__":
    main()
