"""
Trend-day / range-day price-action strategy (from the September 2026 research note, section 06) and its backtest.

Rules (per session, 1-minute bars, all decisions on 15-minute closes):
  Opening range (OR) = high/low of 09:15-09:30 (first `or_bars` bars).
  Skip the day if it is the index's weekly expiry weekday or |gap vs previous close| > `max_gap_pct`.
  At `classify_bar` (default 10:30):
    TREND day  : close beyond OR by >= `trend_k` x OR width.
                 Enter in that direction on the first later 15-min close that is still beyond OR by >= trend_k x width.
                 Stop = far side of OR. Exit at 15:15. One trade.
    RANGE day  : close inside OR and OR width >= `min_range_pts`.
                 After a 1-min bar touches an OR edge, enter against it on the next 15-min close back inside the OR.
                 Stop = edge +/- `range_stop_k` x width. Target = opposite edge. Exit 15:15. Max `max_range_trades`.
    Otherwise  : no trade.
Pricing: same delta / theta / slippage / Dhan-cost model as backtest_12m.py and pa_strategy.py.
"""
import sys, json, argparse
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pa_study import load, tstat
from pa_strategy import PARAMS, DELTA, THETA_PCT, SLIP, costs

def expiry_weekday(sym, date):
    """Weekly expiry weekday by date (Mon=0). NIFTY: Thu until Aug 2025, Tue after. SENSEX: Fri until Dec 2024, Tue Jan-Aug 2025, Thu after."""
    d = pd.Timestamp(date); d = d.tz_localize(None) if d.tzinfo is not None else d
    if sym == "nifty":
        return 3 if d < pd.Timestamp("2025-09-01") else 1
    if d < pd.Timestamp("2025-01-01"): return 4
    if d < pd.Timestamp("2025-09-01"): return 1
    return 3
DEFAULTS = dict(or_bars=15, classify_bar=75, trend_k=0.5, min_range_pts={"nifty": 60.0, "sensex": 210.0},
                range_stop_k=0.25, max_range_trades=2, max_gap_pct=1.0, exit_bar=360, skip_expiry=True)


class TrendRangeStrategy:
    def __init__(self, sym, **kw):
        self.sym = sym
        self.p = {**DEFAULTS, **kw}
        if isinstance(self.p["min_range_pts"], dict):
            self.p["min_range_pts"] = self.p["min_range_pts"][sym]

    def classify(self, g, prev):
        p = self.p; h, l, c = g.high.values, g.low.values, g.close.values
        orh, orl = h[:p["or_bars"]].max(), l[:p["or_bars"]].min(); w = orh - orl
        if prev is not None and p["max_gap_pct"] is not None:
            gap = abs(g.open.values[0] - prev.close.iloc[-1]) / prev.close.iloc[-1] * 100
            if gap > p["max_gap_pct"]:
                return "SKIP_GAP", orh, orl, w
        if p["skip_expiry"] and pd.Timestamp(g.timestamp.iloc[0]).weekday() == expiry_weekday(self.sym, g.timestamp.iloc[0]):
            return "SKIP_EXPIRY", orh, orl, w
        pc = c[p["classify_bar"]]
        if pc >= orh + p["trend_k"] * w: return "TREND_UP", orh, orl, w
        if pc <= orl - p["trend_k"] * w: return "TREND_DOWN", orh, orl, w
        if orl <= pc <= orh and w >= p["min_range_pts"]: return "RANGE", orh, orl, w
        return "NONE", orh, orl, w

    def signals(self, g, prev):
        """Yield candidate entries in time order: dicts with bar, dir, stop, target, exit_bar, kind."""
        p = self.p; kind, orh, orl, w = self.classify(g, prev)
        h, l, c = g.high.values, g.low.values, g.close.values
        cb, xb = p["classify_bar"], p["exit_bar"]
        out = []
        closes15 = [i for i in range(cb + 15, xb, 15) if i % 15 == 14 or (i - 14) % 15 == 0]  # 15-min closes: bars 14, 29, 44, ...
        closes15 = [i for i in range(14, xb, 15) if i > cb]
        if kind in ("TREND_UP", "TREND_DOWN"):
            d = 1 if kind == "TREND_UP" else -1
            for i in closes15:
                held = (c[i] >= orh + p["trend_k"] * w) if d > 0 else (c[i] <= orl - p["trend_k"] * w)
                if held:
                    out.append({"bar": i, "dir": d, "stop": orl if d > 0 else orh, "target": None, "exit_bar": xb, "kind": "TREND"}); break
        elif kind == "RANGE":
            touched = None  # ("HI"|"LO", bar)
            for i in range(cb + 1, xb):
                if h[i] >= orh: touched = ("HI", i)
                elif l[i] <= orl: touched = ("LO", i)
                if touched and (i % 15 == 14):
                    side, tb = touched
                    if side == "HI" and c[i] < orh:
                        out.append({"bar": i, "dir": -1, "stop": orh + p["range_stop_k"] * w, "target": orl, "exit_bar": xb, "kind": "RANGE"}); touched = None
                    elif side == "LO" and c[i] > orl:
                        out.append({"bar": i, "dir": 1, "stop": orl - p["range_stop_k"] * w, "target": orh, "exit_bar": xb, "kind": "RANGE"}); touched = None
        return kind, out


def run(sym, days, dates, strat, use_theta=True, max_range_trades=None):
    P = PARAMS[sym]; trades = []; kinds = []
    mrt = max_range_trades if max_range_trades is not None else strat.p["max_range_trades"]
    for j, g in enumerate(days):
        prev = days[j - 1] if j > 0 and (dates[j] - dates[j - 1]).days <= 5 else None
        kind, sigs = strat.signals(g, prev); kinds.append(kind)
        h, l, c = g.high.values, g.low.values, g.close.values
        busy_until, n_today = -1, 0
        for s in sigs:
            i, d = s["bar"], s["dir"]
            if i <= busy_until or i >= s["exit_bar"]: continue
            if s["kind"] == "RANGE" and n_today >= mrt: break
            e = c[i]; p0 = e * P["prem_pct"]; fill_in = p0 * (1 + SLIP)
            stop, tgt, xb = s["stop"], s["target"], s["exit_bar"]
            exit_i, exit_spot, reason = xb, c[xb], "time"
            for k in range(i + 1, xb + 1):
                adverse = l[k] if d > 0 else h[k]; fav = h[k] if d > 0 else l[k]
                if (adverse <= stop) if d > 0 else (adverse >= stop):
                    exit_i, exit_spot, reason = k, stop, "stop"; break
                if tgt is not None and ((fav >= tgt) if d > 0 else (fav <= tgt)):
                    exit_i, exit_spot, reason = k, tgt, "target"; break
            held = exit_i - i
            prem_out = max(0.5, p0 + DELTA * d * (exit_spot - e) - (p0 * THETA_PCT * held if use_theta else 0)) * (1 - SLIP)
            gross = (prem_out - fill_in) * P["lot"]; cst = costs(fill_in * P["lot"], prem_out * P["lot"])
            trades.append({"date": dates[j], "kind": s["kind"], "bar": i, "dir": d, "spot_pts": (exit_spot - e) * d, "held": held,
                           "gross": gross, "costs": cst, "net": gross - cst, "reason": reason})
            busy_until = exit_i; n_today += 1
    return pd.DataFrame(trades), pd.Series(kinds).value_counts().to_dict()


def summarize(t, label=""):
    if t.empty: return {"label": label, "n": 0}
    w, L = t[t.net > 0], t[t.net <= 0]; cum = t.net.cumsum(); dd = float((cum - cum.cummax()).min())
    pf = float(w.net.sum() / -L.net.sum()) if len(L) and L.net.sum() < 0 else float("inf")
    yr = pd.to_datetime(t.date.astype(str)).dt.year
    return {"label": label, "n": int(len(t)), "net": float(t.net.sum()), "gross": float(t.gross.sum()), "costs": float(t.costs.sum()),
            "win": float(len(w) / len(t)), "pf": pf, "net_per_trade": float(t.net.mean()), "t": float(tstat(t.net)), "maxdd": dd,
            "hold": float(t.held.mean()), "spot_pts": float(t.spot_pts.mean()),
            "nov25": float(t[yr == 2025].net.sum()), "y2026": float(t[yr == 2026].net.sum()),
            "reasons": t.reason.value_counts().to_dict()}


def fmt(s):
    if s.get("n", 0) == 0: return f"{s.get('label',''):38s} no trades"
    return (f"{s['label']:38s} n={s['n']:3d} net {s['net']:+9.0f} (gross {s['gross']:+8.0f}, costs {s['costs']:6.0f}) | win {100*s['win']:4.1f}% PF {s['pf']:4.2f} "
            f"| net/trade {s['net_per_trade']:+6.0f} t={s['t']:+5.2f} | maxDD {s['maxdd']:7.0f} | hold {s['hold']:4.0f}m spot {s['spot_pts']:+6.1f} | Nov25 {s['nov25']:+7.0f} 2026 {s['y2026']:+8.0f}")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--grid", action="store_true"); ap.add_argument("--out", default=None)
    ap.add_argument("--data", default="12m", help="data set tag: 12m (bundled) or 3y (Dhan download)"); a = ap.parse_args()
    results = {}
    for sym in ("nifty", "sensex"):
        df = load(sym, a.data); days = [g.reset_index(drop=True) for _, g in df.groupby("date", sort=True)]; dates = [g.date.iloc[0] for g in days]
        print(f"\n===== {sym.upper()} ({len(days)} sessions, data={a.data}) =====")
        strat = TrendRangeStrategy(sym)
        t, kinds = run(sym, days, dates, strat)
        print("day classification:", kinds)
        if not t.empty:
            t["year"] = pd.to_datetime(t.date.astype(str)).dt.year
            print("per-year net:", {int(y): (int(len(x)), round(float(x.net.sum()))) for y, x in t.groupby("year")})
        rows = [summarize(t, "BASE spec (options, with theta)"),
                summarize(t[t.kind == "TREND"], "  trend-day trades only"),
                summarize(t[t.kind == "RANGE"], "  range-day trades only"),
                summarize(run(sym, days, dates, strat, use_theta=False)[0], "BASE spec, no theta (futures / deep ITM)"),
                summarize(run(sym, days, dates, TrendRangeStrategy(sym, skip_expiry=False, max_gap_pct=None))[0], "  without expiry/gap filters")]
        for r in rows: print(fmt(r))
        print("exit reasons:", rows[0].get("reasons"))
        results[sym] = {"kinds": kinds, "rows": rows}
        if a.grid:
            print("-- parameter grid (net INR with theta / n) --")
            grid = {}
            for cb, lab in ((60, "10:15"), (75, "10:30"), (90, "10:45"), (105, "11:00")):
                cells = []
                for k in (0.3, 0.5, 0.75):
                    tt, _ = run(sym, days, dates, TrendRangeStrategy(sym, classify_bar=cb, trend_k=k))
                    tr_, rg_ = tt[tt.kind == "TREND"], tt[tt.kind == "RANGE"]
                    cells.append(f"k{k}: all {tt.net.sum():+7.0f}/{len(tt):3d}  T {tr_.net.sum():+7.0f}/{len(tr_):2d}  R {rg_.net.sum():+7.0f}/{len(rg_):2d}")
                    grid[f"{lab}_k{k}"] = {"net": float(tt.net.sum()), "n": int(len(tt)), "trend_net": float(tr_.net.sum()), "range_net": float(rg_.net.sum())}
                print(f"  classify {lab}: " + " | ".join(cells))
            cells = []
            for rs in (0.15, 0.25, 0.4):
                for mr in (1, 2, 3):
                    tt, _ = run(sym, days, dates, TrendRangeStrategy(sym, range_stop_k=rs, max_range_trades=mr)); rg_ = tt[tt.kind == "RANGE"]
                    cells.append(f"stop{rs}/max{mr}: {rg_.net.sum():+7.0f}/{len(rg_):2d}")
            print("  range-day variants (range trades only): " + " | ".join(cells))
            cells = []
            for ob in (5, 15, 30):
                tt, _ = run(sym, days, dates, TrendRangeStrategy(sym, or_bars=ob)); cells.append(f"OR{ob}: {tt.net.sum():+7.0f}/{len(tt):3d}")
            print("  opening-range window: " + " | ".join(cells))
            results[sym]["grid"] = grid
        if a.out:
            t.to_csv(f"{a.out}_{sym}_trades.csv", index=False)
    if a.out:
        json.dump(results, open(f"{a.out}_summary.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
