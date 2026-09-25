"""Candidate price-action strategies vs the current momentum trigger, same premium/cost model as backtest_12m.py."""
import sys, json
import numpy as np, pandas as pd
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from pa_study import load, tstat
pd.set_option("display.width", 220)

PARAMS = {"nifty": dict(lot=65, prem_pct=0.007, noise15=13.7, sl_pts=12.0), "sensex": dict(lot=20, prem_pct=0.007, noise15=46.8, sl_pts=40.0)}
DELTA, THETA_PCT, SLIP = 0.72, 0.00025, 0.001

def costs(buy_v, sell_v):
    txn = 0.00035 * (buy_v + sell_v); return 40 + 0.001 * sell_v + txn + 0.000001 * (buy_v + sell_v) + 0.00003 * buy_v + 0.18 * (40 + txn)

def simulate(days, dates, signal_fn, sym, hold_to=360, use_theta=True):
    """signal_fn(g, prev_g) -> list of dicts {bar, dir, stop(spot), target(spot or None), exit_bar}. Positions are sequential, 1 at a time."""
    P = PARAMS[sym]; trades = []
    for j, g in enumerate(days):
        prev = days[j - 1] if j > 0 and (dates[j] - dates[j - 1]).days <= 5 else None
        sigs = signal_fn(g, prev, P)
        h, l, c = g.high.values, g.low.values, g.close.values
        busy_until = -1
        for s in sigs:
            i, d = s["bar"], s["dir"]
            if i <= busy_until or i >= hold_to: continue
            e = c[i]; p0 = e * P["prem_pct"]; fill_in = p0 * (1 + SLIP)
            stop, tgt, xb = s["stop"], s.get("target"), min(s.get("exit_bar", hold_to), hold_to)
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
            trades.append({"date": dates[j], "bar": i, "dir": d, "spot_pts": (exit_spot - e) * d, "held": held, "gross": gross, "costs": cst, "net": gross - cst, "reason": reason})
            busy_until = exit_i
    return pd.DataFrame(trades)

def report(name, t, n_days):
    if t.empty: print(f"{name:34s} no trades"); return {}
    w = t[t.net > 0]; L = t[t.net <= 0]; cum = t.net.cumsum(); dd = (cum - cum.cummax()).min()
    pf = w.net.sum() / -L.net.sum() if len(L) and L.net.sum() < 0 else np.inf
    t["month"] = pd.to_datetime(t.date.astype(str)).dt.to_period("M").astype(str)
    mo = t.groupby("month").net.sum().round(0)
    sub = {"2025-11": t[t.month.str.startswith("2025")].net.sum(), "2026": t[t.month.str.startswith("2026")].net.sum()}
    print(f"{name:34s} n={len(t):4d} ({len(t)/n_days:.2f}/day) spot {t.spot_pts.mean():+6.1f} pts | gross {t.gross.sum():9.0f} costs {t.costs.sum():7.0f} net {t.net.sum():9.0f} | win {100*len(w)/len(t):4.1f}% PF {pf:4.2f} | net/trade {t.net.mean():+6.0f} (t={tstat(t.net):+5.2f}) | maxDD {dd:8.0f} | hold {t.held.mean():4.0f}m | Nov25 {sub['2025-11']:+8.0f} 2026 {sub['2026']:+8.0f}")
    return {"n": int(len(t)), "net": float(t.net.sum()), "gross": float(t.gross.sum()), "costs": float(t.costs.sum()), "win": float(len(w) / len(t)), "pf": float(pf), "net_per_trade": float(t.net.mean()), "t": float(tstat(t.net)), "maxdd": float(dd), "hold": float(t.held.mean()), "monthly": mo.to_dict(), "sub": {k: float(v) for k, v in sub.items()}, "reasons": t.reason.value_counts().to_dict()}

# ---------------- strategies ----------------
def s_momentum15(g, prev, P):
    """Proxy of the current algo trigger: first bar where 15-min move >= noise15, trade with it; fixed stop/target in spot pts; 1/day."""
    c = g.close.values
    for i in range(15, 300):
        mv = c[i] - c[i - 15]
        if abs(mv) >= P["noise15"]:
            d = int(np.sign(mv)); return [{"bar": i, "dir": d, "stop": c[i] - d * P["sl_pts"], "target": c[i] + d * 1.5 * P["sl_pts"], "exit_bar": 360}]
    return []

def s_momentum15_multi(g, prev, P):
    """Same trigger, re-armed after each exit, up to 12/day (like the live cap)."""
    c = g.close.values; out = []
    for i in range(15, 300):
        mv = c[i] - c[i - 15]
        if abs(mv) >= P["noise15"]:
            d = int(np.sign(mv)); out.append({"bar": i, "dir": d, "stop": c[i] - d * P["sl_pts"], "target": c[i] + d * 1.5 * P["sl_pts"], "exit_bar": 360})
    return out[:12]

def s_orb15(g, prev, P):
    h, l, c = g.high.values, g.low.values, g.close.values
    orh, orl = h[:15].max(), l[:15].min(); orr = orh - orl; mid = (orh + orl) / 2
    for i in range(15, 165):
        if c[i] > orh: return [{"bar": i, "dir": 1, "stop": mid, "target": c[i] + orr, "exit_bar": 360}]
        if c[i] < orl: return [{"bar": i, "dir": -1, "stop": mid, "target": c[i] - orr, "exit_bar": 360}]
    return []

def make_fade30(thr_pct, exit_bar, stop_mult=1.0, target="open"):
    def f(g, prev, P):
        c, h, l, o = g.close.values, g.high.values, g.low.values, g.open.values[0]
        mv = c[29] - o
        if abs(mv) / o * 100 < thr_pct or mv == 0: return []
        d = -int(np.sign(mv)); rng = h[:30].max() - l[:30].min()
        stop = (l[:30].min() - 0.1 * rng) if d > 0 else (h[:30].max() + 0.1 * rng)
        stop = c[30] - d * stop_mult * abs(c[30] - stop)
        tgt = o if target == "open" else None
        return [{"bar": 30, "dir": d, "stop": stop, "target": tgt, "exit_bar": exit_bar}]
    return f

def make_gapfade(thr_pct, exit_bar=360):
    def f(g, prev, P):
        if prev is None: return []
        pdc = prev.close.iloc[-1]; o = g.open.values[0]; gap = (o - pdc) / pdc * 100
        if abs(gap) < thr_pct: return []
        d = -int(np.sign(gap)); h, l, c = g.high.values, g.low.values, g.close.values
        stop = (l[:5].min() - 0.5 * abs(o - pdc)) if d > 0 else (h[:5].max() + 0.5 * abs(o - pdc))
        return [{"bar": 5, "dir": d, "stop": stop, "target": pdc, "exit_bar": exit_bar}]
    return f

def make_rev15(k, hold=15):
    def f(g, prev, P):
        c = g.close.values
        for i in range(15, 61):
            mv = c[i] - c[i - 15]
            if abs(mv) >= k * P["noise15"]:
                d = -int(np.sign(mv)); ext = (g.high.values[i-15:i+1].max() if d < 0 else g.low.values[i-15:i+1].min())
                return [{"bar": i, "dir": d, "stop": ext + (-d) * 0.25 * abs(mv), "target": None, "exit_bar": i + hold}]
        return []
    return f

def make_trendday(T, exit_bar=360):
    def f(g, prev, P):
        h, l, c = g.high.values, g.low.values, g.close.values
        orh, orl = h[:15].max(), l[:15].min(); orr = orh - orl; p = c[T]
        if p >= orh + 0.5 * orr: d = 1
        elif p <= orl - 0.5 * orr: d = -1
        else: return []
        stop = (orh if d > 0 else orl)  # back inside the opening range = thesis failed
        return [{"bar": T, "dir": d, "stop": stop, "target": None, "exit_bar": exit_bar}]
    return f

STRATS = {
    "CTRL momentum15 (1/day)": s_momentum15,
    "CTRL momentum15 (<=12/day)": s_momentum15_multi,
    "CTRL ORB15 stop=mid tgt=1xOR": s_orb15,
    "fade30 any, exit 11:15": make_fade30(0.0, 120),
    "fade30 any, exit 15:15": make_fade30(0.0, 360),
    "fade30 >=0.3%, exit 15:15": make_fade30(0.3, 360),
    "fade30 >=0.3%, exit 15:15 notgt": make_fade30(0.3, 360, target=None),
    "gapfade >=0.3%": make_gapfade(0.3),
    "gapfade >=0.5%": make_gapfade(0.5),
    "rev15 k=2 hold15": make_rev15(2.0, 15),
    "rev15 k=2 hold30": make_rev15(2.0, 30),
    "trendday 10:30 -> close": make_trendday(75),
    "trendday 11:00 -> close": make_trendday(105),
}

if __name__ == "__main__":
    res = {}
    for sym in ("nifty", "sensex"):
        df = load(sym); days = [g.reset_index(drop=True) for _, g in df.groupby("date", sort=True)]; dates = [g.date.iloc[0] for g in days]
        print(f"\n===== {sym.upper()} ({len(days)} sessions; costs+theta+slippage, delta {DELTA}) =====")
        res[sym] = {}
        for name, fn in STRATS.items():
            t = simulate(days, dates, fn, sym)
            res[sym][name] = report(name, t, len(days))
        print("  -- same, but WITHOUT theta (i.e. as if traded via futures/deep ITM):")
        for name in ("fade30 >=0.3%, exit 15:15", "gapfade >=0.5%", "trendday 10:30 -> close"):
            t = simulate(days, dates, STRATS[name], sym, use_theta=False); report(name + " [no theta]", t, len(days))
    json.dump(res, open(str(__import__("pathlib").Path(__file__).resolve().parent / "pa_strategy.json"), "w"), indent=1, default=str)
