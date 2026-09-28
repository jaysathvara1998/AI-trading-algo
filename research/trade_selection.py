"""
Trade-selection study: does anything observable at entry separate the engine's profitable trades from the rest?

Input : 3-year replay trade logs from backtest_12m.py (--data 3y, new engine) + the 3y bar files.
Output: gross P&L per trade by feature bucket, with n, t-stat, per-year sign and cross-index agreement.
Pass  : a rule counts only if the selected subset earns > COST_FLOOR gross per trade on BOTH indices,
        in EVERY year, with >= 0.5 trades/day. Fitting window 2023Q4-2024; 2025 and 2026 are held out.

    .venv/bin/python research/trade_selection.py --trades-dir /path/with/e3y_<SYM>_new_trades.csv
"""
import sys, argparse
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pa_study import load, tstat
from trend_range_strategy import expiry_weekday

COST_FLOOR = 73.0      # statutory cost per round trip (1 lot); spread would add more
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)

def features(sym, trades):
    days = {g.date.iloc[0]: g.reset_index(drop=True) for _, g in load(sym, "3y").groupby("date", sort=True)}
    dates = sorted(days); prev_of = {d: dates[i - 1] for i, d in enumerate(dates) if i > 0}
    t = trades.copy(); t["entry_ts"] = pd.to_datetime(t.entry_ts); t["date"] = t.entry_ts.dt.date
    t = t.sort_values("entry_ts").reset_index(drop=True)
    rows = []
    for _, r in t.iterrows():
        g = days.get(r.date)
        if g is None: rows.append({}); continue
        m = (r.entry_ts.hour - 9) * 60 + r.entry_ts.minute - 15; m = int(max(0, min(374, m)))
        c, h, l = g.close.values, g.high.values, g.low.values
        p = days.get(prev_of.get(r.date))
        pdh, pdl, pdc = (p.high.max(), p.low.min(), p.close.iloc[-1]) if p is not None else (np.nan, np.nan, np.nan)
        orh, orl = h[:15].max(), l[:15].min(); orw = orh - orl
        atr = float(np.median((h[:m + 1] - l[:m + 1])[-30:])) if m >= 5 else float(h[0] - l[0])
        f = {
            "minute": m, "hour_bucket": "09:15-09:45" if m < 30 else "09:45-11:00" if m < 105 else "11:00-13:30" if m < 255 else "13:30-15:15",
            "dow": r.entry_ts.weekday(), "expiry": int(r.entry_ts.weekday() == expiry_weekday(sym, r.entry_ts)),
            "gap_pct": (g.open.values[0] - pdc) / pdc * 100 if p is not None else np.nan,
            "or_width_atr": orw / atr if atr > 0 else np.nan,
            "or_pos": (c[m] - orl) / orw if orw > 0 else np.nan,             # <0 below OR, >1 above OR
            "beyond_or": int(c[m] > orh or c[m] < orl),
            "ret5": (c[m] - c[max(0, m - 5)]) / atr if atr > 0 else np.nan,
            "ret15": (c[m] - c[max(0, m - 15)]) / atr if atr > 0 else np.nan,
            "ret30": (c[m] - c[max(0, m - 30)]) / atr if atr > 0 else np.nan,
            "day_move_atr": (c[m] - g.open.values[0]) / atr if atr > 0 else np.nan,
            "dist_pdh_atr": (pdh - c[m]) / atr if p is not None and atr > 0 else np.nan,
            "dist_pdl_atr": (c[m] - pdl) / atr if p is not None and atr > 0 else np.nan,
            "range_so_far_atr": (h[:m + 1].max() - l[:m + 1].min()) / atr if atr > 0 else np.nan,
            "prev_day_ret": (pdc - p.open.iloc[0]) / p.open.iloc[0] * 100 if p is not None else np.nan,
            "atr": atr,
        }
        rows.append(f)
    F = pd.DataFrame(rows, index=t.index); t = pd.concat([t, F], axis=1)
    t["dir"] = np.where(t.side == "CALL", 1, -1)
    t["with_day"] = np.sign(t.day_move_atr) * t.dir            # +1 = trade in the direction of the day's move so far
    t["with_ret15"] = np.sign(t.ret15) * t.dir
    t["trade_no"] = t.groupby("date").cumcount() + 1
    t["prev_outcome"] = t.groupby("date").net_pnl.shift(1).apply(lambda x: "none" if pd.isna(x) else ("win" if x > 0 else "loss"))
    t["year"] = t.entry_ts.dt.year
    return t

def bucketize(t):
    b = pd.DataFrame(index=t.index)
    b["hour"] = t.hour_bucket
    b["trade_no"] = t.trade_no.clip(upper=4).astype(str)
    b["side"] = t.side
    b["prev_outcome"] = t.prev_outcome
    b["dow"] = t.dow.map({0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri"})
    b["expiry"] = t.expiry.map({0: "no", 1: "yes"})
    b["gap"] = pd.cut(t.gap_pct.abs(), [-0.01, 0.2, 0.5, 1.0, 99], labels=["<0.2%", "0.2-0.5%", "0.5-1%", ">1%"]).astype(str)
    b["or_width"] = pd.cut(t.or_width_atr, [0, 3, 5, 8, 999], labels=["<3atr", "3-5", "5-8", ">8"]).astype(str)
    b["vs_OR"] = np.select([t.or_pos < 0, t.or_pos > 1], ["below OR", "above OR"], "inside OR")
    b["with_day"] = t.with_day.map({1: "with day", -1: "against day", 0: "flat"})
    b["with_ret15"] = t.with_ret15.map({1: "with 15m", -1: "against 15m", 0: "flat"})
    b["ret15_abs"] = pd.cut(t.ret15.abs(), [-0.01, 1, 2, 4, 99], labels=["<1atr", "1-2", "2-4", ">4"]).astype(str)
    b["day_move"] = pd.cut(t.day_move_atr.abs(), [-0.01, 3, 6, 12, 999], labels=["<3atr", "3-6", "6-12", ">12"]).astype(str)
    b["near_pdh"] = pd.cut(t.dist_pdh_atr, [-999, -0.5, 0.5, 2, 999], labels=["above PDH", "at PDH", "0.5-2atr below", "far below"]).astype(str)
    b["near_pdl"] = pd.cut(t.dist_pdl_atr, [-999, -0.5, 0.5, 2, 999], labels=["below PDL", "at PDL", "0.5-2atr above", "far above"]).astype(str)
    b["range_so_far"] = pd.cut(t.range_so_far_atr, [0, 8, 12, 18, 999], labels=["<8atr", "8-12", "12-18", ">18"]).astype(str)
    b["prev_day"] = np.select([t.prev_day_ret > 0.5, t.prev_day_ret < -0.5], ["prev up", "prev down"], "prev flat")
    return b

def table(sym, t, b):
    out = []
    for col in b.columns:
        for val, idx in b.groupby(col).groups.items():
            x = t.loc[idx]
            if len(x) < 40: continue
            yrs = x.groupby("year").gross_pnl.mean()
            out.append({"feature": col, "value": val, "n": len(x), "per_day": round(len(x) / t.date.nunique(), 2), "gross/trade": round(x.gross_pnl.mean(), 1),
                        "t": round(tstat(x.gross_pnl), 2), "years>floor": f"{int((yrs > COST_FLOOR).sum())}/{len(yrs)}", "yrs": " ".join(f"{int(y)}:{v:+.0f}" for y, v in yrs.items())})
    return pd.DataFrame(out)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--trades-dir", required=True); a = ap.parse_args()
    T, B = {}, {}
    for sym in ("nifty", "sensex"):
        tr = pd.read_csv(f"{a.trades_dir}/e3y_{sym.upper()}_new_trades.csv")
        T[sym] = features(sym, tr); B[sym] = bucketize(T[sym])
        print(f"\n===== {sym.upper()}: {len(T[sym])} trades, gross/trade {T[sym].gross_pnl.mean():+.1f}, cost floor {COST_FLOOR} =====")
        tab = table(sym, T[sym], B[sym]); print(tab.sort_values("gross/trade", ascending=False).to_string(index=False))
    # ---- cross-index agreement: buckets above the floor on BOTH indices, significant, and consistent by year
    print("\n===== BUCKETS CLEARING THE COST FLOOR ON BOTH INDICES =====")
    tn, ts = table("nifty", T["nifty"], B["nifty"]), table("sensex", T["sensex"], B["sensex"])
    m = tn.merge(ts, on=["feature", "value"], suffixes=("_N", "_S"))
    hits = m[(m["gross/trade_N"] > COST_FLOOR) & (m["gross/trade_S"] > COST_FLOOR)]
    print(hits.to_string(index=False) if not hits.empty else "none")
    # ---- held-out check of the best single rule chosen on the fit window (2023-2024) only
    print("\n===== FIT ON 2023-2024, TEST ON 2025 AND 2026 (single-bucket rules ranked on the fit window, both indices pooled) =====")
    fit_rows = []
    for col in B["nifty"].columns:
        for val in set(B["nifty"][col]).intersection(set(B["sensex"][col])):
            sel = {s: T[s][(B[s][col] == val)] for s in T}
            fit = pd.concat([sel[s][sel[s].year <= 2024] for s in T]); tst25 = pd.concat([sel[s][sel[s].year == 2025] for s in T]); tst26 = pd.concat([sel[s][sel[s].year == 2026] for s in T])
            if len(fit) < 80 or len(tst25) < 40: continue
            fit_rows.append({"rule": f"{col}={val}", "fit_n": len(fit), "fit_gross": round(fit.gross_pnl.mean(), 1), "fit_t": round(tstat(fit.gross_pnl), 2),
                             "2025_n": len(tst25), "2025_gross": round(tst25.gross_pnl.mean(), 1), "2026_n": len(tst26), "2026_gross": round(tst26.gross_pnl.mean(), 1) if len(tst26) else np.nan,
                             "N_2025": round(sel["nifty"][sel["nifty"].year == 2025].gross_pnl.mean(), 1), "S_2025": round(sel["sensex"][sel["sensex"].year == 2025].gross_pnl.mean(), 1)})
    fr = pd.DataFrame(fit_rows).sort_values("fit_gross", ascending=False)
    print(fr.head(12).to_string(index=False))
    # ---- simple 2-feature combo: hour x with_ret15, and hour x trade_no
    print("\n===== TWO-FEATURE COMBINATIONS (gross/trade, n) =====")
    for s in T:
        t, b = T[s], B[s]
        for f1, f2 in (("hour", "with_ret15"), ("hour", "trade_no"), ("hour", "side"), ("with_day", "with_ret15")):
            piv = t.assign(a=b[f1], c=b[f2]).pivot_table(index="a", columns="c", values="gross_pnl", aggfunc=["mean", "size"]).round(0)
            print(f"\n{s.upper()} {f1} x {f2}:\n{piv.to_string()}")

if __name__ == "__main__":
    main()
