"""Volatility-day filter for the momentum engine: trade only on sessions whose 09:15-09:30 opening range is wide
relative to the trailing 20-session median, or whose gap is large. Thresholds chosen on 2023Q4-2024, tested on 2025-2026.
Uses the 3-year replay trade logs (gross P&L per trade) and the 3y bar files."""
import sys, argparse
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pa_study import load, tstat
COST = 73.0
def day_table(sym):
    days = [g.reset_index(drop=True) for _, g in load(sym, "3y").groupby("date", sort=True)]
    rows = []
    for j, g in enumerate(days):
        orw = g.high.values[:15].max() - g.low.values[:15].min()
        pdc = days[j-1].close.iloc[-1] if j > 0 else np.nan
        rows.append({"date": g.date.iloc[0], "orw": orw, "gap": abs(g.open.values[0] - pdc) / pdc * 100 if j > 0 else np.nan, "dow": pd.Timestamp(g.timestamp.iloc[0]).weekday(),
                     "dayrange": g.high.max() - g.low.min()})
    d = pd.DataFrame(rows); d["or_rel"] = d.orw / d.orw.rolling(20, min_periods=10).median().shift(1)
    return d
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--trades-dir", required=True); a = ap.parse_args()
    res = {}
    for sym in ("nifty", "sensex"):
        t = pd.read_csv(f"{a.trades_dir}/e3y_{sym.upper()}_new_trades.csv"); t["date"] = pd.to_datetime(t.entry_ts).dt.date
        d = day_table(sym); t = t.merge(d, on="date", how="left"); t["year"] = pd.to_datetime(t.date.astype(str)).dt.year
        res[sym] = t
        print(f"\n===== {sym.upper()} | all trades: n={len(t)} gross/trade {t.gross_pnl.mean():+.1f} net {t.net_pnl.sum():+.0f} =====")
        print("  or_rel decile -> gross/trade:", t.groupby(pd.qcut(t.or_rel, 5, labels=["q1 narrow","q2","q3","q4","q5 wide"])).gross_pnl.mean().round(0).to_dict())
        print("  gap bucket    -> gross/trade:", t.groupby(pd.cut(t.gap, [-0.01,0.2,0.5,1,99], labels=["<0.2","0.2-0.5","0.5-1",">1"])).gross_pnl.mean().round(0).to_dict())
        print("  Monday vs rest -> gross/trade:", {"Mon": round(t[t.dow==0].gross_pnl.mean()), "Tue-Fri": round(t[t.dow>0].gross_pnl.mean())}, "| Monday or_rel median", round(d[d.dow==0].or_rel.median(),2), "vs rest", round(d[d.dow>0].or_rel.median(),2))
    print("\n===== FILTER SWEEP (rule: trade only if or_rel >= K or gap >= G). Chosen on 2023-2024; 2025/2026 are held out =====")
    print(f"{'rule':22s} | {'fit N g/t':>10s} {'fit S g/t':>10s} | {'25 N':>7s} {'25 S':>7s} | {'26 N':>7s} {'26 S':>7s} | {'t/day N':>7s} {'t/day S':>7s} | {'net25+26 N':>11s} {'net25+26 S':>11s}")
    for K in (1.0, 1.2, 1.5, 2.0, 9.9):
        for G in (0.3, 0.5, 0.75, 9.9):
            cells = []
            for sym in ("nifty", "sensex"):
                t = res[sym]; sel = t[(t.or_rel >= K) | (t.gap >= G)]
                fit = sel[sel.year <= 2024]; y25 = sel[sel.year == 2025]; y26 = sel[sel.year == 2026]
                days_total = t.date.nunique()
                cells.append((fit.gross_pnl.mean() if len(fit) else np.nan, y25.gross_pnl.mean() if len(y25) else np.nan, y26.gross_pnl.mean() if len(y26) else np.nan, len(sel) / days_total, y25.net_pnl.sum() + y26.net_pnl.sum()))
            n, s = cells
            print(f"or_rel>={K:<4} gap>={G:<4} | {n[0]:10.0f} {s[0]:10.0f} | {n[1]:7.0f} {s[1]:7.0f} | {n[2]:7.0f} {s[2]:7.0f} | {n[3]:7.2f} {s[3]:7.2f} | {n[4]:11.0f} {s[4]:11.0f}")
    print("\n===== SAME, MONDAY ONLY (calendar rule, for comparison) =====")
    for sym in ("nifty", "sensex"):
        t = res[sym]; m = t[t.dow == 0]
        print(f"  {sym.upper()}: n={len(m)} ({len(m)/t.date.nunique():.2f}/day) gross/trade fit {m[m.year<=2024].gross_pnl.mean():+.0f} | 2025 {m[m.year==2025].gross_pnl.mean():+.0f} | 2026 {m[m.year==2026].gross_pnl.mean():+.0f} | net 3y {m.net_pnl.sum():+.0f} | maxDD {(m.net_pnl.cumsum()-m.net_pnl.cumsum().cummax()).min():.0f}")
if __name__ == "__main__":
    main()
