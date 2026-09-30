"""Score the day-type call and the pullback trade on historical sessions. Usage: python -m trend_day NIFTY data/nifty_3y_1min.csv.gz [--set key=value ...]"""
from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from pa_engine.backtest import EXPIRY_WEEKDAY
from pa_engine.candles import load_sessions
from pa_engine.config import CostConfig
from pa_engine.costs import evaluate
from .rules import Rules, run_day

LOTS = {"NIFTY": 65, "SENSEX": 20, "BANKNIFTY": 30}


def run(symbol: str, sessions, r: Rules, skip_expiry=True):
    skip = EXPIRY_WEEKDAY.get(symbol.upper())
    days, trades = [], []
    ranges = []
    for s in sessions:
        avg_range = float(np.mean(ranges[-10:])) if len(ranges) >= 5 else 0.0
        ranges.append(max(c.high for c in s.candles) - min(c.low for c in s.candles))
        if skip_expiry and skip is not None and pd.Timestamp(s.day).weekday() == skip(s.day):
            continue
        res = run_day(s.candles, s.prev_high, s.prev_low, r, str(s.day), avg_range, s.prev_close or 0.0)
        days.append({"day": res.day, "type": res.day_type, "accept_minute": res.accept_minute, "drift": res.drift_to_close, "or_ratio": res.or_ratio, "gap_pct": res.gap_pct})
        trades += res.trades
    d = pd.DataFrame(days); t = pd.DataFrame(trades)
    lot = LOTS[symbol.upper()]; cost = CostConfig()
    if len(t):
        eco = [evaluate(cost, r_.direction, r_.entry, r_.exit, r_.hold_min, lot) for r_ in t.itertuples()]
        t["net"] = [e.net for e in eco]; t["year"] = t.day.str[:4]; t["r"] = t.pts / t.risk_pts
    d["year"] = d.day.str[:4]
    return d, t


def tstat(x):
    x = np.asarray(x, float); return x.mean() / (x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 2 and x.std() > 0 else 0.0


def report(symbol, d, t):
    out = []
    out.append(f"== {symbol}: {len(d)} sessions, day types: " + ", ".join(f"{k} {v} ({100 * v / len(d):.0f}%)" for k, v in d.type.value_counts().items()))
    acc = d[d.type != "BALANCED"].dropna(subset=["drift"])
    out.append(f"   accepted days: drift from acceptance to flatten, in the accepted direction: mean {acc.drift.mean():+.1f} pts, median {acc.drift.median():+.1f}, "
               f"t {tstat(acc.drift):.2f}, share continuing {100 * (acc.drift > 0).mean():.0f}%")
    g = acc.groupby("year").drift.agg(["size", "mean", "median", tstat, lambda x: (x > 0).mean()]).round(2)
    g.columns = ["n", "mean", "median", "t", "p_continue"]
    out.append("   by year:\n" + g.to_string())
    if len(t) == 0:
        out.append("   no trades"); return "\n".join(out)
    w = t[t.net > 0]; L = t[t.net <= 0]
    out.append(f"   trades {len(t)} ({len(t) / (len(d) / 250):.0f}/yr), win {100 * (t.pts > 0).mean():.0f}%, avg win {w.pts.mean():+.1f} pts, avg loss {L.pts.mean():+.1f} pts, "
               f"expectancy {t.pts.mean():+.1f} pts ({t.r.mean():+.2f}R), option net Rs {t.net.sum():,.0f}, PF {w.net.sum() / max(1e-9, -L.net.sum()):.2f}, "
               f"max DD Rs {(t.net.cumsum() - t.net.cumsum().cummax()).min():,.0f}, avg hold {t.hold_min.mean():.0f} min")
    g = t.groupby("year").agg(n=("pts", "size"), win=("pts", lambda x: round((x > 0).mean(), 2)), pts=("pts", "mean"), R=("r", "mean"), net=("net", "sum"), t=("net", tstat)).round(2)
    out.append("   trades by year:\n" + g.to_string())
    out.append("   exits: " + str(t.reason.value_counts().to_dict()))
    return "\n".join(out)


def main(argv):
    symbol, data = argv[0], argv[1]
    r = Rules()
    if "--set" in argv:
        for kv in argv[argv.index("--set") + 1:]:
            k, v = kv.split("=", 1); cur = getattr(r, k)
            setattr(r, k, type(cur)(v) if not isinstance(cur, bool) else v.lower() in ("1", "true", "yes"))
    sessions = load_sessions(data)
    d, t = run(symbol, sessions, r)
    print(report(symbol, d, t))
    out = Path("research/trend_day"); out.mkdir(parents=True, exist_ok=True)
    tag = "_".join(a.replace("=", "-") for a in argv[argv.index("--set") + 1:]) if "--set" in argv else "base"
    d.to_csv(out / f"{symbol}_{tag}_days.csv", index=False); t.to_csv(out / f"{symbol}_{tag}_trades.csv", index=False)


if __name__ == "__main__":
    main(sys.argv[1:])
