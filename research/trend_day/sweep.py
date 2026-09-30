import sys; sys.path.insert(0, ".")
import pandas as pd, numpy as np
from trend_day.rules import Rules
from trend_day.backtest import run, tstat, LOTS
from pa_engine.candles import load_sessions
from pa_engine.config import CostConfig
from pa_engine.costs import evaluate
V = {
 "base": {},
 "beyond_pdr": {"require_beyond_pdr": True},
 "confirm3": {"confirm_closes": 3},
 "or60": {"or_minutes": 60},
 "or60_pdr": {"or_minutes": 60, "require_beyond_pdr": True},
 "deadline_10:30": {"deadline": "10:30"},
 "deadline_12:30": {"deadline": "12:30", "last_entry": "14:00"},
 "trail_none": {"trail": "none"},
 "buffer_0.25": {"stop_buffer_atr": 0.25},
 "max_trades_2": {"max_trades": 2},
 "pdr_or60_trail_none": {"or_minutes": 60, "require_beyond_pdr": True, "trail": "none"},
}
rows = []
fut = CostConfig(instrument="futures")
for sym in ("NIFTY", "SENSEX"):
    ss = load_sessions(f"data/{sym.lower()}_3y_1min.csv.gz")
    for name, kv in V.items():
        r = Rules(**kv); d, t = run(sym, ss, r)
        acc = d[d.type != "BALANCED"].dropna(subset=["drift"])
        row = {"sym": sym, "variant": name, "acc%": round(100 * len(acc) / len(d)), "drift": round(acc.drift.mean(), 1), "drift_t": round(tstat(acc.drift), 2), "p_cont": round((acc.drift > 0).mean(), 2)}
        if len(t):
            t["fut"] = [evaluate(fut, x.direction, x.entry, x.exit, x.hold_min, LOTS[sym]).net for x in t.itertuples()]
            row.update({"n": len(t), "win": round((t.pts > 0).mean(), 2), "pts": round(t.pts.mean(), 1), "R": round(t.r.mean(), 2), "opt_net": round(t.net.sum()), "fut_net": round(t.fut.sum()),
                        "yr_fut": " ".join(f"{y[2:]}:{int(v / 1000):+d}k" for y, v in t.groupby("year").fut.sum().items())})
        rows.append(row); print(row, flush=True)
df = pd.DataFrame(rows); df.to_csv("research/trend_day/sweep.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30); print(df.to_string(index=False))
