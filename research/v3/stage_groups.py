"""Forward drift after RETEST events, split by context alignment, regime, level source, hour and penetration."""
import json, sys, collections
import numpy as np, pandas as pd
sys.path.insert(0, ".")
from pa_engine.candles import load_sessions, minute_index
out_dir, data = sys.argv[1], sys.argv[2]
sessions = {str(s.day): s for s in load_sessions(data)}
ev = [json.loads(l) for l in open(f"{out_dir}/events.jsonl")]
sweeps = {e["setup"]: e for e in ev if e["event"] == "LIQUIDITY_EVENT"}
rows = []
for e in ev:
    if e["event"] != "RETEST":
        continue
    s = sessions.get(e["ts"][:10]); m = minute_index(pd.Timestamp(e["ts"])); d = e["direction"]
    if s is None or m + 60 >= len(s.candles):
        continue
    c0 = s.candles[m].close; sw = sweeps.get(e["setup"], {})
    ctx = e.get("trend_ctx", ""); st = e.get("trend_setup", "")
    rows.append({"align_ctx": "with" if (ctx == "BULL" and d > 0) or (ctx == "BEAR" and d < 0) else ("against" if ctx in ("BULL", "BEAR") else "none"),
                 "align_setup": "with" if (st == "BULL" and d > 0) or (st == "BEAR" and d < 0) else ("against" if st in ("BULL", "BEAR") else "none"),
                 "regime": e.get("regime", ""), "source": e.get("source", ""), "hour": pd.Timestamp(e["ts"]).hour,
                 "pen": "deep" if sw.get("penetration_atr", 0) >= 0.3 else "shallow", "year": e["ts"][:4],
                 "f15": (s.candles[m + 15].close - c0) * d, "f30": (s.candles[m + 30].close - c0) * d, "f60": (s.candles[m + 60].close - c0) * d})
df = pd.DataFrame(rows)
def tstat(x): return x.mean() / (x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 2 else 0
print(f"RETEST events with 60m of data: {len(df)}   overall f30 {df.f30.mean():.2f} (t {tstat(df.f30):.2f})  f60 {df.f60.mean():.2f} (t {tstat(df.f60):.2f})")
for col in ("align_ctx", "align_setup", "regime", "source", "hour", "pen", "year"):
    g = df.groupby(col).agg(n=("f30", "size"), f15=("f15", "mean"), f30=("f30", "mean"), t30=("f30", tstat), f60=("f60", "mean"), t60=("f60", tstat)).round(2)
    print(f"\n-- by {col}\n{g.to_string()}")
