"""
Root-cause diagnostics for pa_engine v3 (run after a backtest with --events):
  1. Stage drift: forward move of the underlying (in the setup's direction) after every sweep, shift, retest and
     confirmation event, at 5/15/30/60 minutes. Tells whether any stage carries directional information.
  2. Trade path: MFE / MAE over 60 minutes after each filled entry, ignoring the engine's exits, and the
     probability that k x R is reached before -1R. Tells whether the exits waste moves that were there.
  3. Symmetry: same numbers with the direction flipped (a mean-reversion check).
Usage: .venv/bin/python research/v3/diagnose.py research/v3/nifty_baseline data/nifty_3y_1min.csv.gz
"""
import json, sys, collections
import numpy as np, pandas as pd
sys.path.insert(0, ".")
from pa_engine.candles import load_sessions, minute_index

out_dir, data = sys.argv[1], sys.argv[2]
sessions = {str(s.day): s for s in load_sessions(data)}
ev = [json.loads(l) for l in open(f"{out_dir}/events.jsonl")]
H = (5, 15, 30, 60)

def fwd(day, m, d, h):
    s = sessions.get(day)
    if s is None or m + h >= len(s.candles):
        return None
    return (s.candles[m + h].close - s.candles[m].close) * d

print("=== 1. Forward drift of the underlying after each stage (points in the setup direction; t = mean/se) ===")
rows = []
for stage in ("LIQUIDITY_EVENT", "STRUCTURE_SHIFT", "RETEST", "ENTRY_CONFIRMED"):
    dirs = {}
    for e in ev:
        if e["event"] == "LIQUIDITY_EVENT":
            dirs[e["setup"]] = e["direction"]
    pts = {h: [] for h in H}
    for e in ev:
        if e["event"] != stage:
            continue
        d = e.get("direction") or dirs.get(e["setup"]); ts = pd.Timestamp(e["ts"]); day = e["ts"][:10]; m = minute_index(ts)
        for h in H:
            v = fwd(day, m, d, h)
            if v is not None:
                pts[h].append(v)
    r = {"stage": stage, "n": len(pts[15])}
    for h in H:
        a = np.array(pts[h]); r[f"{h}m mean"] = round(a.mean(), 2); r[f"{h}m t"] = round(a.mean() / (a.std(ddof=1) / np.sqrt(len(a))), 2) if len(a) > 2 else 0
    rows.append(r)
print(pd.DataFrame(rows).to_string(index=False))

t = pd.read_csv(f"{out_dir}/trades.csv")
print(f"\n=== 2. Trade paths after entry ({len(t)} trades), exits ignored, 60-minute window capped at 15:15 ===")
mfe, mae, hit = [], [], collections.defaultdict(int)
targets = (0.5, 1.0, 1.5, 2.0, 3.0)
for r in t.itertuples():
    s = sessions[r.date]; d = r.direction; e = r.entry; R = r.risk_pts
    seg = s.candles[r.entry_minute: min(len(s.candles), r.entry_minute + 60, 361)]
    best = max((c.high if d > 0 else -c.low) * 1 for c in seg) ; best = (best - e) if d > 0 else (e + best)
    worst = min((c.low if d > 0 else -c.high) for c in seg); worst = (worst - e) if d > 0 else (e + worst)
    mfe.append(best / R); mae.append(worst / R)
    # first touch: k x R target vs -1R stop (adverse-first inside a candle)
    for k in targets:
        done = False
        for c in seg:
            adverse = (c.low - e) * d; fav = (c.high - e) * d if d > 0 else (e - c.low)
            if adverse <= -R:
                break
            if fav >= k * R:
                hit[k] += 1; done = True; break
mfe, mae = np.array(mfe), np.array(mae)
print("MFE (R): median %.2f  mean %.2f  share>=1R %.0f%%  share>=2R %.0f%%" % (np.median(mfe), mfe.mean(), 100 * (mfe >= 1).mean(), 100 * (mfe >= 2).mean()))
print("MAE (R): median %.2f  share hitting -1R within 60m %.0f%%" % (np.median(mae), 100 * (mae <= -1).mean()))
print("P(target before -1R stop):", {k: f"{100 * hit[k] / len(t):.0f}%" for k in targets}, " breakeven need for k:", {k: f"{100 / (1 + k):.0f}%" for k in targets})
print("(with a random walk P(kR before 1R) = 1/(1+k); anything above that is edge)")

print("\n=== 3. Realised R by exit reason vs what the same trades reached (MFE) ===")
t["mfe"] = mfe
print(t.groupby("exit_reason").agg(n=("net", "size"), realised_R=("r_mult", "mean"), mfe_R=("mfe", "mean")).round(2).to_string())

print("\n=== 4. Same entries, direction flipped: P(kR before -1R) ===")
hit2 = collections.defaultdict(int)
for r in t.itertuples():
    s = sessions[r.date]; d = -r.direction; e = r.entry; R = r.risk_pts
    seg = s.candles[r.entry_minute: min(len(s.candles), r.entry_minute + 60, 361)]
    for k in targets:
        for c in seg:
            adverse = (c.low - e) if d > 0 else (e - c.high); fav = (c.high - e) if d > 0 else (e - c.low)
            if adverse <= -R:
                break
            if fav >= k * R:
                hit2[k] += 1; break
print({k: f"{100 * hit2[k] / len(t):.0f}%" for k in targets})
