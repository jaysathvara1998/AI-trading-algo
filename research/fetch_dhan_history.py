"""
Fetch multi-year 1-minute index bars from Dhan's intraday history endpoint (90-day windows) and save
them as algo_vpin_v2/data/{symbol}_{tag}_1min.csv.gz in the same schema as the bundled 12m files.

Auth: uses DHAN_CLIENT_ID from .env and the newest Dependencies/token_*.txt access token that the
live bot already generated (or DHAN_ACCESS_TOKEN from the environment). No login is performed.

    .venv/bin/python research/fetch_dhan_history.py --years 3 --symbols NIFTY SENSEX --tag 3y
"""
import os, sys, glob, time, argparse, datetime as dt
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
SEC = {"NIFTY": "13", "SENSEX": "51", "BANKNIFTY": "25", "FINNIFTY": "27"}
IST = "Asia/Kolkata"

def token():
    if os.getenv("DHAN_ACCESS_TOKEN_OVERRIDE"): return os.environ["DHAN_ACCESS_TOKEN_OVERRIDE"]
    files = sorted(glob.glob(str(ROOT / "Dependencies" / "token_*.txt")))
    if files:
        parts = open(files[-1]).read().strip().split("|")
        if len(parts) >= 3 and dt.datetime.fromisoformat(parts[2]) > dt.datetime.now():
            return parts[1]
    sys.exit("No valid Dhan access token found (Dependencies/token_*.txt expired or missing).")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=float, default=3.0); ap.add_argument("--symbols", nargs="+", default=["NIFTY", "SENSEX"])
    ap.add_argument("--tag", default="3y"); ap.add_argument("--end", default=str(dt.date.today()))
    a = ap.parse_args()
    env = dict(l.strip().split("=", 1) for l in open(ROOT / ".env") if "=" in l and not l.startswith("#"))
    from dhanhq import DhanContext, dhanhq
    d = dhanhq(DhanContext(env["DHAN_CLIENT_ID"], token()))
    end = dt.date.fromisoformat(a.end); start = end - dt.timedelta(days=int(365.25 * a.years))
    for sym in a.symbols:
        frames = []; frm = start
        while frm < end:
            to = min(frm + dt.timedelta(days=89), end)
            for attempt in range(3):
                r = d.intraday_minute_data(security_id=SEC[sym], exchange_segment="IDX_I", instrument_type="INDEX", from_date=str(frm), to_date=str(to))
                if isinstance(r, dict) and r.get("status") == "success": break
                print(f"  retry {attempt+1} {sym} {frm}->{to}: {str(r)[:160]}", file=sys.stderr); time.sleep(2.0)
            data = r.get("data") or {}
            n = len(data.get("timestamp", []))
            if n:
                frames.append(pd.DataFrame(data))
            print(f"{sym} {frm} -> {to}: {n} bars", file=sys.stderr, flush=True)
            frm = to + dt.timedelta(days=1); time.sleep(0.7)
        df = pd.concat(frames, ignore_index=True)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True).dt.tz_convert(IST)
        df = df.drop_duplicates("timestamp").sort_values("timestamp")
        t = df["timestamp"].dt.time
        df = df[(t >= dt.time(9, 15)) & (t < dt.time(15, 30))]
        df = df[["open", "high", "low", "close", "volume", "timestamp"]]
        out = ROOT / "algo_vpin_v2" / "data" / f"{sym.lower()}_{a.tag}_1min.csv.gz"
        df.to_csv(out, index=False, compression="gzip")
        per_day = df.groupby(df["timestamp"].dt.date).size()
        print(f"{sym}: {len(df)} session bars, {per_day.size} sessions, {df.timestamp.min()} -> {df.timestamp.max()}, sessions with <370 bars: {int((per_day < 370).sum())} -> {out}")

if __name__ == "__main__":
    main()
