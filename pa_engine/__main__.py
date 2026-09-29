"""
CLI:
  python -m pa_engine backtest  --symbol NIFTY --data data/nifty_3y_1min.csv.gz [--config cfg.json] [--set key=value ...] [--out research/v3/nifty]
  python -m pa_engine matrix    --symbol NIFTY --data ... --out research/v3/matrix_nifty   (research matrix, section 32)
  python -m pa_engine paper     --symbol NIFTY                                         (live paper runner, Kite feed)
  python -m pa_engine login                                                             (daily Kite token)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .config import EngineConfig


def _cfg(args) -> EngineConfig:
    cfg = EngineConfig.load(args.config) if getattr(args, "config", None) else EngineConfig()
    cfg = cfg.for_symbol(args.symbol)
    if getattr(args, "set", None):
        cfg = cfg.override(**{k: v for k, v in (kv.split("=", 1) for kv in args.set)})
    return cfg


def cmd_backtest(args):
    import pandas as pd
    from .backtest import DEFAULT_SPLITS, metrics, run, split_metrics
    from .candles import load_sessions
    cfg = _cfg(args)
    t0 = time.time(); sessions = load_sessions(args.data)
    if args.start:
        sessions = [s for s in sessions if str(s.day) >= args.start]
    if args.end:
        sessions = [s for s in sessions if str(s.day) <= args.end]
    if args.limit:
        sessions = sessions[:args.limit]
    print(f"{len(sessions)} sessions loaded in {time.time() - t0:.1f}s ({sessions[0].day} .. {sessions[-1].day})", flush=True)
    out = Path(args.out) if args.out else None
    if out:
        out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    trades = run(cfg, sessions, log_path=str(out / "events.jsonl") if out and args.events else None, verbose=True)
    print(f"replay done in {time.time() - t0:.1f}s: {len(trades)} trades", flush=True)
    m = metrics(trades, len(sessions)); sm = split_metrics(trades, sessions, DEFAULT_SPLITS)
    print(json.dumps({"all": m, "splits": sm}, indent=1, default=str))
    if out:
        trades.to_csv(out / "trades.csv", index=False)
        cfg.save(out / "config.json")
        (out / "metrics.json").write_text(json.dumps({"all": m, "splits": sm}, indent=1, default=str))
        print(f"written to {out}")


def cmd_matrix(args):
    from .backtest import experiment
    from .candles import load_sessions
    cfg = _cfg(args)
    sessions = load_sessions(args.data)
    if args.limit:
        sessions = sessions[:args.limit]
    overrides = {
        "baseline": {},
        "no_retest": {"retest.enabled": False},
        "no_shift_required": {"structure.require_shift": False},
        "sweep_pen_0.05": {"sweep.min_penetration_atr": 0.05},
        "sweep_pen_0.20": {"sweep.min_penetration_atr": 0.20},
        "sweep_wick": {"sweep.require_wick": True},
        "bos_band_0.05": {"structure.bos_min_atr": 0.05},
        "bos_band_0.30": {"structure.bos_min_atr": 0.30},
        "threshold_6": {"score.entry_threshold": 6},
        "threshold_10": {"score.entry_threshold": 10},
        "stop_buffer_0.10": {"risk.buffer_atr": 0.10},
        "stop_buffer_0.50": {"risk.buffer_atr": 0.50},
        "min_rr_1.5": {"risk.min_rr": 1.5},
        "min_rr_3.0": {"risk.min_rr": 3.0},
        "no_partials": {"exit.tp1_fraction": 0.0},
        "trail_structure": {"exit.trail": "structure"},
        "trail_atr": {"exit.trail": "atr"},
        "trail_none_full_tp2": {"exit.trail": "none", "exit.tp1_fraction": 0.0, "exit.tp2_fraction": 1.0},
        "no_time_exit": {"exit.time_exit_bars": 0},
        "no_invalidation_exit": {"exit.invalidation_exit": False},
        "futures_costs": {"cost.instrument": "futures"},
        "raw_points": {"cost.instrument": "points"},
    }
    if args.only:
        overrides = {k: v for k, v in overrides.items() if k in args.only}
    df = experiment(cfg, sessions, overrides, out_dir=args.out)
    import pandas as pd
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    print(df.to_string(index=False))
    if args.out:
        df.to_csv(Path(args.out) / "matrix.csv", index=False)


def cmd_paper(args):
    from .live import main as live_main
    live_main(_cfg(args), paper=not args.live, poll=args.poll)


def cmd_login(args):
    from .kite_login import main as login_main
    login_main(args.request_token)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="pa_engine")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("backtest", "matrix"):
        p = sub.add_parser(name)
        p.add_argument("--symbol", default="NIFTY"); p.add_argument("--data", required=True); p.add_argument("--config")
        p.add_argument("--set", nargs="*"); p.add_argument("--out"); p.add_argument("--limit", type=int); p.add_argument("--start"); p.add_argument("--end")
        p.add_argument("--events", action="store_true", help="write events.jsonl (large)")
        p.add_argument("--only", nargs="*")
    p = sub.add_parser("paper"); p.add_argument("--symbol", default="NIFTY"); p.add_argument("--config"); p.add_argument("--set", nargs="*")
    p.add_argument("--live", action="store_true", help="place real orders (requires PAPER_TRADING=false in .env)"); p.add_argument("--poll", type=float, default=2.0)
    p = sub.add_parser("login"); p.add_argument("--request-token")
    args = ap.parse_args(argv)
    {"backtest": cmd_backtest, "matrix": cmd_matrix, "paper": cmd_paper, "login": cmd_login}[args.cmd](args)


if __name__ == "__main__":
    main()
