"""
Deterministic historical replay (spec sections 28, 30, 31): signals on completed candles, fills at the next
candle's open, adverse-first intra-candle exits, configurable costs, chronological splits, metrics, and a
research-matrix runner that applies config overrides one experiment at a time.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .candles import Session, load_sessions
from .config import EngineConfig
from .costs import evaluate
from .engine import SessionEngine
from .logger import EventLog


EXPIRY_WEEKDAY = {"NIFTY": (lambda d: 3 if d < pd.Timestamp("2025-09-01").date() else 1),
                  "SENSEX": (lambda d: 4 if d < pd.Timestamp("2025-01-01").date() else (1 if d < pd.Timestamp("2025-09-01").date() else 3)),
                  "BANKNIFTY": (lambda d: 2 if d < pd.Timestamp("2025-09-01").date() else 1)}


def run_session(cfg: EngineConfig, s: Session, log: Optional[EventLog] = None) -> List[dict]:
    eng = SessionEngine(cfg, s, log)
    trades: List[dict] = []
    pending = None
    open_rec = None
    for m, c in enumerate(s.candles):
        # fill yesterday's intent at this candle's open
        if pending is not None:
            fill = c.open
            eng.open_position(pending, fill, m)
            open_rec = {"date": str(s.day), "setup": pending.setup.id, "direction": pending.plan.direction, "entry_minute": m,
                        "entry": fill, "stop0": pending.plan.stop, "tp1": pending.plan.tp1 + (fill - pending.plan.entry), "tp2": pending.plan.tp2 + (fill - pending.plan.entry),
                        "target": pending.plan.target_name, "rr": pending.plan.rr, "score": pending.score, "regime": pending.regime,
                        "quantity": pending.plan.quantity, "lots": pending.plan.lots, "risk_pts": abs(fill - pending.plan.stop), "confirm": pending.setup.confirm_kind,
                        "retest": pending.setup.retest_seen, "level": pending.setup.sweep.level.source, "fills": [], "gross": 0.0, "costs": 0.0, "net": 0.0}
            pending = None
        dec = eng.on_candle(c)
        for f in dec.fills:
            if open_rec is None:
                continue
            eco = evaluate(cfg.cost, open_rec["direction"], open_rec["entry"], f.price, f.minute - open_rec["entry_minute"], f.quantity)
            open_rec["fills"].append({"minute": f.minute, "price": f.price, "qty": f.quantity, "reason": f.reason, "net": round(eco.net, 2)})
            open_rec["gross"] += eco.gross; open_rec["costs"] += eco.costs; open_rec["net"] += eco.net
            if f.state == "EXIT":
                open_rec["exit_minute"] = f.minute; open_rec["exit"] = f.price; open_rec["exit_reason"] = f.reason
                open_rec["hold_min"] = f.minute - open_rec["entry_minute"]; open_rec["spot_pts"] = (f.price - open_rec["entry"]) * open_rec["direction"]
                open_rec["r_mult"] = open_rec["spot_pts"] / open_rec["risk_pts"] if open_rec["risk_pts"] else 0.0
                eng.record_result(open_rec["net"])
                trades.append(open_rec); open_rec = None
        if dec.intents and m + 1 < len(s.candles):
            pending = dec.intents[0]     # one position at a time
    # a position still open at the last candle is force-closed at the close (should not happen with flatten)
    if open_rec is not None and eng.position is not None and eng.position.open:
        c = s.candles[-1]; q = eng.position.remaining
        eco = evaluate(cfg.cost, open_rec["direction"], open_rec["entry"], c.close, len(s.candles) - 1 - open_rec["entry_minute"], q)
        open_rec.update({"exit_minute": len(s.candles) - 1, "exit": c.close, "exit_reason": "EOD_FORCE", "hold_min": len(s.candles) - 1 - open_rec["entry_minute"]})
        open_rec["gross"] += eco.gross; open_rec["costs"] += eco.costs; open_rec["net"] += eco.net
        open_rec["spot_pts"] = (c.close - open_rec["entry"]) * open_rec["direction"]; open_rec["r_mult"] = open_rec["spot_pts"] / open_rec["risk_pts"] if open_rec["risk_pts"] else 0.0
        trades.append(open_rec)
    return trades


def run(cfg: EngineConfig, sessions: Sequence[Session], log_path: Optional[str] = None, verbose: bool = False) -> pd.DataFrame:
    log = EventLog(log_path, keep=False) if log_path else None
    rows: List[dict] = []
    skip = EXPIRY_WEEKDAY.get(cfg.symbol.upper())
    for i, s in enumerate(sessions):
        if cfg.session.skip_expiry_day and skip is not None and pd.Timestamp(s.day).weekday() == skip(s.day):
            continue
        rows += run_session(cfg, s, log)
        if verbose and (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(sessions)} sessions, {len(rows)} trades", flush=True)
    if log:
        log.close()
    return pd.DataFrame(rows)


def metrics(t: pd.DataFrame, n_sessions: int) -> Dict[str, float]:
    if t.empty:
        return {"trades": 0, "sessions": n_sessions}
    w = t[t.net > 0]; L = t[t.net <= 0]; cum = t.net.cumsum(); dd = float((cum - cum.cummax()).min())
    gp, gl = float(w.net.sum()), float(-L.net.sum())
    daily = t.groupby("date").net.sum()
    return {"trades": int(len(t)), "sessions": n_sessions, "trades_per_day": round(len(t) / max(1, n_sessions), 2),
            "gross": round(float(t.gross.sum())), "costs": round(float(t.costs.sum())), "net": round(float(t.net.sum())),
            "win_rate": round(len(w) / len(t), 3), "avg_win": round(float(w.net.mean()) if len(w) else 0.0), "avg_loss": round(float(L.net.mean()) if len(L) else 0.0),
            "expectancy": round(float(t.net.mean()), 1), "profit_factor": round(gp / gl, 2) if gl > 0 else float("inf"),
            "max_drawdown": round(dd), "recovery_factor": round(float(t.net.sum()) / -dd, 2) if dd < 0 else float("inf"),
            "avg_r": round(float(t.r_mult.mean()), 3), "t_stat_net": round(float(t.net.mean() / (t.net.std(ddof=1) / np.sqrt(len(t)))), 2) if len(t) > 2 and t.net.std() > 0 else 0.0,
            "avg_hold_min": round(float(t.hold_min.mean()), 1), "daily_sharpe": round(float(daily.mean() / daily.std(ddof=1) * np.sqrt(252)), 2) if len(daily) > 2 and daily.std() > 0 else 0.0,
            "exit_reasons": t.exit_reason.value_counts().to_dict()}


def split_metrics(t: pd.DataFrame, sessions: Sequence[Session], splits: Dict[str, tuple]) -> Dict[str, Dict]:
    out = {}
    for name, (a, b) in splits.items():
        ss = [s for s in sessions if a <= str(s.day) <= b]
        tt = t[(t.date >= a) & (t.date <= b)] if not t.empty else t
        out[name] = metrics(tt, len(ss))
    return out


DEFAULT_SPLITS = {"dev_2023Q4_2024": ("2023-01-01", "2024-12-31"), "val_2025": ("2025-01-01", "2025-12-31"), "oos_2026": ("2026-01-01", "2026-12-31")}


def experiment(base: EngineConfig, sessions: Sequence[Session], overrides: Dict[str, dict], out_dir: Optional[str] = None) -> pd.DataFrame:
    """Research matrix: run each named override set and tabulate per-split metrics."""
    rows = []
    for name, ov in overrides.items():
        cfg = base.override(**ov) if ov else base
        t = run(cfg, sessions)
        sm = split_metrics(t, sessions, DEFAULT_SPLITS)
        row = {"experiment": name, "trades": int(len(t)), "net": round(float(t.net.sum())) if len(t) else 0}
        for k, v in sm.items():
            row[f"{k}:net"] = v.get("net", 0); row[f"{k}:n"] = v.get("trades", 0); row[f"{k}:pf"] = v.get("profit_factor", 0); row[f"{k}:avgR"] = v.get("avg_r", 0)
        rows.append(row)
        if out_dir:
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            t.to_csv(Path(out_dir) / f"{name}_trades.csv", index=False)
            cfg.save(Path(out_dir) / f"{name}_config.json")
    return pd.DataFrame(rows)
