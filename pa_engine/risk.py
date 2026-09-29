"""
Structural stop, structural targets, reward-to-risk validation and position sizing (spec sections 14-17, 22).
All prices are in the underlying; monetary conversion goes through costs.risk_inr.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

from .costs import risk_inr
from .levels import Level


@dataclass
class RiskPlan:
    direction: int
    entry: float
    stop: float
    risk_pts: float
    tp1: float
    tp2: float
    target_name: str
    rr: float
    quantity: int
    lots: int
    risk_inr: float
    stop_reason: str


def structural_stop(direction: int, invalidation: float, atr_value: float, cfg) -> float:
    buf = cfg.buffer_atr * atr_value
    return invalidation - buf if direction > 0 else invalidation + buf


def nearest_target(direction: int, entry: float, levels: Sequence[Level], atr_value: float, min_target_atr: float) -> Optional[Level]:
    """Nearest opposing level in the trade direction, ignoring levels closer than min_target_atr."""
    cands = []
    for lv in levels:
        if direction > 0 and lv.kind == "H" and lv.price - entry >= min_target_atr * atr_value:
            cands.append((lv.price - entry, lv))
        if direction < 0 and lv.kind == "L" and entry - lv.price >= min_target_atr * atr_value:
            cands.append((entry - lv.price, lv))
    if not cands:
        return None
    return min(cands, key=lambda x: x[0])[1]


def plan(direction: int, entry: float, invalidation: float, levels: Sequence[Level], atr_value: float, risk_cfg, cost_cfg,
         exit_cfg, stop_reason: str) -> tuple[Optional[RiskPlan], str]:
    """Returns (plan, reason). plan is None when the setup fails a risk rule; reason explains which."""
    stop = structural_stop(direction, invalidation, atr_value, risk_cfg)
    risk_pts = abs(entry - stop)
    if risk_pts < risk_cfg.min_stop_atr * atr_value:
        stop = entry - direction * risk_cfg.min_stop_atr * atr_value
        risk_pts = abs(entry - stop)
        stop_reason += " (widened to min stop)"
    if risk_pts > risk_cfg.max_stop_atr * atr_value:
        return None, f"stop {risk_pts:.1f} pts exceeds max {risk_cfg.max_stop_atr} x ATR ({risk_cfg.max_stop_atr * atr_value:.1f})"
    tgt = nearest_target(direction, entry, levels, atr_value, risk_cfg.min_target_atr)
    tp1 = entry + direction * exit_cfg.tp1_r * risk_pts
    if tgt is None:
        # no structural target on the map: fall back to tp2_r only if the config allows a fixed-R target
        tp2 = entry + direction * exit_cfg.tp2_r * risk_pts; tname = f"{exit_cfg.tp2_r:.1f}R (no structural target)"
        rr = exit_cfg.tp2_r
    else:
        tp2 = tgt.price; tname = tgt.name; rr = abs(tp2 - entry) / risk_pts
    if rr < risk_cfg.min_rr:
        return None, f"R:R {rr:.2f} to {tname} below minimum {risk_cfg.min_rr}"
    # sizing
    lot = risk_cfg.lot_size
    per_lot = risk_inr(cost_cfg, risk_pts, lot)
    if risk_cfg.sizing == "risk":
        lots = int(risk_cfg.max_risk_inr // per_lot) if per_lot > 0 else 0
        lots = max(0, min(lots, risk_cfg.max_lots))
        if lots <= 0:
            return None, f"risk per lot Rs {per_lot:.0f} exceeds cap Rs {risk_cfg.max_risk_inr:.0f}"
    else:
        lots = risk_cfg.base_lots
        if per_lot * lots > risk_cfg.max_risk_inr:
            return None, f"risk Rs {per_lot * lots:.0f} exceeds cap Rs {risk_cfg.max_risk_inr:.0f} at {lots} lot(s)"
    return RiskPlan(direction, entry, stop, risk_pts, tp1, tp2, tname, rr, lots * lot, lots, per_lot * lots, stop_reason), "ok"
