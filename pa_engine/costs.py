"""
Execution economics. Strategy decisions are made on the underlying; P&L is reported for the traded instrument:
  option  : premium = premium_pct x spot at entry; moves by delta x spot change; decays theta_per_min; spread per side
  futures : underlying points x lot size; STT 0.02% of sell notional; slippage in points
  points  : raw underlying points x lot size, no costs (informational edge only)
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TradeEconomics:
    gross: float
    costs: float
    net: float
    entry_fill: float
    exit_fill: float


def statutory(kind: str, buy_value: float, sell_value: float, brokerage: float, stt_sell_pct: float) -> float:
    if kind == "option":
        txn = 0.00035 * (buy_value + sell_value); stt = stt_sell_pct * sell_value
    elif kind == "futures":
        txn = 0.0000173 * (buy_value + sell_value); stt = 0.0002 * sell_value
    else:
        return 0.0
    sebi = 0.000001 * (buy_value + sell_value); stamp = 0.00003 * buy_value
    return 2 * brokerage + stt + txn + sebi + stamp + 0.18 * (2 * brokerage + txn + sebi)


def evaluate(cfg, direction: int, entry_spot: float, exit_spot: float, minutes_held: float, quantity: int) -> TradeEconomics:
    """Economics of one (partial) exit of `quantity` units."""
    kind = cfg.instrument
    pts = (exit_spot - entry_spot) * direction
    if kind == "points":
        g = pts * quantity
        return TradeEconomics(g, 0.0, g, entry_spot, exit_spot)
    if kind == "futures":
        fill_in = entry_spot + direction * cfg.slippage_points
        fill_out = exit_spot - direction * cfg.slippage_points
        g = (fill_out - fill_in) * direction * quantity
        c = statutory("futures", entry_spot * quantity, entry_spot * quantity, cfg.brokerage_per_order, 0.0002)
        return TradeEconomics(g, c, g - c, fill_in, fill_out)
    # option (always a long premium position)
    p0 = entry_spot * cfg.premium_pct
    fill_in = p0 * (1 + cfg.spread_pct)
    p_out = max(0.5, p0 + cfg.delta * pts - p0 * cfg.theta_per_min * minutes_held) * (1 - cfg.spread_pct)
    g = (p_out - fill_in) * quantity
    c = statutory("option", fill_in * quantity, p_out * quantity, cfg.brokerage_per_order, cfg.stt_sell_pct)
    return TradeEconomics(g, c, g - c, fill_in, p_out)


def risk_inr(cfg, stop_points: float, quantity: int) -> float:
    """Monetary loss at the stop for `quantity` units of the configured instrument."""
    if cfg.instrument == "option":
        return stop_points * cfg.delta * quantity
    return stop_points * quantity
