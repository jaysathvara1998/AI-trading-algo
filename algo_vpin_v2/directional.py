"""
Directional trade mechanics (no scalping):
  * confirmation on a completed 1-minute candle close beyond the previous candle's high/low with a real body
  * stop-loss from structure: lowest low (CALL) / highest high (PUT) of the last N candles incl. the signal candle
  * take-profit from standard deviation: k x sigma(1-min log returns, lookback) x sqrt(horizon minutes) x spot
Pure functions; used by both the live engine (risk_manager / main) and the backtest harness.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

Bar = Dict[str, float]


def aggregate(bars: Sequence[Bar], tf: int) -> List[Bar]:
    """Group 1-min bars into `tf`-minute candles (last group may be partial). Bars are assumed consecutive."""
    bars = list(bars)
    if tf <= 1 or not bars:
        return bars
    out: List[Bar] = []
    # align so that the LAST candle ends on the last bar (the just-completed minute)
    start = len(bars) % tf
    for i in range(start, len(bars), tf):
        g = bars[i:i + tf]
        out.append({"open": g[0]["open"], "high": max(b["high"] for b in g), "low": min(b["low"] for b in g), "close": g[-1]["close"], "volume": sum(b.get("volume", 0.0) for b in g)})
    return out


def is_candle_close(minute_index: int, tf: int) -> bool:
    """True when the 1-min bar at session minute `minute_index` (0 = 09:15) completes a `tf`-minute candle."""
    return tf <= 1 or (minute_index + 1) % tf == 0


def confirm_close(bars: Sequence[Bar], direction: int, body_frac: float = 0.5) -> Tuple[bool, str]:
    """bars[-1] is the just-completed 1-min candle. direction +1 = CALL, -1 = PUT."""
    if len(bars) < 2:
        return False, "insufficient candles"
    cur, prev = bars[-1], bars[-2]
    rng = cur["high"] - cur["low"]
    if rng <= 0:
        return False, "flat candle"
    body = cur["close"] - cur["open"]
    if direction > 0:
        if cur["close"] <= prev["high"]:
            return False, f"close {cur['close']:.2f} not above previous high {prev['high']:.2f}"
        if body < body_frac * rng:
            return False, f"bullish body {body:.2f} < {body_frac:.0%} of range {rng:.2f}"
        return True, f"1-min close {cur['close']:.2f} above previous high {prev['high']:.2f} (body {body / rng:.0%})"
    else:
        if cur["close"] >= prev["low"]:
            return False, f"close {cur['close']:.2f} not below previous low {prev['low']:.2f}"
        if -body < body_frac * rng:
            return False, f"bearish body {-body:.2f} < {body_frac:.0%} of range {rng:.2f}"
        return True, f"1-min close {cur['close']:.2f} below previous low {prev['low']:.2f} (body {-body / rng:.0%})"


def recent_atr(bars: Sequence[Bar], n: int = 30) -> float:
    rs = [b["high"] - b["low"] for b in list(bars)[-n:] if b["high"] >= b["low"]]
    if not rs:
        return 0.0
    rs = sorted(rs)
    return rs[len(rs) // 2]           # median range: robust to the odd spike


def structure_stop(bars: Sequence[Bar], direction: int, lookback: int = 3, buffer_atr: float = 0.1, min_buffer: float = 0.5) -> float:
    """Spot stop under the lowest low (CALL) / over the highest high (PUT) of the last `lookback` candles."""
    win = list(bars)[-lookback:]
    buf = max(min_buffer, buffer_atr * recent_atr(bars))
    if direction > 0:
        return min(b["low"] for b in win) - buf
    return max(b["high"] for b in win) + buf


def realized_sigma(closes: Sequence[float], window: int = 60) -> float:
    """Std-dev of 1-min log returns over the last `window` closes (population estimate)."""
    c = list(closes)[-(window + 1):]
    if len(c) < 12:
        return 0.0
    rets = [math.log(c[i] / c[i - 1]) for i in range(1, len(c)) if c[i - 1] > 0 and c[i] > 0]
    if len(rets) < 10:
        return 0.0
    m = sum(rets) / len(rets)
    return math.sqrt(sum((r - m) ** 2 for r in rets) / len(rets))


def sigma_target(entry_spot: float, direction: int, closes: Sequence[float], window: int = 60, horizon_min: int = 60, mult: float = 1.0) -> Tuple[float, float]:
    """Spot target = entry +/- mult x sigma_1min x sqrt(horizon) x spot. Returns (target, one_sigma_move_pts)."""
    s = realized_sigma(closes, window)
    move = s * math.sqrt(max(1, horizon_min)) * entry_spot
    return entry_spot + direction * mult * move, move


def option_levels(entry_premium: float, entry_spot: float, spot_stop: float, spot_target: float, direction: int, delta: float) -> Tuple[float, float, float, float]:
    """Translate spot stop/target into premium levels with the contract delta. Returns (opt_sl, opt_tp, sl_pts, tp_pts)."""
    d = max(0.2, min(1.0, delta))
    sl_pts = abs(entry_spot - spot_stop) * d
    tp_pts = abs(spot_target - entry_spot) * d
    return max(0.5, entry_premium - sl_pts), entry_premium + tp_pts, sl_pts, tp_pts
