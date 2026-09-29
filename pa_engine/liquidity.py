"""
Liquidity sweep detection on the setup timeframe. A sweep is a candidate event, not a trade:
price trades beyond a meaningful level by at least min_penetration (ATR units), does not accept beyond it
(closes back inside within reclaim_bars), and the penetration is not so large that it is a breakout.
Every numeric feature is recorded so the definition can be tightened or loosened by config.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

from .candles import Candle
from .levels import Level


@dataclass(frozen=True)
class SweepEvent:
    idx: int                 # setup-tf candle index at which the sweep was confirmed (close back inside)
    start_idx: int           # candle that penetrated the level
    level: Level
    direction: int           # implied trade direction: swept high -> -1 (bearish), swept low -> +1 (bullish)
    penetration_atr: float
    close_location: float    # (close - level) / atr, sign in the trade direction (positive = reclaimed)
    wick_ratio: float        # rejection wick / candle range on the penetrating candle
    bars_beyond: int
    sweep_id: str


def detect_sweeps(c: Sequence[Candle], levels: Sequence[Level], atr_value: float, idx: int, cfg,
                  already: set) -> List[SweepEvent]:
    """Evaluate at setup-tf candle `idx` (just completed). Returns sweeps confirmed on this candle."""
    out: List[SweepEvent] = []
    if idx < 1 or atr_value <= 0:
        return out
    cur = c[idx]
    for lv in levels:
        d = -1 if lv.kind == "H" else +1
        # find the penetrating candle within the reclaim window ending at idx
        for k in range(idx, max(-1, idx - cfg.reclaim_bars), -1):
            pen_c = c[k]
            pen = (pen_c.high - lv.price) if d < 0 else (lv.price - pen_c.low)
            pen_atr = pen / atr_value
            if pen_atr < cfg.min_penetration_atr:
                continue
            if pen_atr > cfg.max_penetration_atr:
                break
            # reclaimed: current candle closes back inside; all candles k..idx-1 must not have closed beyond+accepted
            inside = (cur.close < lv.price) if d < 0 else (cur.close > lv.price)
            if not inside:
                break
            key = f"{lv.key}@{k}"
            if key in already:
                break
            rng = pen_c.range or 1e-9
            wick = (pen_c.high - max(pen_c.open, pen_c.close)) if d < 0 else (min(pen_c.open, pen_c.close) - pen_c.low)
            wick_ratio = max(0.0, wick / rng)
            if cfg.require_wick and wick_ratio < 0.4:
                break
            close_loc = ((lv.price - cur.close) if d < 0 else (cur.close - lv.price)) / atr_value
            out.append(SweepEvent(idx=idx, start_idx=k, level=lv, direction=d, penetration_atr=pen_atr, close_location=close_loc,
                                  wick_ratio=wick_ratio, bars_beyond=idx - k + 1, sweep_id=key))
            already.add(key)
            break
    return out
