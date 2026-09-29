"""
Reference / liquidity levels available at a given minute of the session, with freshness (touch count).
Kinds: "H" = a high (buy-side liquidity above), "L" = a low (sell-side liquidity below).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from .candles import Candle, Session
from .swings import Swing


@dataclass
class Level:
    name: str
    price: float
    kind: str               # "H" | "L"
    source: str             # PDH, PDL, SESSION_HIGH, SESSION_LOW, OR_HIGH, OR_LOW, EQUAL_HIGHS, EQUAL_LOWS, SWING_H, SWING_L
    touches: int = 0

    @property
    def key(self) -> str:
        return f"{self.source}:{round(self.price, 2)}"


class LevelTracker:
    """Rebuilds the level set from the session state each setup-tf candle and keeps touch counts across the day."""

    def __init__(self, cfg, session: Session):
        self.cfg = cfg
        self.session = session
        self.touches: Dict[str, int] = {}

    def build(self, c1: Sequence[Candle], upto_min: int, swings_setup: Sequence[Swing], setup_candles: Sequence[Candle],
              atr_setup: float) -> List[Level]:
        s = self.session; lv: List[Level] = []
        if s.prev_high is not None:
            lv += [Level("PDH", s.prev_high, "H", "PDH"), Level("PDL", s.prev_low, "L", "PDL")]
        orm = self.cfg.opening_range_min
        if upto_min >= orm - 1:
            lv += [Level("OR_HIGH", max(x.high for x in c1[:orm]), "H", "OR_HIGH"), Level("OR_LOW", min(x.low for x in c1[:orm]), "L", "OR_LOW")]
        # session extremes formed BEFORE the current setup candle (exclude the last 3 minutes to avoid self-reference)
        if upto_min >= 20:
            past = c1[:max(1, upto_min - 3)]
            lv += [Level("SESSION_HIGH", max(x.high for x in past), "H", "SESSION_HIGH"), Level("SESSION_LOW", min(x.low for x in past), "L", "SESSION_LOW")]
        hs = [x for x in swings_setup if x.kind == "H"][-self.cfg.recent_swings:]
        ls = [x for x in swings_setup if x.kind == "L"][-self.cfg.recent_swings:]
        for x in hs:
            lv.append(Level(f"SWING_H@{x.idx}", x.price, "H", "SWING_H"))
        for x in ls:
            lv.append(Level(f"SWING_L@{x.idx}", x.price, "L", "SWING_L"))
        tol = self.cfg.equal_tol_atr * atr_setup
        for a, b in zip(hs, hs[1:]):
            if abs(a.price - b.price) <= tol:
                lv.append(Level("EQUAL_HIGHS", max(a.price, b.price), "H", "EQUAL_HIGHS"))
        for a, b in zip(ls, ls[1:]):
            if abs(a.price - b.price) <= tol:
                lv.append(Level("EQUAL_LOWS", min(a.price, b.price), "L", "EQUAL_LOWS"))
        # de-duplicate by (kind, rounded price)
        seen = {}
        for l in lv:
            k = (l.kind, round(l.price, 1))
            if k not in seen:
                l.touches = self.touches.get(l.key, 0)
                seen[k] = l
        return list(seen.values())

    def record_touch(self, level: Level) -> None:
        self.touches[level.key] = self.touches.get(level.key, 0) + 1

    def is_fresh(self, level: Level) -> bool:
        return level.touches < self.cfg.max_touches
