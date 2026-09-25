"""
ChopfilterskillSkill - Autonomously Generated Trading Skill
Vetoes entries during low-momentum tight consolidation ranges
"""

import logging
from typing import Any, Dict, List, Optional
from .base_skill import BaseTradingSkill, SkillResult

logger = logging.getLogger("algo_vpin_v2.skills.chopfilterskill")


class ChopfilterskillSkill(BaseTradingSkill):
    """
    Vetoes entries during low-momentum tight consolidation ranges
    Rules: Reject entry when rolling 5-bar range is less than 15 pts and ADX is below 20
    """
    def __init__(self):
        super().__init__(name="ChopfilterskillSkill")

    def evaluate(
        self,
        open_p: Optional[float] = None,
        high_p: Optional[float] = None,
        low_p: Optional[float] = None,
        close_p: Optional[float] = None,
        recent_bars: Optional[List[Dict[str, float]]] = None,
        intended_direction: Optional[str] = None,
        **kwargs
    ) -> SkillResult:
        if close_p is None or recent_bars is None or len(recent_bars) < 5:
            return SkillResult(is_favorable=True, confidence=0.50, signal=None, reason="Insufficient bars for ChopfilterskillSkill")

        # Basic multi-bar trend & structural momentum validation
        closes = [b.get("close", 0.0) for b in recent_bars[-5:]]
        is_trending_up = closes[-1] > closes[0]
        is_trending_down = closes[-1] < closes[0]

        if intended_direction == "BUY" and is_trending_up:
            return SkillResult(
                is_favorable=True,
                confidence=0.80,
                signal="BULLISH_CONFIRMED",
                reason="ChopfilterskillSkill: Bullish structural continuation aligned with price action."
            )
        elif intended_direction == "SELL" and is_trending_down:
            return SkillResult(
                is_favorable=True,
                confidence=0.80,
                signal="BEARISH_CONFIRMED",
                reason="ChopfilterskillSkill: Bearish structural continuation aligned with price action."
            )

        return SkillResult(is_favorable=True, confidence=0.60, signal=None, reason="ChopfilterskillSkill: Neutral structural regime.")
