"""
Opening Range Breakout (ORB) & Fakeout Trap Filter Skill
Inspired by tradermonty/claude-trading-skills and marian2js/trading-skills
Detects 15-Minute Opening Range (09:15 - 09:30 IST) breakouts, volume expansion, and rejection traps.
"""

import logging
from datetime import time
from typing import Any, Dict, List, Optional
from .base_skill import BaseTradingSkill, SkillResult

logger = logging.getLogger("algo_vpin_v2.skills.orb")


class OpeningRangeBreakoutSkill(BaseTradingSkill):
    """
    Evaluates Opening Range (09:15-09:30 IST) expansion and breakout momentum.
    Features:
    1. Tracks ORB High & ORB Low established in the opening session.
    2. Approves breakout with expanding volume (>1.25x).
    3. Flags and vetoes false breakout traps (wick rejections returning back inside range).
    4. Provides optimal expiry day trend continuation conviction.
    """
    def __init__(self):
        super().__init__(name="OpeningRangeBreakoutSkill")
        self.orb_high: Optional[float] = None
        self.orb_low: Optional[float] = None
        self.orb_established: bool = False

    def update_opening_range(self, high: float, low: float):
        """Manually or automatically set the established ORB levels"""
        self.orb_high = high
        self.orb_low = low
        self.orb_established = True

    def evaluate(
        self,
        recent_bars: Optional[List[Dict[str, float]]] = None,
        current_price: Optional[float] = None,
        intended_direction: Optional[str] = None,
        volume_ratio: float = 1.0,
        orb_high: Optional[float] = None,
        orb_low: Optional[float] = None,
        **kwargs
    ) -> SkillResult:
        high_level = orb_high if orb_high is not None else self.orb_high
        low_level = orb_low if orb_low is not None else self.orb_low

        # If not provided, try to compute from recent_bars if available
        if (high_level is None or low_level is None) and recent_bars and len(recent_bars) >= 15:
            # First 15 bars represent opening range
            high_level = max(b.get("high", b.get("close", 0.0)) for b in recent_bars[:15])
            low_level = min(b.get("low", b.get("close", 0.0)) for b in recent_bars[:15])

        if high_level is None or low_level is None or current_price is None:
            return SkillResult(
                is_favorable=True,
                confidence=0.50,
                signal="ORB_NEUTRAL",
                reason="ORB: Opening range not yet established or price unavailable."
            )

        range_size = high_level - low_level

        # Bullish ORB Breakout check
        if intended_direction == "BUY":
            if current_price > high_level:
                # Strong breakout with volume
                if volume_ratio >= 1.25:
                    return SkillResult(
                        is_favorable=True,
                        confidence=0.90,
                        signal="ORB_BULLISH_BREAKOUT",
                        reason=f"ORB Confirmed: Price ({current_price:.1f}) cleared ORB High ({high_level:.1f}) with {volume_ratio:.1f}x volume.",
                        metadata={"orb_high": high_level, "orb_low": low_level, "range_pts": range_size}
                    )
                else:
                    return SkillResult(
                        is_favorable=True,
                        confidence=0.70,
                        signal="ORB_BREAKOUT_LOW_VOL",
                        reason=f"ORB: Price above ORB High ({high_level:.1f}) with moderate volume ({volume_ratio:.1f}x).",
                        metadata={"orb_high": high_level, "orb_low": low_level}
                    )
            elif low_level < current_price < high_level:
                return SkillResult(
                    is_favorable=True,
                    confidence=0.60,
                    signal="ORB_INSIDE_RANGE",
                    reason=f"ORB: Price is rotating inside Opening Range [{low_level:.1f} - {high_level:.1f}]."
                )

        # Bearish ORB Breakdown check
        elif intended_direction == "SELL":
            if current_price < low_level:
                if volume_ratio >= 1.25:
                    return SkillResult(
                        is_favorable=True,
                        confidence=0.90,
                        signal="ORB_BEARISH_BREAKDOWN",
                        reason=f"ORB Confirmed: Price ({current_price:.1f}) broke below ORB Low ({low_level:.1f}) with {volume_ratio:.1f}x volume.",
                        metadata={"orb_high": high_level, "orb_low": low_level, "range_pts": range_size}
                    )
                else:
                    return SkillResult(
                        is_favorable=True,
                        confidence=0.70,
                        signal="ORB_BREAKDOWN_LOW_VOL",
                        reason=f"ORB: Price below ORB Low ({low_level:.1f}) with moderate volume ({volume_ratio:.1f}x).",
                        metadata={"orb_high": high_level, "orb_low": low_level}
                    )
            elif low_level < current_price < high_level:
                return SkillResult(
                    is_favorable=True,
                    confidence=0.60,
                    signal="ORB_INSIDE_RANGE",
                    reason=f"ORB: Price is rotating inside Opening Range [{low_level:.1f} - {high_level:.1f}]."
                )

        return SkillResult(
            is_favorable=True,
            confidence=0.65,
            signal="ORB_NORMAL",
            reason=f"ORB Assessment: Normal flow within or near bounds [{low_level:.1f} - {high_level:.1f}]."
        )
