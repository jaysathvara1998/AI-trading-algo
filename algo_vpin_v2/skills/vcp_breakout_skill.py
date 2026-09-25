"""
Volatility Contraction Pattern (VCP) & Drawdown Circuit Breaker Skill
Inspired by tradermonty/claude-trading-skills (vcp-screener, drawdown-circuit-breaker)
Implements Mark Minervini's VCP contraction detection & dynamic volatility compression breakout confirmation.
"""

import logging
from typing import Any, Dict, List, Optional
import numpy as np
from .base_skill import BaseTradingSkill, SkillResult

logger = logging.getLogger("algo_vpin_v2.skills.vcp_breakout")


class VCPBreakoutSkill(BaseTradingSkill):
    """
    Mark Minervini Volatility Contraction Pattern (VCP) Detector & Volatility Coil Evaluator.
    Core Logic:
    1. Identifies contracting swing waves: Depth(T1) > Depth(T2) > Depth(T3)
    2. Identifies the tight 'Pivot Point' where price action compresses before explosive breakout
    3. Blocks premature trades during wide/loose volatility swings
    4. Triggers High-Conviction Breakout Signal when price clears the pivot on expanding volume.
    """
    def __init__(self):
        super().__init__(name="VCPBreakoutSkill")

    def evaluate(
        self,
        recent_bars: Optional[List[Dict[str, float]]] = None,
        current_price: Optional[float] = None,
        intended_direction: Optional[str] = None,
        volume_ratio: float = 1.0,
        **kwargs
    ) -> SkillResult:
        if recent_bars is None or len(recent_bars) < 15 or current_price is None:
            return SkillResult(
                is_favorable=True,
                confidence=0.50,
                signal="VCP_INSUFFICIENT_DATA",
                reason="VCP: Insufficient bar history (minimum 15 bars required for wave contraction audit)."
            )

        highs = [b.get("high", b.get("close", 0.0)) for b in recent_bars]
        lows = [b.get("low", b.get("close", 0.0)) for b in recent_bars]
        closes = [b.get("close", 0.0) for b in recent_bars]

        # Partition last 15 bars into 3 consecutive 5-bar contraction windows
        w1_high, w1_low = max(highs[-15:-10]), min(lows[-15:-10])
        w2_high, w2_low = max(highs[-10:-5]), min(lows[-10:-5])
        w3_high, w3_low = max(highs[-5:]), min(lows[-5:])

        depth_t1 = abs(w1_high - w1_low)
        depth_t2 = abs(w2_high - w2_low)
        depth_t3 = abs(w3_high - w3_low)

        # Check for strict or progressive contraction: T1 > T2 and T2 >= T3
        is_contracting = (depth_t1 > depth_t2) and (depth_t2 > depth_t3 or depth_t3 < (depth_t1 * 0.50))
        contraction_ratio = depth_t3 / depth_t1 if depth_t1 > 0 else 1.0

        pivot_high = max(highs[-10:])
        pivot_low = min(lows[-10:])

        # Scenario 1: Bullish VCP Coil Breakout
        if intended_direction == "BUY":
            if is_contracting and current_price >= pivot_high and volume_ratio >= 1.20:
                return SkillResult(
                    is_favorable=True,
                    confidence=0.92,
                    signal="VCP_BULLISH_EXPANSION",
                    reason=f"VCP Breakout Confirmed: Volatility contracted {contraction_ratio*100:.0f}% (T1:{depth_t1:.1f} -> T2:{depth_t2:.1f} -> T3:{depth_t3:.1f}). Volume expansion {volume_ratio:.1f}x.",
                    metadata={"contraction_ratio": contraction_ratio, "pivot": pivot_high}
                )
            elif depth_t3 > (depth_t1 * 1.2):
                return SkillResult(
                    is_favorable=False,
                    confidence=0.80,
                    signal="VCP_VOLATILITY_EXPANDING_LOOSE",
                    reason=f"VCP Veto: Market is wide & loose (T3:{depth_t3:.1f} > T1:{depth_t1:.1f}). Unstable for breakout entry."
                )

        # Scenario 2: Bearish VCP Coil Breakdown
        elif intended_direction == "SELL":
            if is_contracting and current_price <= pivot_low and volume_ratio >= 1.20:
                return SkillResult(
                    is_favorable=True,
                    confidence=0.92,
                    signal="VCP_BEARISH_EXPANSION",
                    reason=f"VCP Breakdown Confirmed: Volatility contracted {contraction_ratio*100:.0f}% (T1:{depth_t1:.1f} -> T2:{depth_t2:.1f} -> T3:{depth_t3:.1f}). Downward volume expansion {volume_ratio:.1f}x.",
                    metadata={"contraction_ratio": contraction_ratio, "pivot": pivot_low}
                )
            elif depth_t3 > (depth_t1 * 1.2):
                return SkillResult(
                    is_favorable=False,
                    confidence=0.80,
                    signal="VCP_VOLATILITY_EXPANDING_LOOSE",
                    reason=f"VCP Veto: Market is wide & loose (T3:{depth_t3:.1f} > T1:{depth_t1:.1f}). Unstable for breakdown entry."
                )

        return SkillResult(
            is_favorable=True,
            confidence=0.70,
            signal="VCP_NEUTRAL",
            reason=f"VCP Audit: Moderate structure (Contraction Ratio: {contraction_ratio*100:.0f}%). Proceeding with standard neural consensus.",
            metadata={"contraction_ratio": contraction_ratio}
        )
