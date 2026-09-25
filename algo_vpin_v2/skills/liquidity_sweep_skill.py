"""
Liquidity Sweep & Stop Hunt Reversal Skill
Inspired by tradermonty/claude-trading-skills (liquidity-sweeps) and Smart Money Concepts (SMC)
Detects fakeouts above key highs/lows (PDH, PDL, Session Highs/Lows) followed by strong rejection wicks.
"""

import logging
from typing import Any, Dict, List, Optional
from .base_skill import BaseTradingSkill, SkillResult

logger = logging.getLogger("algo_vpin_v2.skills.liquidity_sweep")


class LiquiditySweepSkill(BaseTradingSkill):
    """
    Evaluates institutional liquidity grabs and stop-hunting behavior.
    Core Logic:
    1. Identifies when price pierces a key level (e.g. Previous Day High / Low, Swing High / Low)
    2. Measures if the candle closed back inside the range (long rejection wick / pin bar)
    3. Confirms high-probability mean-reversion reversal entries after retail stops are cleared.
    """
    def __init__(self):
        super().__init__(name="LiquiditySweepSkill")

    def evaluate(
        self,
        recent_bars: Optional[List[Dict[str, float]]] = None,
        current_price: Optional[float] = None,
        intended_direction: Optional[str] = None,
        key_high: Optional[float] = None,
        key_low: Optional[float] = None,
        **kwargs
    ) -> SkillResult:
        if recent_bars is None or len(recent_bars) < 3 or current_price is None:
            return SkillResult(
                is_favorable=True,
                confidence=0.50,
                signal="SWEEP_INSUFFICIENT_DATA",
                reason="Liquidity Sweep: Insufficient candle history for rejection audit."
            )

        last_bar = recent_bars[-1]
        o = last_bar.get("open", current_price)
        h = last_bar.get("high", current_price)
        l = last_bar.get("low", current_price)
        c = last_bar.get("close", current_price)
        bar_range = max(h - l, 0.01)
        body = abs(c - o)
        upper_wick = h - max(o, c)
        lower_wick = min(o, c) - l

        # If key_high / key_low not provided, infer from previous 10 bars
        if key_high is None:
            key_high = max(b.get("high", 0.0) for b in recent_bars[:-1])
        if key_low is None:
            key_low = min(b.get("low", 999999.0) for b in recent_bars[:-1])

        # Scenario 1: Bearish Liquidity Sweep (Swept High, Rejected Down)
        # High pierced key_high, but Close ended back below key_high with upper wick > 40% of candle
        if h >= key_high and c < key_high and (upper_wick / bar_range) >= 0.38:
            if intended_direction == "SELL":
                return SkillResult(
                    is_favorable=True,
                    confidence=0.92,
                    signal="BEARISH_LIQUIDITY_SWEEP",
                    reason=f"Bearish Liquidity Grab: Swept High {h:.1f} > Key Level {key_high:.1f} and rejected with {upper_wick/bar_range*100:.0f}% upper wick. High-probability short.",
                    metadata={"swept_level": key_high, "rejection_wick_pct": upper_wick / bar_range}
                )
            elif intended_direction == "BUY":
                return SkillResult(
                    is_favorable=False,
                    confidence=0.85,
                    signal="BUY_BLOCKED_BULL_TRAP",
                    reason=f"Veto BUY: Bull Trap detected. Price swept {key_high:.1f} and got rejected."
                )

        # Scenario 2: Bullish Liquidity Sweep (Swept Low, Rejected Up)
        # Low pierced key_low, but Close ended back above key_low with lower wick > 40% of candle
        if l <= key_low and c > key_low and (lower_wick / bar_range) >= 0.38:
            if intended_direction == "BUY":
                return SkillResult(
                    is_favorable=True,
                    confidence=0.92,
                    signal="BULLISH_LIQUIDITY_SWEEP",
                    reason=f"Bullish Liquidity Grab: Swept Low {l:.1f} < Key Level {key_low:.1f} and rebounded with {lower_wick/bar_range*100:.0f}% lower wick. High-probability long.",
                    metadata={"swept_level": key_low, "rejection_wick_pct": lower_wick / bar_range}
                )
            elif intended_direction == "SELL":
                return SkillResult(
                    is_favorable=False,
                    confidence=0.85,
                    signal="SELL_BLOCKED_BEAR_TRAP",
                    reason=f"Veto SELL: Bear Trap detected. Price swept {key_low:.1f} and rebounded strongly."
                )

        return SkillResult(
            is_favorable=True,
            confidence=0.70,
            signal="SWEEP_CLEAN",
            reason=f"Liquidity Sweep Check: Order flow clean, no active trap detected around [{key_low:.1f}, {key_high:.1f}]."
        )
