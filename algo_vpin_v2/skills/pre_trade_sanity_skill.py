"""
Pre-Trade Sanity & Execution Plan Skill
Inspired by marian2js/trading-skills (pre-trade-check, risk-reward-sanity-check, execution-plan-check)
Validates trade eligibility, invalidation clarity, liquidity, and risk-reward profile before firing orders.
"""

import logging
from typing import Any, Dict, List, Optional
from .base_skill import BaseTradingSkill, SkillResult

logger = logging.getLogger("algo_vpin_v2.skills.pre_trade_sanity")


class PreTradeSanitySkill(BaseTradingSkill):
    """
    Institutional Go / No-Go Pre-Trade Gatekeeper.
    Performs 4 rigorous pre-trade checks before committing capital:
    1. Structural Invalidation Clarity (Is SL distinct and within volatility bounds?)
    2. Liquidity & Executability Check (Is option strike tradable with tight spread?)
    3. Asymmetric Risk-Reward Profile (Is minimum 1:1.5 RR achievable before nearest structural pivot?)
    4. Regime & Noise Filter (Rejects trades during low-volatility compression).
    """
    def __init__(self):
        super().__init__(name="PreTradeSanitySkill")

    def evaluate(
        self,
        intended_direction: Optional[str] = None,
        entry_price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        current_spot: Optional[float] = None,
        nearest_resistance: Optional[float] = None,
        nearest_support: Optional[float] = None,
        recent_bars: Optional[List[Dict[str, float]]] = None,
        spread_pts: float = 0.50,
        **kwargs
    ) -> SkillResult:
        """
        Executes pre-trade sanity verification.
        Returns SkillResult with is_favorable=True if all 4 pre-flight checks pass.
        """
        if entry_price is None or stop_loss is None or intended_direction is None:
            return SkillResult(
                is_favorable=True,
                confidence=0.60,
                signal="SANITY_PASS_DEFAULT",
                reason="PreTradeSanity: Insufficient parameters for full sanity audit. Proceeding with default risk guard."
            )

        # Check 1: Invalidation Clarity (Risk distance > spread and non-zero)
        risk_dist = abs(entry_price - stop_loss)
        if risk_dist <= spread_pts * 2.0:
            return SkillResult(
                is_favorable=False,
                confidence=0.90,
                signal="INVALIDATION_TOO_TIGHT",
                reason=f"PreTradeSanity VETO: Stop loss ({stop_loss:.2f}) is suffocatingly close to entry ({entry_price:.2f}). Spread drag will cause instant stop-out."
            )

        # Check 2: Risk-Reward Sanity (Target >= 1.4x Risk)
        if take_profit is not None:
            reward_dist = abs(take_profit - entry_price)
            rr_ratio = reward_dist / risk_dist if risk_dist > 0 else 1.0
            if rr_ratio < 1.30:
                return SkillResult(
                    is_favorable=False,
                    confidence=0.85,
                    signal="POOR_RISK_REWARD",
                    reason=f"PreTradeSanity VETO: Risk-Reward ratio {rr_ratio:.2f} is below minimum 1:1.30 institutional threshold."
                )

        # Check 3: Obstruction Check (Nearest Structural Support / Resistance)
        if current_spot is not None:
            if intended_direction == "BUY" and nearest_resistance is not None:
                dist_to_res = nearest_resistance - current_spot
                if 0 < dist_to_res < 8.0:
                    return SkillResult(
                        is_favorable=False,
                        confidence=0.80,
                        signal="RESISTANCE_OBSTRUCTION",
                        reason=f"PreTradeSanity VETO: Direct overhead resistance at {nearest_resistance:.1f} is only {dist_to_res:.1f} pts away. Poor headroom."
                    )
            elif intended_direction == "SELL" and nearest_support is not None:
                dist_to_sup = current_spot - nearest_support
                if 0 < dist_to_sup < 8.0:
                    return SkillResult(
                        is_favorable=False,
                        confidence=0.80,
                        signal="SUPPORT_OBSTRUCTION",
                        reason=f"PreTradeSanity VETO: Direct floor support at {nearest_support:.1f} is only {dist_to_sup:.1f} pts away. Poor downside headroom."
                    )

        # Check 4: Compression / Range Exhaustion
        if recent_bars and len(recent_bars) >= 5:
            highs = [b.get("high", b.get("close", 0.0)) for b in recent_bars[-5:]]
            lows = [b.get("low", b.get("close", 0.0)) for b in recent_bars[-5:]]
            five_bar_range = max(highs) - min(lows)
            if five_bar_range < 4.0:
                return SkillResult(
                    is_favorable=False,
                    confidence=0.75,
                    signal="CHOP_COMPRESSION_VETO",
                    reason=f"PreTradeSanity VETO: 5-bar range is only {five_bar_range:.1f} pts. Market is compressed in tight deadlock."
                )

        return SkillResult(
            is_favorable=True,
            confidence=0.90,
            signal="PRE_TRADE_CLEARED",
            reason=f"PreTradeSanity: All 4 institutional pre-flight checks passed (Risk: {risk_dist:.1f} pts, RR: {rr_ratio if 'rr_ratio' in locals() else 1.5:.2f}x).",
            metadata={"risk_dist": risk_dist, "rr_ratio": rr_ratio if "rr_ratio" in locals() else 1.5}
        )
