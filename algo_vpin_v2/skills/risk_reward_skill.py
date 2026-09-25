"""
Risk-Reward & Sizing Skill (Chinmay Option Scalping Methodology)
Enforces strict mathematical trade expectancy:
- Minimum 1:1.5 to 1:2 Risk-to-Reward Ratio
- Anchors Stop Loss to Spot Index swing high/low structure
- Maps Spot risk to Option premium risk using Delta
- Rejects unfavorable trades where the nearest structural barrier restricts profit potential
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .base_skill import BaseTradingSkill, SkillResult


@dataclass
class RiskRewardPlan:
    """Detailed structural execution plan calculated by the RiskRewardSkill"""
    is_favorable: bool
    risk_reward_ratio: float
    spot_entry: float
    spot_sl: float
    spot_tp: float
    spot_risk_pts: float
    spot_reward_pts: float
    option_entry: float
    option_sl: float
    option_tp: float
    option_risk_pts: float
    option_reward_pts: float
    reason: str


class RiskRewardSkill(BaseTradingSkill):
    """
    Expert skill for calculating trade expectancy, dynamic stop loss, and target profit.
    """
    def __init__(
        self,
        name: str = "RiskRewardSkill",
        min_rr_ratio: float = 1.50,
        default_delta: float = 0.55
    ):
        super().__init__(name=name)
        self.min_rr_ratio = min_rr_ratio
        self.default_delta = default_delta

    def calculate_spot_levels(
        self,
        spot_entry: float,
        action: str,  # "BUY" or "SELL"
        candle_high: Optional[float] = None,
        candle_low: Optional[float] = None,
        nearest_resistance: Optional[float] = None,
        nearest_support: Optional[float] = None,
        asset_sl_budget_pts: float = 18.0  # default 18 pts NIFTY, 50 pts SENSEX
    ) -> Tuple[float, float]:
        """
        Calculates Spot SL and Spot TP based on candle structure or key S/R levels.
        """
        is_buy = (action.upper() == "BUY")

        if is_buy:
            # Spot SL: below candle low or fixed structural buffer
            if candle_low is not None and (spot_entry - candle_low) <= asset_sl_budget_pts * 1.5:
                spot_sl = candle_low - 2.0
            else:
                spot_sl = spot_entry - asset_sl_budget_pts

            spot_risk = spot_entry - spot_sl

            # Spot TP: nearest resistance or 1.5x risk
            if nearest_resistance is not None and nearest_resistance > spot_entry:
                spot_tp = nearest_resistance
            else:
                spot_tp = spot_entry + (spot_risk * self.min_rr_ratio)

        else:  # SELL / PUT
            if candle_high is not None and (candle_high - spot_entry) <= asset_sl_budget_pts * 1.5:
                spot_sl = candle_high + 2.0
            else:
                spot_sl = spot_entry + asset_sl_budget_pts

            spot_risk = spot_sl - spot_entry

            # Spot TP: nearest support or 1.5x risk
            if nearest_support is not None and nearest_support < spot_entry:
                spot_tp = nearest_support
            else:
                spot_tp = spot_entry - (spot_risk * self.min_rr_ratio)

        return round(spot_sl, 2), round(spot_tp, 2)

    def evaluate(
        self,
        spot_entry: float,
        action: str,  # "BUY" or "SELL"
        option_entry: float,
        candle_high: Optional[float] = None,
        candle_low: Optional[float] = None,
        nearest_resistance: Optional[float] = None,
        nearest_support: Optional[float] = None,
        delta: Optional[float] = None,
        asset_sl_budget_pts: float = 18.0
    ) -> SkillResult:
        """
        Generates full Risk-to-Reward plan and checks if trade satisfies minimum expectancy.
        """
        eff_delta = delta or self.default_delta
        spot_sl, spot_tp = self.calculate_spot_levels(
            spot_entry=spot_entry,
            action=action,
            candle_high=candle_high,
            candle_low=candle_low,
            nearest_resistance=nearest_resistance,
            nearest_support=nearest_support,
            asset_sl_budget_pts=asset_sl_budget_pts
        )

        spot_risk = abs(spot_entry - spot_sl)
        spot_reward = abs(spot_tp - spot_entry)

        rr_ratio = (spot_reward / spot_risk) if spot_risk > 0 else 0.0

        # Translate Spot Risk to Option Premium Points
        # Option SL is Spot Risk * Delta, bounded between 6.0 and 12.0 pts (or 10-15% of premium)
        opt_sl_pts = max(6.0, min(14.0, spot_risk * eff_delta))
        opt_tp_pts = opt_sl_pts * max(self.min_rr_ratio, rr_ratio)

        opt_sl = round(option_entry - opt_sl_pts, 2)
        opt_tp = round(option_entry + opt_tp_pts, 2)

        is_favorable = (rr_ratio >= (self.min_rr_ratio - 0.05))

        if is_favorable:
            reason = f"Favorable R:R 1:{rr_ratio:.2f} (Target +{opt_tp_pts:.1f} pts, Risk -{opt_sl_pts:.1f} pts)"
        else:
            reason = f"Unfavorable R:R 1:{rr_ratio:.2f} (< minimum required 1:{self.min_rr_ratio:.2f})"

        plan = RiskRewardPlan(
            is_favorable=is_favorable,
            risk_reward_ratio=round(rr_ratio, 2),
            spot_entry=spot_entry,
            spot_sl=spot_sl,
            spot_tp=spot_tp,
            spot_risk_pts=round(spot_risk, 2),
            spot_reward_pts=round(spot_reward, 2),
            option_entry=option_entry,
            option_sl=opt_sl,
            option_tp=opt_tp,
            option_risk_pts=round(opt_sl_pts, 2),
            option_reward_pts=round(opt_tp_pts, 2),
            reason=reason
        )

        return SkillResult(
            is_favorable=is_favorable,
            confidence=0.85 if is_favorable else 0.25,
            signal=action,
            reason=reason,
            metadata={"plan": plan}
        )
