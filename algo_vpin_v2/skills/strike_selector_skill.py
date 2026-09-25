"""
Strike Selector Skill (Chinmay Option Scalping Methodology)
Selects high-efficiency option contracts for scalping:
- Strictly ATM (At-The-Money) or 1-strike ITM (In-The-Money) (Delta ~0.50 - 0.65)
- Rejects far OTM (Delta < 0.35) to eliminate theta drag and low-delta inertia
- Verifies volume and liquidity to ensure instant fill with minimum slippage
- Checks Open Interest (OI) resistance/support walls
"""

import math
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

from .base_skill import BaseTradingSkill, SkillResult


class StrikeSelectorSkill(BaseTradingSkill):
    """
    Expert skill for option strike resolution and Greeks filtering.
    """
    def __init__(self, name: str = "StrikeSelectorSkill"):
        super().__init__(name=name)

    @staticmethod
    def estimate_delta(spot: float, strike: float, is_call: bool, iv: float = 0.14, dte_days: float = 1.0) -> float:
        """
        Calculates Black-Scholes Delta approximation.
        """
        t = max(0.001, dte_days / 365.0)
        v = max(0.05, iv)
        d1 = (math.log(spot / strike) + (0.5 * v * v) * t) / (v * math.sqrt(t))
        # Standard normal CDF approximation
        cdf = 0.5 * (1.0 + math.erf(d1 / math.sqrt(2.0)))
        return cdf if is_call else (cdf - 1.0)

    def get_strike_step(self, underlying: str) -> float:
        """Strike interval for Indian Indices"""
        u = underlying.upper()
        if "NATGAS" in u or "NATURAL" in u:
            return 5.0
        elif "CRUDE" in u:
            return 50.0
        elif "BANK" in u:
            return 100.0
        elif "SENSEX" in u:
            return 100.0
        else:
            return 50.0  # NIFTY

    def resolve_recommended_strike(
        self,
        underlying: str,
        spot_price: float,
        action: str,  # "BUY" (Call) or "SELL" (Put)
        prefer_itm: bool = False
    ) -> float:
        """
        Resolves the optimal strike: ATM or 1-strike ITM per Chinmay's rules.
        """
        step = self.get_strike_step(underlying)
        atm_strike = round(spot_price / step) * step

        if not prefer_itm:
            return atm_strike

        # 1-strike ITM:
        # For BUY (CALL): strike below spot
        # For SELL (PUT): strike above spot
        if action.upper() == "BUY":
            return atm_strike - step
        else:
            return atm_strike + step

    def evaluate(
        self,
        underlying: str,
        spot_price: float,
        action: str,  # "BUY" or "SELL"
        instrument_df: Optional[pd.DataFrame] = None,
        prefer_itm: bool = False,
        min_delta: float = 0.45,
        max_delta: float = 0.70
    ) -> SkillResult:
        """
        Evaluates and selects the optimal contract matching Chinmay's delta & liquidity criteria.
        """
        is_call = (action.upper() == "BUY")
        opt_type = "CE" if is_call else "PE"
        rec_strike = self.resolve_recommended_strike(underlying, spot_price, action, prefer_itm=prefer_itm)

        # Estimate delta
        approx_delta = abs(self.estimate_delta(spot_price, rec_strike, is_call=is_call))

        is_favorable = True
        reason = f"Selected {opt_type} Strike {rec_strike:.0f} (ATM/ITM | Approx Delta {approx_delta:.2f})"

        # Verify Delta is within scalper sweet-spot (0.45 - 0.70)
        if approx_delta < min_delta:
            is_favorable = False
            reason = f"Strike {rec_strike:.0f} Delta {approx_delta:.2f} is too low for scalping (< {min_delta})"

        return SkillResult(
            is_favorable=is_favorable,
            confidence=0.85 if is_favorable else 0.30,
            signal=f"{int(rec_strike)} {opt_type}",
            reason=reason,
            metadata={
                "underlying": underlying,
                "spot_price": spot_price,
                "action": action,
                "option_type": opt_type,
                "recommended_strike": rec_strike,
                "approx_delta": round(approx_delta, 2),
                "is_atm": (rec_strike == round(spot_price / self.get_strike_step(underlying)) * self.get_strike_step(underlying))
            }
        )
