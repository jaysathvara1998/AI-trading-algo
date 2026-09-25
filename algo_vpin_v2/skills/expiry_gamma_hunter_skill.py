"""
Expiry 0-DTE Gamma Hunter Skill for Algo VPIN v2.0 (AURA-v2)
Specialized for 0-DTE Expiry Day Gamma Expansion:
- Schedule: NIFTY (Tuesday / Thursday), SENSEX (Thursday / Friday)
- Execution Window: 13:15 IST – 15:00 IST (Peak Gamma Acceleration)
- Integrates Dhan / Tradehull Option Chain for dynamic Delta (0.35 - 0.50) strike selection
- Enforces Exponential Multi-Tier Step-Trailing Shield (+30%, +80%, +150%, +250%+)
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import pytz
import logging

from .base_skill import BaseTradingSkill, SkillResult
from ..config import CONFIG

logger = logging.getLogger("algo_vpin_v2.skills.expiry_gamma_hunter")


class ExpiryGammaHunterSkill(BaseTradingSkill):
    """
    Identifies 0-DTE explosive gamma breakouts on expiry afternoons
    and manages exponential multi-tier profit trailing.
    """
    def __init__(self):
        super().__init__(name="ExpiryGammaHunterSkill")
        self.tz = pytz.timezone("Asia/Kolkata")

    def is_expiry_day(self, symbol: str, current_time: Optional[datetime] = None) -> bool:
        """
        Validates if today is an active Expiry day for the underlying index.
        SENSEX: Thursday (weekday 3)
        NIFTY: Tuesday (weekday 1) / Thursday (weekday 3)
        """
        now = current_time or datetime.now(self.tz)
        weekday = now.weekday()  # Monday=0, Tuesday=1, Wednesday=2, Thursday=3, Friday=4

        sym = symbol.upper()
        if "SENSEX" in sym:
            return weekday in (3, 4)  # Thursday (or Friday BSE shift)
        elif "NIFTY" in sym:
            return weekday in (1, 3)  # Tuesday (FINNIFTY/NIFTY) or Thursday
        return False

    def is_gamma_window(self, current_time: Optional[datetime] = None) -> bool:
        """
        Gamma expansion peak window: 13:15 IST to 15:00 IST.
        """
        now = current_time or datetime.now(self.tz)
        t = now.time()
        start_t = datetime.strptime("13:15", "%H:%M").time()
        end_t = datetime.strptime("15:00", "%H:%M").time()
        return start_t <= t <= end_t

    def evaluate(
        self,
        symbol: str,
        current_spot: float,
        directional_bias: str,  # "BULLISH" / "BEARISH"
        vpin_toxicity: float,
        volume_ratio: float,
        five_min_htf_aligned: bool,
        day_high: float,
        day_low: float,
        current_time: Optional[datetime] = None,
        *args, **kwargs
    ) -> SkillResult:
        """
        Evaluates 0-DTE Gamma Breakout setup.
        """
        now = current_time or datetime.now(self.tz)

        # 1. Expiry Day & Time Window Gate
        if not self.is_expiry_day(symbol, now):
            return SkillResult(
                is_favorable=False,
                confidence=0.0,
                signal="NOT_EXPIRY_DAY",
                reason=f"Today ({now.strftime('%A')}) is not 0-DTE expiry for {symbol}."
            )

        if not self.is_gamma_window(now):
            return SkillResult(
                is_favorable=False,
                confidence=0.0,
                signal="OUTSIDE_GAMMA_WINDOW",
                reason=f"Current time {now.strftime('%H:%M')} is outside 13:15-15:00 IST Gamma window."
            )

        # 2. HTF & Orderflow Confirmation
        if not five_min_htf_aligned:
            return SkillResult(
                is_favorable=False,
                confidence=0.30,
                signal="HTF_MISALIGNED",
                reason="5-Minute HTF candle does not support gamma explosion."
            )

        # 3. Institutional Toxicity / Volume Surge Check
        has_volume_thrust = (volume_ratio >= 1.5 or vpin_toxicity >= 0.22)

        # 4. Structural Day High/Low Expansion Check
        is_breakdown = (directional_bias == "BEARISH" and current_spot <= day_low + (day_high - day_low) * 0.15)
        is_breakout = (directional_bias == "BULLISH" and current_spot >= day_high - (day_high - day_low) * 0.15)

        if not (is_breakdown or is_breakout):
            return SkillResult(
                is_favorable=False,
                confidence=0.45,
                signal="NO_STRUCTURAL_EXPANSION",
                reason="Spot is inside middle consolidation range; waiting for extreme channel breakout."
            )

        confidence = 0.85 if (has_volume_thrust and (is_breakdown or is_breakout)) else 0.70
        signal_type = "GAMMA_PUT_SURGE" if directional_bias == "BEARISH" else "GAMMA_CALL_SURGE"
        
        return SkillResult(
            is_favorable=True,
            confidence=confidence,
            signal=signal_type,
            reason=f"0-DTE Gamma Surge Active: {directional_bias} expansion with VolRatio={volume_ratio:.2f}, VPIN={vpin_toxicity:.3f}",
            metadata={
                "target_delta_range": (0.35, 0.50),
                "strategy_type": "GAMMA_HUNTER_SCALP",
                "recommended_trailing_tier": "EXPONENTIAL_GAMMA_STEP",
                "timestamp": now.isoformat()
            }
        )

    def calculate_gamma_step_trailing(
        self,
        entry_premium: float,
        highest_premium: float,
        current_premium: float
    ) -> Tuple[float, str]:
        """
        Exponential Multi-Tier Step-Trailing Shield for Gamma Runners:
        - +30% gain -> SL to Cost (Breakeven Shield)
        - +80% gain -> SL locked at +40%
        - +150% gain -> SL locked at +100% (2x Double)
        - >= +250% gain -> Continuous 20% trailing behind peak price
        """
        if entry_premium <= 0:
            return 0.0, "INITIAL_SL"

        peak_gain_pct = ((highest_premium - entry_premium) / entry_premium) * 100.0

        if peak_gain_pct >= 250.0:
            trail_sl = round(highest_premium * 0.80, 2)
            return trail_sl, f"GAMMA_SUPER_RUNNER (+{peak_gain_pct:.0f}% Peak -> 20% Peak Trail @ INR {trail_sl:.2f})"
        elif peak_gain_pct >= 150.0:
            trail_sl = round(entry_premium * 2.00, 2)
            return trail_sl, f"GAMMA_TIER_3_LOCK (+150% Peak -> Locked +100% @ INR {trail_sl:.2f})"
        elif peak_gain_pct >= 80.0:
            trail_sl = round(entry_premium * 1.40, 2)
            return trail_sl, f"GAMMA_TIER_2_LOCK (+80% Peak -> Locked +40% @ INR {trail_sl:.2f})"
        elif peak_gain_pct >= 30.0:
            trail_sl = round(entry_premium * 1.02, 2)
            return trail_sl, f"GAMMA_BREAKEVEN_SHIELD (+30% Peak -> Locked Cost @ INR {trail_sl:.2f})"

        return round(entry_premium * 0.70, 2), "INITIAL_GAMMA_SL"
