"""
Position Guardian Skill (Chinmay Option Scalping Methodology)
Manages active trade lifecycle on every tick:
- Spot SL Shield: Protects against option premium wicks while Spot structure holds
- Breakeven Lock: Moves SL to (Entry + 1.5 pts) as soon as 1.0R profit is achieved
- Scalper Trailing: Dynamic 0.5R profit locking above 1.5R gain
- Automatic Take Profit Trigger
- Disaster Stop Loss: Emergency kill-switch at -25% to -30%
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from .base_skill import BaseTradingSkill, SkillResult


@dataclass
class GuardianAction:
    """Action recommendation from the Position Guardian Skill"""
    should_exit: bool
    exit_reason: str
    updated_sl: float
    is_shielded: bool
    pnl_pts: float
    pnl_pct: float
    trailing_tier: str


class PositionGuardianSkill(BaseTradingSkill):
    """
    Autonomous position supervisor applying Chinmay's scalper defense and profit-locking rules.
    """
    def __init__(self, name: str = "PositionGuardianSkill"):
        super().__init__(name=name)

    def evaluate(
        self,
        symbol: str,
        is_call: bool,
        entry_price: float,
        current_option_price: float,
        current_spot_price: float,
        spot_sl: float,
        current_sl: float,
        target_tp: float,
        initial_risk_pts: float,
        highest_price_seen: float,
        current_time_hour: int = 12,
        current_time_minute: int = 0
    ) -> SkillResult:
        """
        Evaluates position state on current tick and returns recommended action.
        """
        # 1. Update peak price & gain
        peak = max(highest_price_seen, current_option_price)
        gain_pts = peak - entry_price
        pnl_pts = current_option_price - entry_price
        pnl_pct = (pnl_pts / entry_price * 100.0) if entry_price > 0 else 0.0

        updated_sl = current_sl
        trailing_tier = "Tier 0.0R"

        # 2. Scalper Breakeven & Trailing Ratchet
        # At 1.0R gain: Move SL to Breakeven (+1.5 pts)
        if gain_pts >= 1.0 * initial_risk_pts:
            be_level = entry_price + 1.5
            if updated_sl < be_level:
                updated_sl = round(be_level, 2)
                trailing_tier = "Breakeven (+1.5 pts)"

        # At 1.5R gain or higher: Step-lock profits in 0.5R increments
        if gain_pts >= 1.5 * initial_risk_pts:
            step_units = int((gain_pts / initial_risk_pts - 1.0) / 0.5)
            locked_sl = entry_price + (step_units * 0.5 * initial_risk_pts)
            if locked_sl > updated_sl:
                updated_sl = round(locked_sl, 2)
                trailing_tier = f"Scalper Tier {1.0 + (step_units * 0.5):.1f}R"

        # 3. Check Exits
        should_exit = False
        exit_reason = ""
        is_shielded = False

        # Rule A: Take Profit Hit
        if current_option_price >= target_tp:
            should_exit = True
            exit_reason = f"TAKE_PROFIT_TRIGGERED (Opt INR {current_option_price:.2f} >= TP INR {target_tp:.2f})"

        # Rule B: Intraday Square-off (15:24 IST)
        elif current_time_hour == 15 and current_time_minute >= 24:
            should_exit = True
            exit_reason = "INTRADAY_SQUARE_OFF (15:24 IST)"

        # Rule C: Option price touched or dipped below current SL
        elif current_option_price <= updated_sl:
            is_trail_locked = (updated_sl >= entry_price)
            spot_broken = (current_spot_price <= spot_sl) if is_call else (current_spot_price >= spot_sl)
            disaster_hit = (current_option_price <= entry_price * 0.75)

            # If it's a trailing stop that is already locked in profit, execute immediately
            if is_trail_locked:
                should_exit = True
                exit_reason = f"TRAILING_STOP_LOCKED (Opt INR {current_option_price:.2f} <= SL INR {updated_sl:.2f} | {trailing_tier})"

            # If disaster stop is reached (-25%), execute emergency exit
            elif disaster_hit:
                should_exit = True
                exit_reason = f"HARD_DISASTER_SL_HIT (Opt INR {current_option_price:.2f} <= -25% Disaster Level)"

            # If Spot Index confirms structural breakdown, execute stop loss
            elif spot_broken:
                should_exit = True
                exit_reason = f"DUAL_STOP_LOSS (Opt INR {current_option_price:.2f} <= SL INR {updated_sl:.2f} & Spot Broken @ {current_spot_price:.2f})"

            # SPOT SL SHIELD: Option price dipped but Spot has NOT broken structure!
            else:
                is_shielded = True
                exit_reason = f"SPOT_SL_SHIELD_ACTIVE (Opt INR {current_option_price:.2f} <= SL INR {updated_sl:.2f}, but Spot {current_spot_price:.2f} holding structure)"

        action = GuardianAction(
            should_exit=should_exit,
            exit_reason=exit_reason,
            updated_sl=updated_sl,
            is_shielded=is_shielded,
            pnl_pts=round(pnl_pts, 2),
            pnl_pct=round(pnl_pct, 2),
            trailing_tier=trailing_tier
        )

        return SkillResult(
            is_favorable=not should_exit,
            confidence=0.90 if should_exit else 0.50,
            signal="EXIT" if should_exit else "HOLD",
            reason=exit_reason or f"Holding position ({trailing_tier})",
            metadata={
                "action": action,
                "peak_price": peak,
                "updated_sl": updated_sl
            }
        )
