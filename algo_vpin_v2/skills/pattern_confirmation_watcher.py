"""
Pattern Confirmation Watcher & Post-Trade Cooldown Engine for Algo VPIN v2.0
Implements the Confirmation Watcher State Machine from architecture.md:
1. `FORMING` -> `CONFIRMATION_WATCHING` -> `CONFIRMED` -> `INVALIDATED`
2. Rejection Wick / Fakeout Filter (Prevents entering on false breakouts)
3. Post-Trade Cooldown Guard (Prevents re-entering the peak of an impulse after profit exit)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
from datetime import datetime
import logging
import pytz

from .base_skill import BaseTradingSkill, SkillResult
from ..garch_engine import DirectionalSignal

logger = logging.getLogger("algo_vpin_v2.skills.pattern_confirmation_watcher")


class PatternState(Enum):
    IDLE = "IDLE"
    FORMING = "FORMING"
    CONFIRMATION_WATCHING = "CONFIRMATION_WATCHING"
    CONFIRMED = "CONFIRMED"
    INVALIDATED = "INVALIDATED"
    COOLDOWN = "COOLDOWN"


@dataclass
class WatchlistCandidate:
    pattern_type: str
    bias: str                     # "BULLISH" or "BEARISH"
    trigger_level: float          # Breakout Neckline or Trendline price
    invalidation_level: float     # Pattern invalidation price (e.g. valley/peak)
    formed_time: datetime
    bars_elapsed: int = 0
    max_wait_bars: int = 4
    confidence: float = 0.70
    description: str = ""
    status: PatternState = PatternState.CONFIRMATION_WATCHING


class PatternConfirmationWatcher(BaseTradingSkill):
    """
    Patience Engine: Watches candidate patterns and verifies candle close beyond trigger levels.
    """
    def __init__(self):
        super().__init__(name="PatternConfirmationWatcher")
        self.tz = pytz.timezone("Asia/Kolkata")
        self.active_candidate: Optional[WatchlistCandidate] = None
        
        # Post-Trade Cooldown tracking
        self.cooldown_until_bar: int = 0
        self.cooldown_symbol: Optional[str] = None
        self.cooldown_side: Optional[str] = None
        self.current_bar_index: int = 0

    def evaluate(self, *args, **kwargs) -> SkillResult:
        """Evaluates active pattern candidate state"""
        if self.active_candidate is None:
            return SkillResult(is_favorable=True, confidence=0.50, signal="NO_ACTIVE_WATCHLIST", reason="No pattern currently being watched.")
        
        if getattr(self.active_candidate.status, "value", self.active_candidate.status) == PatternState.CONFIRMED.value:
            sig = "BUY_CALL" if self.active_candidate.bias == "BULLISH" else "BUY_PUT"
            return SkillResult(
                is_favorable=True,
                confidence=self.active_candidate.confidence,
                signal=sig,
                reason=f"CONFIRMED_BREAKOUT: {self.active_candidate.description} crossed {self.active_candidate.trigger_level:.2f}",
                metadata={"pattern_type": self.active_candidate.pattern_type, "trigger_level": self.active_candidate.trigger_level}
            )
        
        return SkillResult(
            is_favorable=False,
            confidence=0.50,
            signal="WATCHING",
            reason=f"Awaiting confirmation candle close for {self.active_candidate.pattern_type} @ {self.active_candidate.trigger_level:.2f}"
        )

    def set_post_trade_cooldown(self, symbol: str, side: str, bars: int = 3, current_bar: int = 0):
        """Activates cooldown to prevent buying at the peak of an impulse after taking profits"""
        self.cooldown_symbol = symbol
        self.cooldown_side = side
        self.current_bar_index = current_bar
        self.cooldown_until_bar = current_bar + bars
        logger.info(f"[COOLDOWN GUARD] Activated {bars}-bar re-entry lock on {symbol} ({side}) until bar {self.cooldown_until_bar}.")

    def is_in_cooldown(self, current_bar: int) -> bool:
        """Checks if current cycle is within post-trade cooldown"""
        self.current_bar_index = current_bar
        return current_bar < self.cooldown_until_bar

    def register_forming_pattern(
        self,
        pattern_type: str,
        bias: str,
        trigger_level: float,
        invalidation_level: float,
        confidence: float = 0.75,
        description: str = ""
    ) -> WatchlistCandidate:
        """Registers a newly detected pattern into the confirmation watchlist"""
        self.active_candidate = WatchlistCandidate(
            pattern_type=pattern_type,
            bias=bias,
            trigger_level=trigger_level,
            invalidation_level=invalidation_level,
            formed_time=datetime.now(self.tz),
            bars_elapsed=0,
            confidence=confidence,
            description=description,
            status=PatternState.CONFIRMATION_WATCHING
        )
        logger.info(f"[CONFIRMATION WATCHER] Registered candidate: {pattern_type} ({bias}) -> Watching Neckline/Trigger @ {trigger_level:.2f}")
        return self.active_candidate

    def on_candle_close(
        self,
        open_p: float,
        high_p: float,
        low_p: float,
        close_p: float,
        volume: float,
        current_bar: int
    ) -> Tuple[PatternState, str]:
        """
        Evaluates candle close against the active watchlist candidate.
        Returns (PatternState, reason)
        """
        self.current_bar_index = current_bar
        if self.active_candidate is None:
            return PatternState.IDLE, "No active candidate"

        self.active_candidate.bars_elapsed += 1
        cand = self.active_candidate

        # 1. Timeout Check
        if cand.bars_elapsed > cand.max_wait_bars:
            logger.info(f"[CONFIRMATION WATCHER] Pattern {cand.pattern_type} TIMEOUT ({cand.bars_elapsed} bars elapsed). Candidate dropped.")
            self.active_candidate = None
            return PatternState.INVALIDATED, "Confirmation timeout"

        # 2. Invalidation Level Check
        if cand.bias == "BULLISH" and close_p < cand.invalidation_level:
            logger.info(f"[CONFIRMATION WATCHER] Bullish pattern INVALIDATED: Close {close_p:.2f} < Invalidation {cand.invalidation_level:.2f}")
            self.active_candidate = None
            return PatternState.INVALIDATED, "Invalidation level breached"
        elif cand.bias == "BEARISH" and close_p > cand.invalidation_level:
            logger.info(f"[CONFIRMATION WATCHER] Bearish pattern INVALIDATED: Close {close_p:.2f} > Invalidation {cand.invalidation_level:.2f}")
            self.active_candidate = None
            return PatternState.INVALIDATED, "Invalidation level breached"

        # 3. Confirmation Trigger & Rejection Wick Check
        body_size = abs(close_p - open_p)
        total_candle = max(0.01, high_p - low_p)

        if cand.bias == "BULLISH":
            if close_p >= cand.trigger_level:
                # Check for upper rejection wick (fakeout test)
                upper_wick = high_p - max(open_p, close_p)
                if upper_wick > 2.0 * body_size and (upper_wick / total_candle) > 0.50:
                    logger.warning(f"[FAKEOUT VETO] Bullish trigger rejected with long upper wick ({upper_wick:.1f} pts). Invalidating.")
                    self.active_candidate = None
                    return PatternState.INVALIDATED, "Upper rejection wick (Fakeout)"
                
                # Confirmed!
                cand.status = PatternState.CONFIRMED
                logger.info(f"[CONFIRMATION SUCCESS] {cand.pattern_type} confirmed with strong candle close {close_p:.2f} >= {cand.trigger_level:.2f}!")
                return PatternState.CONFIRMED, f"Confirmed close above {cand.trigger_level:.2f}"

        elif cand.bias == "BEARISH":
            if close_p <= cand.trigger_level:
                # Check for lower rejection wick (fakeout test)
                lower_wick = min(open_p, close_p) - low_p
                if lower_wick > 2.0 * body_size and (lower_wick / total_candle) > 0.50:
                    logger.warning(f"[FAKEOUT VETO] Bearish trigger rejected with long lower wick ({lower_wick:.1f} pts). Invalidating.")
                    self.active_candidate = None
                    return PatternState.INVALIDATED, "Lower rejection wick (Fakeout)"

                # Confirmed!
                cand.status = PatternState.CONFIRMED
                logger.info(f"[CONFIRMATION SUCCESS] {cand.pattern_type} confirmed with strong candle close {close_p:.2f} <= {cand.trigger_level:.2f}!")
                return PatternState.CONFIRMED, f"Confirmed close below {cand.trigger_level:.2f}"

        return PatternState.CONFIRMATION_WATCHING, f"Still watching (Bar {cand.bars_elapsed}/{cand.max_wait_bars})"

    def reset_candidate(self):
        """Clears active candidate once trade is taken or dropped"""
        self.active_candidate = None
