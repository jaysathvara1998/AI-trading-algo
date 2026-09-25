"""
Price Action Skill (Chinmay Option Scalping Methodology)
Detects high-probability candlestick patterns & institutional structural triggers:
- Pin Bar / Hammer (Support Rejection)
- Shooting Star (Resistance Rejection)
- Bullish & Bearish Engulfing
- Support/Resistance Flip & Retest
- Range / Consolidation Breakout
"""

from typing import Dict, List, Optional, Tuple
import numpy as np

from .base_skill import BaseTradingSkill, SkillResult


class PriceActionSkill(BaseTradingSkill):
    """
    Expert skill for candlestick geometry and institutional price structure.
    """
    def __init__(self, name: str = "PriceActionSkill"):
        super().__init__(name=name)

    def detect_pin_bar(
        self,
        open_p: float,
        high_p: float,
        low_p: float,
        close_p: float
    ) -> Optional[str]:
        """
        Detects Hammer (Bullish Rejection) or Shooting Star (Bearish Rejection).
        Requires rejection wick to be at least 2.0x the body size.
        """
        candle_range = high_p - low_p
        if candle_range <= 0.0:
            return None

        body = abs(close_p - open_p)
        upper_wick = high_p - max(open_p, close_p)
        lower_wick = min(open_p, close_p) - low_p

        # Prevent division by zero for Doji
        eff_body = max(body, candle_range * 0.05)

        # Bullish Hammer: long lower shadow >= 2x body, upper wick <= 1x body
        if lower_wick >= 2.0 * eff_body and upper_wick <= 1.0 * eff_body:
            return "HAMMER_BULLISH"

        # Bearish Shooting Star: long upper shadow >= 2x body, lower wick <= 1x body
        if upper_wick >= 2.0 * eff_body and lower_wick <= 1.0 * eff_body:
            return "SHOOTING_STAR_BEARISH"

        return None

    def detect_engulfing(
        self,
        prev_o: float,
        prev_c: float,
        curr_o: float,
        curr_c: float
    ) -> Optional[str]:
        """
        Detects Bullish or Bearish Engulfing candles.
        """
        prev_is_red = prev_c < prev_o
        prev_is_green = prev_c > prev_o
        curr_is_green = curr_c > curr_o
        curr_is_red = curr_c < curr_o

        if prev_is_red and curr_is_green:
            if curr_o <= (prev_c + 1.0) and curr_c >= (prev_o - 0.5):
                return "BULLISH_ENGULFING"

        if prev_is_green and curr_is_red:
            if curr_o >= (prev_c - 1.0) and curr_c <= (prev_o + 0.5):
                return "BEARISH_ENGULFING"

        return None

    def detect_breakout(
        self,
        recent_highs: List[float],
        recent_lows: List[float],
        curr_close: float,
        curr_high: float,
        curr_low: float
    ) -> Optional[str]:
        """
        Detects breakout from consolidation range (past 5-10 bars).
        """
        if len(recent_highs) < 5 or len(recent_lows) < 5:
            return None

        range_high = max(recent_highs[:-1])
        range_low = min(recent_lows[:-1])
        consolidation_range = range_high - range_low

        if consolidation_range <= 0.0:
            return None

        # Breakout to upside
        if curr_close > range_high:
            return "BREAKOUT_UP"

        # Breakout to downside
        if curr_close < range_low:
            return "BREAKOUT_DOWN"

        return None

    def detect_sr_retest(
        self,
        spot_price: float,
        key_levels: List[float],
        tolerance_pts: float = 6.0
    ) -> Tuple[bool, Optional[float]]:
        """
        Checks if the spot price is retesting a key institutional support or resistance level.
        """
        for level in key_levels:
            if abs(spot_price - level) <= tolerance_pts:
                return True, level
        return False, None

    def evaluate(
        self,
        open_p: float,
        high_p: float,
        low_p: float,
        close_p: float,
        prev_open: Optional[float] = None,
        prev_close: Optional[float] = None,
        recent_highs: Optional[List[float]] = None,
        recent_lows: Optional[List[float]] = None,
        key_levels: Optional[List[float]] = None,
        intended_direction: Optional[str] = None,  # "BUY" or "SELL"
        chart_pattern: Optional[Any] = None
    ) -> SkillResult:
        """
        Consolidates price action cues and scores agreement with intended direction.
        """
        patterns_found = []
        conf_boost = 0.0

        # 1. Pin bar check
        pin = self.detect_pin_bar(open_p, high_p, low_p, close_p)
        if pin:
            patterns_found.append(pin)

        # 2. Engulfing check
        if prev_open is not None and prev_close is not None:
            eng = self.detect_engulfing(prev_open, prev_close, open_p, close_p)
            if eng:
                patterns_found.append(eng)

        # 3. Consolidation breakout check
        if recent_highs and recent_lows:
            bo = self.detect_breakout(recent_highs, recent_lows, close_p, high_p, low_p)
            if bo:
                patterns_found.append(bo)

        # 4. S/R retest check
        at_sr, sr_lvl = False, None
        if key_levels:
            at_sr, sr_lvl = self.detect_sr_retest(close_p, key_levels)
            if at_sr:
                patterns_found.append(f"SR_RETEST_{sr_lvl:.1f}")

        # 5. Classical Chart Pattern check (Double Bottom W, Double Top M, etc.)
        if chart_pattern is not None and hasattr(chart_pattern, "pattern_type"):
            p_val = getattr(chart_pattern.pattern_type, "value", str(chart_pattern.pattern_type))
            if p_val and p_val != "NONE":
                patterns_found.append(p_val)

        # Check alignment with intended direction
        bullish_signals = ["HAMMER_BULLISH", "BULLISH_ENGULFING", "BREAKOUT_UP", "DOUBLE_BOTTOM_W", "ASCENDING_TRIANGLE", "BULL_FLAG", "TRENDLINE_BREAKOUT_UP"]
        bearish_signals = ["SHOOTING_STAR_BEARISH", "BEARISH_ENGULFING", "BREAKOUT_DOWN", "DOUBLE_TOP_M", "DESCENDING_TRIANGLE", "BEAR_FLAG", "TRENDLINE_BREAKDOWN_DOWN"]

        is_favorable = True
        direction_confirmed = None

        has_bull = any(p in bullish_signals for p in patterns_found)
        has_bear = any(p in bearish_signals for p in patterns_found)

        if intended_direction == "BUY":
            if has_bear and not has_bull:
                is_favorable = False
                reason = f"Bearish price action contradicts BUY setup: {patterns_found}"
            elif has_bull:
                direction_confirmed = "BUY"
                conf_boost = 0.25
                reason = f"Bullish confirmation: {patterns_found}"
            else:
                reason = "Neutral price action (no adverse pattern)"
        elif intended_direction == "SELL":
            if has_bull and not has_bear:
                is_favorable = False
                reason = f"Bullish price action contradicts SELL setup: {patterns_found}"
            elif has_bear:
                direction_confirmed = "SELL"
                conf_boost = 0.25
                reason = f"Bearish confirmation: {patterns_found}"
            else:
                reason = "Neutral price action (no adverse pattern)"
        else:
            reason = f"Detected: {patterns_found}" if patterns_found else "Neutral"

        return SkillResult(
            is_favorable=is_favorable,
            confidence=round(0.50 + conf_boost, 2),
            signal=direction_confirmed or intended_direction,
            reason=reason,
            metadata={
                "patterns": patterns_found,
                "at_key_level": at_sr,
                "key_level": sr_lvl
            }
        )
