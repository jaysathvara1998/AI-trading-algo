"""
Adaptive Wave-Based M-Pattern & W-Pattern AI Trading Skill
(Human-Brain Swing Wave Logic: UP -> MID DOWN -> MID UP -> DOWN & DOWN -> MID UP -> MID DOWN -> UP)

Works on:
- Multi-Timeframe: 1m, 3m, 5m, 15m, 1h, 4h, Daily, Weekly
- Multi-Asset: Spot Index (NIFTY, BANKNIFTY, SENSEX) and Option Premium charts (CE/PE)
- Scale-Free: No fixed 10-bar, 25-bar, or 40-bar limits. Swings adapt dynamically to market moves.
- Continuous Live Market: Operates across all market sessions (morning, mid-day, afternoon).
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from .base_skill import BaseTradingSkill, SkillResult


class PatternType(Enum):
    M_PATTERN = "M_PATTERN"  # Double Top: UP -> MID DOWN -> MID UP -> DOWN (Breakdown)
    W_PATTERN = "W_PATTERN"  # Double Bottom: DOWN -> MID UP -> MID DOWN -> UP (Breakout)


class MPatternStage(Enum):
    NONE = "NONE"
    FORMING_LEG_1 = "FORMING_LEG_1"         # Leg 1: Initial directional impulse
    FORMING_LEG_2_MID = "FORMING_LEG_2_MID" # Leg 2: Mid retracement defining the Neckline
    FORMING_LEG_3_MID = "FORMING_LEG_3_MID" # Leg 3: Second push to re-test the extreme
    TESTING_NECKLINE = "TESTING_NECKLINE"   # Leg 4: Pullback re-testing the mid neckline level
    BREAKDOWN_CONFIRMED = "BREAKDOWN_CONFIRMED" # Leg 4: Decisive breakdown below neckline (M-Pattern)
    BREAKOUT_CONFIRMED = "BREAKOUT_CONFIRMED"   # Leg 4: Decisive breakout above neckline (W-Pattern)
    TARGET_ACHIEVED = "TARGET_ACHIEVED"     # 1x or 1.618x extension target reached


@dataclass
class SwingLeg:
    """Represents a single directional price swing (independent of bar count)"""
    direction: str       # "UP" or "DOWN"
    start_price: float
    start_idx: int
    end_price: float
    end_idx: int
    magnitude: float
    bar_count: int


@dataclass
class MPatternMetrics:
    stage: MPatternStage
    pattern_type: PatternType
    peak1_price: float
    peak1_index: int
    valley_price: float
    valley_index: int
    peak2_price: float
    peak2_index: int
    neckline_level: float
    pattern_height: float
    target_1x: float
    target_1_618x: float
    stop_loss: float
    symmetry_score: float         # 0.0 to 1.0 (closeness of peaks / troughs)
    volume_expansion: bool        # Higher volume on breakdown/breakout
    quality_score: float          # Setup confidence score
    legs: List[SwingLeg] = field(default_factory=list)


@dataclass
class TrendlineMetrics:
    line_type: str                  # "DESCENDING_RESISTANCE" or "ASCENDING_SUPPORT"
    p1_price: float
    p1_idx: int
    p2_price: float
    p2_idx: int
    slope: float                    # Points per bar (negative for descending, positive for ascending)
    current_trendline_level: float  # Projected trendline price at current bar
    current_price: float
    breakout_type: str              # "BREAKOUT_UP", "BREAKDOWN_DOWN", "TESTING", "NONE"
    confidence: float
    description: str


class MPatternSkill(BaseTradingSkill):
    """
    Wave-based M-Pattern (Double Top) and W-Pattern (Double Bottom) AI Trading Skill.
    Implements human trader wave logic:
      - M-Pattern: UP -> MID DOWN -> MID UP -> DOWN (Breakdown)
      - W-Pattern: DOWN -> MID UP -> MID DOWN -> UP (Breakout)
    Scale-free and timeframe-agnostic (1m, 5m, 15m, 1h, Daily, Options CE/PE).
    """

    def __init__(
        self,
        name: str = "MPatternSkill",
        max_peak_diff_pct: float = 0.0080,   # Up to 0.8% diff between peaks (flexible across spot & options)
        min_pattern_depth_pts: float = 3.0,  # Minimum depth in pts (supports options from 3pts, spot from 8pts)
        min_swing_pct: float = 0.0004,       # Minimum swing reversal threshold (0.04% - approx 9-10 pts on NIFTY)
        min_bars_between_peaks: int = 2,     # Minimum bar separation
        max_lookback_bars: int = 150         # Extended lookback so high-density 1m charts are fully covered
    ):
        super().__init__(name=name)
        self.max_peak_diff_pct = max_peak_diff_pct
        self.min_pattern_depth_pts = min_pattern_depth_pts
        self.min_swing_pct = min_swing_pct
        self.min_bars_between_peaks = min_bars_between_peaks
        self.max_lookback_bars = max_lookback_bars

    def extract_swing_legs(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        min_reversal_pts: float
    ) -> List[SwingLeg]:
        """
        Extracts directional ZigZag swing legs (UP and DOWN) regardless of how many bars each leg takes.
        """
        n_bars = len(closes)
        if n_bars < 3:
            return []

        swings: List[SwingLeg] = []
        
        # Initialize first swing direction
        curr_dir = "UP" if closes[min(2, n_bars - 1)] >= closes[0] else "DOWN"
        swing_start_idx = 0
        swing_start_p = lows[0] if curr_dir == "UP" else highs[0]
        curr_extreme_idx = 0
        curr_extreme_p = swing_start_p

        for i in range(n_bars):
            h, l, c = highs[i], lows[i], closes[i]

            if curr_dir == "UP":
                if h > curr_extreme_p:
                    curr_extreme_p = h
                    curr_extreme_idx = i
                # Check for reversal to DOWN
                reversal = curr_extreme_p - l
                if reversal >= min_reversal_pts and i > curr_extreme_idx:
                    # Finalize UP swing
                    swings.append(SwingLeg(
                        direction="UP",
                        start_price=round(float(swing_start_p), 2),
                        start_idx=swing_start_idx,
                        end_price=round(float(curr_extreme_p), 2),
                        end_idx=curr_extreme_idx,
                        magnitude=round(float(curr_extreme_p - swing_start_p), 2),
                        bar_count=max(1, curr_extreme_idx - swing_start_idx + 1)
                    ))
                    # Start DOWN swing
                    curr_dir = "DOWN"
                    swing_start_idx = curr_extreme_idx
                    swing_start_p = curr_extreme_p
                    curr_extreme_p = l
                    curr_extreme_idx = i

            else:  # curr_dir == "DOWN"
                if l < curr_extreme_p:
                    curr_extreme_p = l
                    curr_extreme_idx = i
                # Check for reversal to UP
                reversal = h - curr_extreme_p
                if reversal >= min_reversal_pts and i > curr_extreme_idx:
                    # Finalize DOWN swing
                    swings.append(SwingLeg(
                        direction="DOWN",
                        start_price=round(float(swing_start_p), 2),
                        start_idx=swing_start_idx,
                        end_price=round(float(curr_extreme_p), 2),
                        end_idx=curr_extreme_idx,
                        magnitude=round(float(swing_start_p - curr_extreme_p), 2),
                        bar_count=max(1, curr_extreme_idx - swing_start_idx + 1)
                    ))
                    # Start UP swing
                    curr_dir = "UP"
                    swing_start_idx = curr_extreme_idx
                    swing_start_p = curr_extreme_p
                    curr_extreme_p = h
                    curr_extreme_idx = i

        # Append trailing active swing
        if curr_extreme_idx > swing_start_idx:
            mag = abs(curr_extreme_p - swing_start_p)
            swings.append(SwingLeg(
                direction=curr_dir,
                start_price=round(float(swing_start_p), 2),
                start_idx=swing_start_idx,
                end_price=round(float(curr_extreme_p), 2),
                end_idx=curr_extreme_idx,
                magnitude=round(float(mag), 2),
                bar_count=max(1, curr_extreme_idx - swing_start_idx + 1)
            ))

        return swings

    def scan_m_pattern(
        self,
        ohlcv: Union[pd.DataFrame, List[dict]],
        current_price: Optional[float] = None
    ) -> Optional[MPatternMetrics]:
        """
        Scans OHLCV sequence for M-Pattern (or W-Pattern) using the 4-Leg Wave State Machine:
        M-Pattern: [UP] -> [MID DOWN] -> [MID UP] -> [DOWN Breakdown]
        W-Pattern: [DOWN] -> [MID UP] -> [MID DOWN] -> [UP Breakout]
        """
        if isinstance(ohlcv, pd.DataFrame):
            df = ohlcv.tail(self.max_lookback_bars).copy()
            highs = df["high"].values.astype(float)
            lows = df["low"].values.astype(float)
            closes = df["close"].values.astype(float)
            volumes = df.get("volume", np.ones(len(df))).values.astype(float)
        else:
            recent = ohlcv[-self.max_lookback_bars:]
            highs = np.array([float(x.get("high", x["close"])) for x in recent])
            lows = np.array([float(x.get("low", x["close"])) for x in recent])
            closes = np.array([float(x["close"]) for x in recent])
            volumes = np.array([float(x.get("volume", 1.0)) for x in recent])

        n_bars = len(closes)
        if n_bars < 5:
            return None

        curr_p = current_price if (current_price and current_price > 0) else closes[-1]
        mean_p = np.mean(closes)

        # Dynamic reversal threshold: 0.04% of price (capped between 2.5 and 12.0 pts)
        reversal_pts = max(self.min_pattern_depth_pts * 0.35, min(12.0, mean_p * self.min_swing_pct))

        # 1. Primary Engine: Wave-Based Leg State Machine
        swings = self.extract_swing_legs(highs, lows, closes, min_reversal_pts=reversal_pts)

        # Search for 4-leg M-pattern: UP -> MID DOWN -> MID UP -> DOWN
        best_metrics = self._match_wave_m_pattern(swings, highs, lows, closes, volumes, curr_p)
        if best_metrics:
            return best_metrics

        # 2. Secondary fallback: High-precision Bar-level scan (for compact/tight 5m candles)
        return self._scan_compact_bars(highs, lows, closes, volumes, curr_p)

    def _match_wave_m_pattern(
        self,
        swings: List[SwingLeg],
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        volumes: np.ndarray,
        curr_p: float
    ) -> Optional[MPatternMetrics]:
        """
        Matches 4-leg sequence:
        Leg 1: UP (to Peak 1)
        Leg 2: MID DOWN (to Valley / Neckline)
        Leg 3: MID UP (to Peak 2)
        Leg 4: DOWN (breaking Neckline)
        """
        n_bars = len(closes)
        if len(swings) < 3 or n_bars < 5:
            return None

        best_m: Optional[MPatternMetrics] = None
        best_q: float = 0.0

        for i in range(len(swings) - 2):
            s1, s2, s3 = swings[i], swings[i + 1], swings[i + 2]

            if s1.direction == "UP" and s2.direction == "DOWN" and s3.direction == "UP":
                h1 = s1.end_price
                p1_idx = s1.end_idx
                valley = s2.end_price
                v_idx = s2.end_idx
                h2 = s3.end_price
                p2_idx = s3.end_idx

                # Check if price broke above double top peaks (Pattern Invalidation)
                trailing_sub = closes[p2_idx:] if p2_idx < n_bars else np.array([curr_p])
                min_trailing = np.min(trailing_sub) if len(trailing_sub) > 0 else curr_p
                max_trailing = np.max(trailing_sub) if len(trailing_sub) > 0 else curr_p

                # If price rose above double top ceiling, this M pattern is DEAD (bullish breakout)
                if max_trailing > (max(h1, h2) + 0.5) or curr_p > (max(h1, h2) + 0.5):
                    continue

                # If pattern is too old (> 25 bars since Peak 2 without breaking neckline), expire it
                if (n_bars - 1 - p2_idx) > 25 and min_trailing > valley:
                    continue

                diff_pct = abs(h1 - h2) / max(0.01, h1)
                if diff_pct > self.max_peak_diff_pct:
                    continue

                depth = min(h1, h2) - valley
                if depth < self.min_pattern_depth_pts:
                    continue

                height = max(h1, h2) - valley
                tp1 = valley - height
                tp2 = valley - (1.618 * height)
                sl = max(h1, h2) + max(1.5, height * 0.15)

                symmetry = max(0.0, 1.0 - (diff_pct / self.max_peak_diff_pct))
                mean_vol = np.mean(volumes) if len(volumes) > 0 else 1.0
                vol_expansion = volumes[-1] >= mean_vol * 1.10

                breakdown_margin = max(0.3, height * 0.05)
                # Only valid if price is actually falling from Peak 2 towards neckline
                if curr_p <= (valley - breakdown_margin) or min_trailing <= (valley - breakdown_margin):
                    if curr_p <= tp1:
                        stage = MPatternStage.TARGET_ACHIEVED
                    else:
                        stage = MPatternStage.BREAKDOWN_CONFIRMED
                elif abs(curr_p - valley) <= max(2.0, height * 0.15) and curr_p < max(h1, h2):
                    stage = MPatternStage.TESTING_NECKLINE
                elif curr_p < (max(h1, h2) - height * 0.40) and curr_p > valley:
                    stage = MPatternStage.FORMING_LEG_3_MID
                else:
                    # Still hovering near the top without drop momentum - do not trigger
                    continue

                quality = (symmetry * 0.40) + (min(1.0, depth / (height + 1e-5)) * 0.35)
                if vol_expansion:
                    quality += 0.25
                quality = min(1.0, quality)

                if quality > best_q:
                    best_q = quality
                    best_m = MPatternMetrics(
                        stage=stage,
                        pattern_type=PatternType.M_PATTERN,
                        peak1_price=round(float(h1), 2),
                        peak1_index=p1_idx,
                        valley_price=round(float(valley), 2),
                        valley_index=v_idx,
                        peak2_price=round(float(h2), 2),
                        peak2_index=p2_idx,
                        neckline_level=round(float(valley), 2),
                        pattern_height=round(float(height), 2),
                        target_1x=round(float(tp1), 2),
                        target_1_618x=round(float(tp2), 2),
                        stop_loss=round(float(sl), 2),
                        symmetry_score=round(float(symmetry), 3),
                        volume_expansion=vol_expansion,
                        quality_score=round(float(quality), 3),
                        legs=[s1, s2, s3]
                    )

        return best_m

    def scan_w_pattern(
        self,
        ohlcv: Union[pd.DataFrame, List[dict]],
        current_price: Optional[float] = None
    ) -> Optional[MPatternMetrics]:
        """
        Scans for W-Pattern (Double Bottom) using 4-Leg Wave State Machine:
        Leg 1: DOWN (to Trough 1)
        Leg 2: MID UP (to Mid Peak / Neckline)
        Leg 3: MID DOWN (to Trough 2)
        Leg 4: UP (breaking Neckline upwards)
        """
        if isinstance(ohlcv, pd.DataFrame):
            df = ohlcv.tail(self.max_lookback_bars).copy()
            highs = df["high"].values.astype(float)
            lows = df["low"].values.astype(float)
            closes = df["close"].values.astype(float)
            volumes = df.get("volume", np.ones(len(df))).values.astype(float)
        else:
            recent = ohlcv[-self.max_lookback_bars:]
            highs = np.array([float(x.get("high", x["close"])) for x in recent])
            lows = np.array([float(x.get("low", x["close"])) for x in recent])
            closes = np.array([float(x["close"]) for x in recent])
            volumes = np.array([float(x.get("volume", 1.0)) for x in recent])

        n_bars = len(closes)
        if n_bars < 5:
            return None

        curr_p = current_price if (current_price and current_price > 0) else closes[-1]
        mean_p = np.mean(closes)
        reversal_pts = max(self.min_pattern_depth_pts * 0.35, min(12.0, mean_p * self.min_swing_pct))

        swings = self.extract_swing_legs(highs, lows, closes, min_reversal_pts=reversal_pts)
        if len(swings) < 3:
            return None

        best_w: Optional[MPatternMetrics] = None
        best_q: float = 0.0

        for i in range(len(swings) - 2):
            s1, s2, s3 = swings[i], swings[i + 1], swings[i + 2]

            if s1.direction == "DOWN" and s2.direction == "UP" and s3.direction == "DOWN":
                l1 = s1.end_price
                p1_idx = s1.end_idx
                mid_peak = s2.end_price
                v_idx = s2.end_idx
                l2 = s3.end_price
                p2_idx = s3.end_idx

                # Check if price broke below double bottom troughs (Pattern Invalidation)
                trailing_sub = closes[p2_idx:] if p2_idx < n_bars else np.array([curr_p])
                min_trailing = np.min(trailing_sub) if len(trailing_sub) > 0 else curr_p
                max_trailing = np.max(trailing_sub) if len(trailing_sub) > 0 else curr_p

                # If price fell below the double bottom floor, this W pattern is DEAD (bearish continuation)
                if min_trailing < (min(l1, l2) - 0.5) or curr_p < (min(l1, l2) - 0.5):
                    continue

                # If pattern is too old (> 25 bars since Trough 2 without breaking neckline), expire it
                if (n_bars - 1 - p2_idx) > 25 and max_trailing < mid_peak:
                    continue

                diff_pct = abs(l1 - l2) / max(0.01, l1)
                if diff_pct > self.max_peak_diff_pct:
                    continue

                depth = mid_peak - max(l1, l2)
                if depth < self.min_pattern_depth_pts:
                    continue

                height = mid_peak - min(l1, l2)
                tp1 = mid_peak + height
                tp2 = mid_peak + (1.618 * height)
                sl = min(l1, l2) - max(1.5, height * 0.15)

                symmetry = max(0.0, 1.0 - (diff_pct / self.max_peak_diff_pct))
                mean_vol = np.mean(volumes) if len(volumes) > 0 else 1.0
                vol_expansion = volumes[-1] >= mean_vol * 1.10

                breakout_margin = max(0.3, height * 0.05)
                # Only valid if price is actually rising from Trough 2 towards neckline
                if curr_p >= (mid_peak + breakout_margin) or max_trailing >= (mid_peak + breakout_margin):
                    if curr_p >= tp1:
                        stage = MPatternStage.TARGET_ACHIEVED
                    else:
                        stage = MPatternStage.BREAKOUT_CONFIRMED
                elif abs(curr_p - mid_peak) <= max(2.0, height * 0.15) and curr_p > min(l1, l2):
                    stage = MPatternStage.TESTING_NECKLINE
                elif curr_p > (min(l1, l2) + height * 0.40) and curr_p < mid_peak:
                    stage = MPatternStage.FORMING_LEG_3_MID
                else:
                    # Still hovering near the bottom without bounce momentum - do not trigger
                    continue

                quality = (symmetry * 0.40) + (min(1.0, depth / (height + 1e-5)) * 0.35)
                if vol_expansion:
                    quality += 0.25
                quality = min(1.0, quality)

                if quality > best_q:
                    best_q = quality
                    best_w = MPatternMetrics(
                        stage=stage,
                        pattern_type=PatternType.W_PATTERN,
                        peak1_price=round(float(l1), 2),
                        peak1_index=p1_idx,
                        valley_price=round(float(mid_peak), 2),
                        valley_index=v_idx,
                        peak2_price=round(float(l2), 2),
                        peak2_index=p2_idx,
                        neckline_level=round(float(mid_peak), 2),
                        pattern_height=round(float(height), 2),
                        target_1x=round(float(tp1), 2),
                        target_1_618x=round(float(tp2), 2),
                        stop_loss=round(float(sl), 2),
                        symmetry_score=round(float(symmetry), 3),
                        volume_expansion=vol_expansion,
                        quality_score=round(float(quality), 3),
                        legs=[s1, s2, s3]
                    )

        return best_w

    def scan_trendline_breakout(
        self,
        ohlcv: Union[pd.DataFrame, List[dict]],
        current_price: Optional[float] = None
    ) -> Optional[TrendlineMetrics]:
        """
        Detects Descending Resistance Trendlines (connecting lower highs) and
        Ascending Support Trendlines (connecting higher lows).
        Fires BREAKOUT_UP when price decisively crosses above a descending trendline,
        and BREAKDOWN_DOWN when price decisively crosses below an ascending trendline.
        """
        if isinstance(ohlcv, pd.DataFrame):
            df = ohlcv.tail(self.max_lookback_bars).copy()
            highs = df["high"].values.astype(float)
            lows = df["low"].values.astype(float)
            closes = df["close"].values.astype(float)
        else:
            recent = ohlcv[-self.max_lookback_bars:]
            highs = np.array([float(x.get("high", x["close"])) for x in recent])
            lows = np.array([float(x.get("low", x["close"])) for x in recent])
            closes = np.array([float(x["close"]) for x in recent])

        n_bars = len(closes)
        if n_bars < 8:
            return None

        curr_p = current_price if (current_price and current_price > 0) else closes[-1]
        mean_p = np.mean(closes)
        reversal_pts = max(self.min_pattern_depth_pts * 0.35, min(12.0, mean_p * self.min_swing_pct))
        swings = self.extract_swing_legs(highs, lows, closes, min_reversal_pts=reversal_pts)

        # Collect all structural swing peaks and valleys
        peaks_dict = {}
        valleys_dict = {}
        for s in swings:
            if s.direction == "DOWN":
                peaks_dict[s.start_idx] = s.start_price
                valleys_dict[s.end_idx] = s.end_price
            else:
                valleys_dict[s.start_idx] = s.start_price
                peaks_dict[s.end_idx] = s.end_price

        sorted_peaks = sorted(peaks_dict.items(), key=lambda x: x[0])     # [(idx, price), ...]
        sorted_valleys = sorted(valleys_dict.items(), key=lambda x: x[0]) # [(idx, price), ...]

        # 1. Evaluate Ascending Support Trendlines (Connecting Higher Lows) -> Check for Breakdown or Floor Bounce
        active_ascending_breakdown = None
        active_descending_breakout = None
        active_ceiling_reject = None

        if len(sorted_valleys) >= 2:
            # Check from most recent valley pairs backwards
            for i in range(len(sorted_valleys) - 2, -1, -1):
                idx1, v1_price = sorted_valleys[i]
                idx2, v2_price = sorted_valleys[i + 1]
                # Higher low separated by at least 2 bars
                if v2_price > (v1_price + 1.5) and (idx2 - idx1) >= 2:
                    slope = (v2_price - v1_price) / float(idx2 - idx1)
                    curr_idx = n_bars - 1
                    tl_level = v2_price + (slope * (curr_idx - idx2))

                    # Decisive breakdown below ascending trendline (at least 0.5 pt buffer)
                    if curr_p < (tl_level - 0.5):
                        active_ascending_breakdown = TrendlineMetrics(
                            line_type="ASCENDING_SUPPORT",
                            p1_price=v1_price,
                            p1_idx=idx1,
                            p2_price=v2_price,
                            p2_idx=idx2,
                            slope=round(slope, 3),
                            current_trendline_level=round(tl_level, 2),
                            current_price=round(curr_p, 2),
                            breakout_type="BREAKDOWN_DOWN",
                            confidence=0.90,
                            description=(
                                f"Bearish Ascending Trendline Breakdown below {tl_level:.1f}! "
                                f"(Valley1={v1_price:.1f} -> Valley2={v2_price:.1f})"
                            )
                        )
                        break

        # 2. Evaluate Descending Resistance Trendlines (Connecting Lower Highs) -> Check for Breakout or Ceiling Reject
        if len(sorted_peaks) >= 2:
            # Check from most prominent/recent peak pairs backwards
            for i in range(len(sorted_peaks) - 2, -1, -1):
                idx1, p1_price = sorted_peaks[i]
                idx2, p2_price = sorted_peaks[i + 1]
                # Lower high separated by at least 2 bars
                if p2_price < (p1_price - 1.5) and (idx2 - idx1) >= 2:
                    slope = (p2_price - p1_price) / float(idx2 - idx1)
                    curr_idx = n_bars - 1
                    tl_level = p2_price + (slope * (curr_idx - idx2))

                    # Decisive breakout above descending trendline (at least 0.5 pt buffer)
                    if curr_p > (tl_level + 0.5):
                        active_descending_breakout = TrendlineMetrics(
                            line_type="DESCENDING_RESISTANCE",
                            p1_price=p1_price,
                            p1_idx=idx1,
                            p2_price=p2_price,
                            p2_idx=idx2,
                            slope=round(slope, 3),
                            current_trendline_level=round(tl_level, 2),
                            current_price=round(curr_p, 2),
                            breakout_type="BREAKOUT_UP",
                            confidence=0.90,
                            description=(
                                f"Bullish Descending Trendline Breakout above {tl_level:.1f}! "
                                f"(Peak1={p1_price:.1f} -> Peak2={p2_price:.1f})"
                            )
                        )
                        break
                    elif curr_p <= tl_level and (tl_level - curr_p) <= 15.0:
                        # Price is trading directly under descending trendline ceiling (Resistance zone)
                        if active_ceiling_reject is None:
                            active_ceiling_reject = TrendlineMetrics(
                                line_type="DESCENDING_RESISTANCE",
                                p1_price=p1_price,
                                p1_idx=idx1,
                                p2_price=p2_price,
                                p2_idx=idx2,
                                slope=round(slope, 3),
                                current_trendline_level=round(tl_level, 2),
                                current_price=round(curr_p, 2),
                                breakout_type="RESISTANCE_CEILING_REJECT",
                                confidence=0.86,
                                description=(
                                    f"Major Descending Trendline Resistance Ceiling at {tl_level:.1f}. "
                                    f"Price rejected under trendline (Peak1={p1_price:.1f} -> Peak2={p2_price:.1f}). BUY CALL FORBIDDEN!"
                                )
                            )

        # Priority 1: Active Breakout / Breakdown triggers
        if active_ascending_breakdown is not None:
            return active_ascending_breakdown
        if active_descending_breakout is not None:
            return active_descending_breakout
        # Priority 2: Resistance ceiling veto
        if active_ceiling_reject is not None:
            return active_ceiling_reject

        return None

    def _scan_compact_bars(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        volumes: np.ndarray,
        curr_p: float
    ) -> Optional[MPatternMetrics]:
        """
        Compact bar-level fallback for tight timeframe candles.
        """
        n_bars = len(closes)
        best_metrics: Optional[MPatternMetrics] = None
        best_quality: float = 0.0

        for p1_idx in range(0, n_bars - 2):
            if p1_idx > 0 and highs[p1_idx] < highs[p1_idx - 1]:
                continue
            if p1_idx + 1 < n_bars and highs[p1_idx] < highs[p1_idx + 1]:
                continue
            h1 = highs[p1_idx]

            for p2_idx in range(p1_idx + self.min_bars_between_peaks, n_bars):
                if p2_idx > 0 and highs[p2_idx] < highs[p2_idx - 1]:
                    continue
                if p2_idx + 1 < n_bars and highs[p2_idx] < highs[p2_idx + 1]:
                    continue
                h2 = highs[p2_idx]

                peak_diff = abs(h1 - h2)
                diff_pct = peak_diff / max(0.01, h1)
                if diff_pct > self.max_peak_diff_pct:
                    continue

                if p2_idx <= p1_idx + 1:
                    continue
                valley_sub = lows[p1_idx + 1 : p2_idx]
                if len(valley_sub) == 0:
                    continue
                v_local_idx = int(np.argmin(valley_sub))
                v_idx = p1_idx + 1 + v_local_idx
                valley = lows[v_idx]

                depth = min(h1, h2) - valley
                if depth < self.min_pattern_depth_pts:
                    continue

                symmetry = max(0.0, 1.0 - (diff_pct / self.max_peak_diff_pct))
                mean_vol = np.mean(volumes) if len(volumes) > 0 else 1.0
                vol_expansion = volumes[-1] >= mean_vol * 1.10

                height = max(h1, h2) - valley
                tp1 = valley - height
                tp2 = valley - (1.618 * height)
                sl = max(h1, h2) + 2.0

                breakdown_margin = 0.5
                if curr_p <= (valley - breakdown_margin):
                    if curr_p <= tp1:
                        stage = MPatternStage.TARGET_ACHIEVED
                    else:
                        stage = MPatternStage.BREAKDOWN_CONFIRMED
                elif abs(curr_p - valley) <= 2.0:
                    stage = MPatternStage.TESTING_NECKLINE
                else:
                    stage = MPatternStage.FORMING_LEG_3_MID

                quality = (symmetry * 0.45) + (min(1.0, depth / 20.0) * 0.35)
                if vol_expansion:
                    quality += 0.20
                quality = min(1.0, quality)

                if quality > best_quality:
                    best_quality = quality
                    best_metrics = MPatternMetrics(
                        stage=stage,
                        pattern_type=PatternType.M_PATTERN,
                        peak1_price=round(float(h1), 2),
                        peak1_index=p1_idx,
                        valley_price=round(float(valley), 2),
                        valley_index=v_idx,
                        peak2_price=round(float(h2), 2),
                        peak2_index=p2_idx,
                        neckline_level=round(float(valley), 2),
                        pattern_height=round(float(height), 2),
                        target_1x=round(float(tp1), 2),
                        target_1_618x=round(float(tp2), 2),
                        stop_loss=round(float(sl), 2),
                        symmetry_score=round(float(symmetry), 3),
                        volume_expansion=vol_expansion,
                        quality_score=round(float(quality), 3)
                    )

        return best_metrics

    def evaluate(
        self,
        ohlcv: Union[pd.DataFrame, List[dict]],
        current_price: Optional[float] = None,
        intended_direction: Optional[str] = None
    ) -> SkillResult:
        """
        Executes domain evaluation of the M/W-pattern skill.
        Returns standardized SkillResult:
        - M-Pattern Breakdown -> BUY_PUT (Strong bearish confirmation)
        - W-Pattern Breakout  -> BUY_CALL (Strong bullish confirmation on spot / option breakout)
        """
        # 1. First check Trendline Breakout (Descending Resistance or Ascending Support)
        tl_metrics = self.scan_trendline_breakout(ohlcv, current_price=current_price)

        # 2. Check W-pattern (Double Bottom Reversal)
        w_metrics = self.scan_w_pattern(ohlcv, current_price=current_price)

        # 3. Check M-pattern (Double Top Reversal)
        metrics = self.scan_m_pattern(ohlcv, current_price=current_price)

        # Dual Confluence Check: W-Pattern + Descending Trendline Breakout
        if w_metrics and w_metrics.stage in (MPatternStage.BREAKOUT_CONFIRMED, MPatternStage.TESTING_NECKLINE) and tl_metrics and tl_metrics.breakout_type == "BREAKOUT_UP":
            return SkillResult(
                is_favorable=True,
                confidence=0.96,
                signal="BUY_CALL",
                reason=(
                    f"MAXIMUM CONVICTION CONFLUENCE! W-Pattern Breakout above {w_metrics.neckline_level:.1f} "
                    f"AND Descending Trendline Breakout above {tl_metrics.current_trendline_level:.1f}! "
                    f"Target: {w_metrics.target_1x:.1f} | SL: {w_metrics.stop_loss:.1f}"
                ),
                metadata={"pattern": "W_AND_TRENDLINE_CONFLUENCE", "neckline": w_metrics.neckline_level, "target_1x": w_metrics.target_1x}
            )

        # Standalone Trendline Breakout
        if tl_metrics and tl_metrics.breakout_type == "BREAKOUT_UP":
            return SkillResult(
                is_favorable=True,
                confidence=tl_metrics.confidence,
                signal="BUY_CALL",
                reason=tl_metrics.description,
                metadata={"pattern": "TRENDLINE_BREAKOUT_UP", "trendline": tl_metrics.current_trendline_level}
            )
        elif tl_metrics and tl_metrics.breakout_type == "BREAKDOWN_DOWN":
            return SkillResult(
                is_favorable=True,
                confidence=tl_metrics.confidence,
                signal="BUY_PUT",
                reason=tl_metrics.description,
                metadata={"pattern": "TRENDLINE_BREAKDOWN_DOWN", "trendline": tl_metrics.current_trendline_level}
            )

        # W-Pattern Breakout or Active Reversal
        if w_metrics:
            if w_metrics.stage == MPatternStage.BREAKOUT_CONFIRMED:
                conf = max(0.82, min(0.95, 0.75 + (w_metrics.quality_score * 0.20)))
                return SkillResult(
                    is_favorable=True,
                    confidence=round(conf, 2),
                    signal="BUY_CALL",
                    reason=(
                        f"W-Pattern (Double Bottom) confirmed! Leg1={w_metrics.peak1_price:.1f}, "
                        f"Leg3={w_metrics.peak2_price:.1f}, Neckline={w_metrics.neckline_level:.1f}. "
                        f"Decisive breakout above neckline. Target 1x: {w_metrics.target_1x:.1f} | SL: {w_metrics.stop_loss:.1f}"
                    ),
                    metadata={"pattern": "W_PATTERN", "neckline": w_metrics.neckline_level, "target_1x": w_metrics.target_1x}
                )
            elif w_metrics.stage == MPatternStage.TESTING_NECKLINE:
                return SkillResult(
                    is_favorable=True,
                    confidence=0.80,
                    signal="BUY_CALL",
                    reason=f"W-Pattern formed! Currently testing neckline at {w_metrics.neckline_level:.1f}. Awaiting upside expansion.",
                    metadata={"pattern": "W_TESTING_NECKLINE", "neckline": w_metrics.neckline_level}
                )

        if not metrics:
            return SkillResult(
                is_favorable=False,
                confidence=0.50,
                signal="NEUTRAL",
                reason="No M-pattern, W-pattern, or trendline breakout confirmed in recent wave structure."
            )

        meta = {
            "stage": metrics.stage.value,
            "peak1": metrics.peak1_price,
            "valley": metrics.valley_price,
            "peak2": metrics.peak2_price,
            "height": metrics.pattern_height,
            "neckline": metrics.neckline_level,
            "target_1x": metrics.target_1x,
            "target_1_618x": metrics.target_1_618x,
            "stop_loss": metrics.stop_loss,
            "quality": metrics.quality_score,
            "legs_count": len(metrics.legs)
        }

        # 1. Actionable Breakdown Trigger -> Strong SELL / BUY PUT
        if metrics.stage == MPatternStage.BREAKDOWN_CONFIRMED:
            conf = max(0.80, min(0.95, 0.75 + (metrics.quality_score * 0.20)))
            reason = (
                f"M-Pattern (Double Top) confirmed! Wave: UP->MID DOWN->MID UP->DOWN | "
                f"Peak1={metrics.peak1_price:.1f}, Peak2={metrics.peak2_price:.1f}, "
                f"Valley/Neckline={metrics.neckline_level:.1f}. Decisive breakdown below neckline. "
                f"Target 1x: {metrics.target_1x:.1f} | Target 1.618x: {metrics.target_1_618x:.1f} | SL: {metrics.stop_loss:.1f}"
            )
            return SkillResult(
                is_favorable=True,
                confidence=round(conf, 2),
                signal="BUY_PUT",
                reason=reason,
                metadata=meta
            )

        # 2. Testing Neckline -> Setup alert
        if metrics.stage == MPatternStage.TESTING_NECKLINE:
            return SkillResult(
                is_favorable=False,
                confidence=0.70,
                signal="ALERT_M_NECKLINE_TEST",
                reason=f"M-Pattern formed! Currently testing neckline support at {metrics.neckline_level:.1f}. Awaiting breakdown candle.",
                metadata=meta
            )

        # 3. Target Achieved -> Avoid chasing bottom
        if metrics.stage == MPatternStage.TARGET_ACHIEVED:
            return SkillResult(
                is_favorable=False,
                confidence=0.60,
                signal="M_PATTERN_TARGET_COMPLETED",
                reason=f"M-Pattern 1x target ({metrics.target_1x:.1f}) already achieved. Risk of mean reversion.",
                metadata=meta
            )

        return SkillResult(
            is_favorable=False,
            confidence=0.50,
            signal="M_PATTERN_FORMING",
            reason=f"M-Pattern forming (Stage: {metrics.stage.value}).",
            metadata=meta
        )
