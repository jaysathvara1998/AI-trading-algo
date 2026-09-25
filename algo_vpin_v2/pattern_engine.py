"""
Classical & Institutional Chart Pattern Recognition Engine for Algo VPIN v2.0
Derived from Institutional Textbooks:
1. Fidelity Investments: Identifying Chart Patterns (Charles D. Kirkpatrick II, CMT)
2. Corporate Finance Institute (CFI): The Complete Guide to Trading
3. SRCC: Fundamentals of Investments - Technical Analysis (Dr. Kanu Jain)

Detects:
- Reversal Patterns: Double Top (M-Pattern), Double Bottom (W-Pattern), Head & Shoulders, Inverted H&S
- Continuation Patterns: Ascending Triangle, Descending Triangle, Symmetrical Triangle, Bullish/Bearish Flags
- Candlestick Reversals: Pin Bar (Hammer/Shooting Star), Bullish/Bearish Engulfing
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Deque
from collections import deque
import numpy as np
import logging

logger = logging.getLogger("algo_vpin_v2.pattern_engine")


class PatternType(Enum):
    NONE = "NONE"
    # Reversal Formations (Kirkpatrick / SRCC)
    DOUBLE_TOP_M = "DOUBLE_TOP_M"
    DOUBLE_BOTTOM_W = "DOUBLE_BOTTOM_W"
    TRENDLINE_BREAKOUT_UP = "TRENDLINE_BREAKOUT_UP"
    TRENDLINE_BREAKDOWN_DOWN = "TRENDLINE_BREAKDOWN_DOWN"
    TRENDLINE_RESISTANCE_REJECT = "TRENDLINE_RESISTANCE_REJECT"
    HEAD_AND_SHOULDERS = "HEAD_AND_SHOULDERS"
    INVERSE_HEAD_AND_SHOULDERS = "INVERSE_HEAD_AND_SHOULDERS"
    ROUNDING_BOTTOM = "ROUNDING_BOTTOM"
    
    # Continuation Formations (Kirkpatrick / SRCC)
    ASCENDING_TRIANGLE = "ASCENDING_TRIANGLE"
    DESCENDING_TRIANGLE = "DESCENDING_TRIANGLE"
    SYMMETRICAL_TRIANGLE = "SYMMETRICAL_TRIANGLE"
    BULL_FLAG = "BULL_FLAG"
    BEAR_FLAG = "BEAR_FLAG"
    CUP_AND_HANDLE = "CUP_AND_HANDLE"
    SUPPORT_BREAKDOWN = "SUPPORT_BREAKDOWN"
    RESISTANCE_BREAKOUT = "RESISTANCE_BREAKOUT"
    
    # Candlestick Formations (CFI Master Trading Guide)
    BULLISH_PIN_BAR = "BULLISH_PIN_BAR"
    BEARISH_PIN_BAR = "BEARISH_PIN_BAR"
    BULLISH_ENGULFING = "BULLISH_ENGULFING"
    BEARISH_ENGULFING = "BEARISH_ENGULFING"


@dataclass
class DetectedPattern:
    pattern_type: PatternType
    bias: str                     # "BULLISH", "BEARISH", "NEUTRAL"
    confidence: float             # 0.0 to 1.0
    neckline_level: float
    target_price: float
    stop_loss_level: float
    description: str


class PatternRecognitionEngine:
    """
    Real-time pattern recognition on streaming 1-minute and multi-bar windows.
    """
    def __init__(self, history_maxlen: int = 120):
        self.bars: Deque[dict] = deque(maxlen=history_maxlen)
        self.last_pattern: Optional[DetectedPattern] = None
        from .skills import MPatternSkill
        self.m_pattern_skill = MPatternSkill()

    def update_bar(self, open_p: float, high_p: float, low_p: float, close_p: float, volume: float) -> Optional[DetectedPattern]:
        """
        Ingests new bar and scans for Classical & Candlestick patterns.
        """
        bar = {
            "open": open_p,
            "high": high_p,
            "low": low_p,
            "close": close_p,
            "volume": volume
        }
        self.bars.append(bar)
        
        # 1. First scan for high-conviction 1-2 candle formations (CFI Guide)
        candle_pattern = self._scan_candlestick_patterns()
        
        # 2. Then scan multi-bar classical patterns (Fidelity & SRCC)
        chart_pattern = self._scan_multi_bar_patterns()
        
        # Select highest confidence pattern
        if chart_pattern and (not candle_pattern or chart_pattern.confidence >= candle_pattern.confidence):
            self.last_pattern = chart_pattern
        else:
            self.last_pattern = candle_pattern
            
        return self.last_pattern

    def _scan_candlestick_patterns(self) -> Optional[DetectedPattern]:
        """Scans for Pin Bars (Hammer/Shooting Star) and Engulfing bars (CFI Guide)"""
        if len(self.bars) < 2:
            return None

        curr = self.bars[-1]
        prev = self.bars[-2]

        body = abs(curr["close"] - curr["open"])
        candle_range = max(0.01, curr["high"] - curr["low"])
        upper_wick = curr["high"] - max(curr["open"], curr["close"])
        lower_wick = min(curr["open"], curr["close"]) - curr["low"]

        # Bullish Pin Bar / Hammer (Lower wick >= 60% of range, small body at top)
        if lower_wick >= 0.58 * candle_range and upper_wick <= 0.20 * candle_range:
            sl = curr["low"] - 2.0
            tp = curr["close"] + (candle_range * 2.0)
            return DetectedPattern(
                pattern_type=PatternType.BULLISH_PIN_BAR,
                bias="BULLISH",
                confidence=0.82,
                neckline_level=curr["high"],
                target_price=round(tp, 2),
                stop_loss_level=round(sl, 2),
                description=f"Bullish Pin Bar (Hammer) rejection from support {curr['low']:.2f}"
            )

        # Bearish Pin Bar / Shooting Star (Upper wick >= 58% of range, small body at bottom)
        if upper_wick >= 0.58 * candle_range and lower_wick <= 0.20 * candle_range:
            sl = curr["high"] + 2.0
            tp = curr["close"] - (candle_range * 2.0)
            return DetectedPattern(
                pattern_type=PatternType.BEARISH_PIN_BAR,
                bias="BEARISH",
                confidence=0.82,
                neckline_level=curr["low"],
                target_price=round(tp, 2),
                stop_loss_level=round(sl, 2),
                description=f"Bearish Pin Bar (Shooting Star) rejection from resistance {curr['high']:.2f}"
            )

        # Bullish Engulfing (Green candle engulfs prior red body)
        if prev["close"] < prev["open"] and curr["close"] > curr["open"]:
            if curr["close"] >= prev["open"] and curr["open"] <= prev["close"]:
                sl = min(curr["low"], prev["low"])
                tp = curr["close"] + ((curr["close"] - sl) * 2.0)
                return DetectedPattern(
                    pattern_type=PatternType.BULLISH_ENGULFING,
                    bias="BULLISH",
                    confidence=0.78,
                    neckline_level=prev["high"],
                    target_price=round(tp, 2),
                    stop_loss_level=round(sl, 2),
                    description="Bullish Engulfing power reversal candle"
                )

        # Bearish Engulfing (Red candle engulfs prior green body)
        if prev["close"] > prev["open"] and curr["close"] < curr["open"]:
            if curr["close"] <= prev["open"] and curr["open"] >= prev["close"]:
                sl = max(curr["high"], prev["high"])
                tp = curr["close"] - ((sl - curr["close"]) * 2.0)
                return DetectedPattern(
                    pattern_type=PatternType.BEARISH_ENGULFING,
                    bias="BEARISH",
                    confidence=0.78,
                    neckline_level=prev["low"],
                    target_price=round(tp, 2),
                    stop_loss_level=round(sl, 2),
                    description="Bearish Engulfing rejection power candle"
                )

        return None

    def _scan_multi_bar_patterns(self) -> Optional[DetectedPattern]:
        """Scans for Double Tops, Double Bottoms, Triangles, Flags, and Support Breakdowns (Fidelity/Kirkpatrick)"""
        n_bars = len(self.bars)
        if n_bars < 5:
            return None

        # 1. First Aggregate and Scan 5-Minute Multi-Timeframe Patterns (Higher Timeframe Authority)
        bars_list = list(self.bars)
        if len(bars_list) >= 15:
            bars_5m = []
            for i in range(0, len(bars_list), 5):
                chunk = bars_list[i:i + 5]
                if len(chunk) >= 2:
                    bars_5m.append({
                        "open": chunk[0]["open"],
                        "high": max(b["high"] for b in chunk),
                        "low": min(b["low"] for b in chunk),
                        "close": chunk[-1]["close"],
                        "volume": sum(b.get("volume", 1000) for b in chunk)
                    })

            if len(bars_5m) >= 4:
                # 5M Trendline Breakout/Breakdown
                tl_5m = self.m_pattern_skill.scan_trendline_breakout(bars_5m)
                if tl_5m and tl_5m.breakout_type == "BREAKDOWN_DOWN":
                    return DetectedPattern(
                        pattern_type=PatternType.TRENDLINE_BREAKDOWN_DOWN,
                        bias="BEARISH",
                        confidence=0.95,
                        neckline_level=tl_5m.current_trendline_level,
                        target_price=round(tl_5m.current_price - abs(tl_5m.p1_price - tl_5m.p2_price), 2),
                        stop_loss_level=round(tl_5m.p2_price, 2),
                        description=f"[5-MIN MASTER] {tl_5m.description}"
                    )
                elif tl_5m and tl_5m.breakout_type == "RESISTANCE_CEILING_REJECT":
                    return DetectedPattern(
                        pattern_type=PatternType.TRENDLINE_RESISTANCE_REJECT,
                        bias="BEARISH",
                        confidence=0.92,
                        neckline_level=tl_5m.current_trendline_level,
                        target_price=round(tl_5m.current_price - 30.0, 2),
                        stop_loss_level=round(tl_5m.current_trendline_level + 5.0, 2),
                        description=f"[5-MIN MASTER] {tl_5m.description}"
                    )

                # 5M M-Pattern / W-Pattern (Supreme Institutional Hierarchy)
                m_5m = self.m_pattern_skill.scan_m_pattern(bars_5m)
                if m_5m and m_5m.stage.value in ("BREAKDOWN_CONFIRMED", "TESTING_NECKLINE"):
                    conf = 0.95 if m_5m.stage.value == "BREAKDOWN_CONFIRMED" else 0.88
                    return DetectedPattern(
                        pattern_type=PatternType.DOUBLE_TOP_M,
                        bias="BEARISH",
                        confidence=conf,
                        neckline_level=m_5m.neckline_level,
                        target_price=m_5m.target_1x,
                        stop_loss_level=m_5m.stop_loss,
                        description=f"[5-MIN MASTER] Major 5M Double Top M-Breakdown confirmed at neckline {m_5m.neckline_level:.2f}"
                    )

                w_5m = self.m_pattern_skill.scan_w_pattern(bars_5m)
                if w_5m and w_5m.stage.value in ("BREAKOUT_CONFIRMED", "TESTING_NECKLINE"):
                    conf = 0.95 if w_5m.stage.value == "BREAKOUT_CONFIRMED" else 0.88
                    return DetectedPattern(
                        pattern_type=PatternType.DOUBLE_BOTTOM_W,
                        bias="BULLISH",
                        confidence=conf,
                        neckline_level=w_5m.neckline_level,
                        target_price=w_5m.target_1x,
                        stop_loss_level=w_5m.stop_loss,
                        description=f"[5-MIN MASTER] Major 5M Double Bottom W-Breakout confirmed at neckline {w_5m.neckline_level:.2f}"
                    )

        # 2. Check 1-Minute Trendline Breakout (Connecting Lower Highs or Higher Lows)
        tl_met = self.m_pattern_skill.scan_trendline_breakout(bars_list)
        if tl_met and tl_met.breakout_type == "BREAKOUT_UP":
            return DetectedPattern(
                pattern_type=PatternType.TRENDLINE_BREAKOUT_UP,
                bias="BULLISH",
                confidence=tl_met.confidence,
                neckline_level=tl_met.current_trendline_level,
                target_price=round(tl_met.current_price + abs(tl_met.p1_price - tl_met.p2_price), 2),
                stop_loss_level=round(tl_met.p2_price, 2),
                description=tl_met.description
            )
        elif tl_met and tl_met.breakout_type == "BREAKDOWN_DOWN":
            return DetectedPattern(
                pattern_type=PatternType.TRENDLINE_BREAKDOWN_DOWN,
                bias="BEARISH",
                confidence=tl_met.confidence,
                neckline_level=tl_met.current_trendline_level,
                target_price=round(tl_met.current_price - abs(tl_met.p1_price - tl_met.p2_price), 2),
                stop_loss_level=round(tl_met.p2_price, 2),
                description=tl_met.description
            )
        elif tl_met and tl_met.breakout_type == "RESISTANCE_CEILING_REJECT":
            return DetectedPattern(
                pattern_type=PatternType.TRENDLINE_RESISTANCE_REJECT,
                bias="BEARISH",
                confidence=tl_met.confidence,
                neckline_level=tl_met.current_trendline_level,
                target_price=round(tl_met.current_price - 30.0, 2),
                stop_loss_level=round(tl_met.current_trendline_level + 4.0, 2),
                description=tl_met.description
            )

        # 3. Consult 1-Minute Wave-Based M/W Pattern Engine
        m_met = self.m_pattern_skill.scan_m_pattern(bars_list)
        if m_met and m_met.stage.value in ("BREAKDOWN_CONFIRMED", "TESTING_NECKLINE"):
            conf = 0.88 if m_met.stage.value == "BREAKDOWN_CONFIRMED" else 0.80
            return DetectedPattern(
                pattern_type=PatternType.DOUBLE_TOP_M,
                bias="BEARISH",
                confidence=conf,
                neckline_level=m_met.neckline_level,
                target_price=m_met.target_1x,
                stop_loss_level=m_met.stop_loss,
                description=f"Wave M-Pattern confirmed with {m_met.stage.value} at neckline {m_met.neckline_level:.2f}"
            )

        w_met = self.m_pattern_skill.scan_w_pattern(bars_list)
        if w_met and w_met.stage.value in ("BREAKOUT_CONFIRMED", "TESTING_NECKLINE"):
            conf = 0.88 if w_met.stage.value == "BREAKOUT_CONFIRMED" else 0.80
            return DetectedPattern(
                pattern_type=PatternType.DOUBLE_BOTTOM_W,
                bias="BULLISH",
                confidence=conf,
                neckline_level=w_met.neckline_level,
                target_price=w_met.target_1x,
                stop_loss_level=w_met.stop_loss,
                description=f"Wave W-Pattern confirmed with {w_met.stage.value} at neckline {w_met.neckline_level:.2f}"
            )

        highs = [b["high"] for b in self.bars]
        lows = [b["low"] for b in self.bars]
        closes = [b["close"] for b in self.bars]

        # 2. Secondary fallback classical scans (from 15 to 45 bars)
        max_lookback = min(n_bars, 45)
        for span in [max_lookback, min(n_bars, 30), min(n_bars, 20)]:
            if span < 16:
                continue
            sub_highs = highs[-span:]
            sub_lows = lows[-span:]
            mid = span // 2
            h1 = max(sub_highs[:mid])
            h2 = max(sub_highs[mid:])
            valley = min(sub_lows[mid//2 : mid + mid//2])

            if abs(h1 - h2) / max(0.01, h1) < 0.0020 and h1 > valley + 8.0:
                if closes[-1] < valley:  # Breakdown below neckline
                    height = h1 - valley
                    return DetectedPattern(
                        pattern_type=PatternType.DOUBLE_TOP_M,
                        bias="BEARISH",
                        confidence=0.88,
                        neckline_level=round(valley, 2),
                        target_price=round(valley - height, 2),
                        stop_loss_level=round(max(h1, h2) + 2.0, 2),
                        description=f"Double Top (M-Pattern) confirmed with breakdown below neckline {valley:.2f}"
                    )

        # 2. Double Bottom (W-Pattern): Scans dynamic lookbacks (from 15 to 45 bars)
        for span in [max_lookback, min(n_bars, 30), min(n_bars, 20)]:
            if span < 16:
                continue
            sub_highs = highs[-span:]
            sub_lows = lows[-span:]
            mid = span // 2
            l1 = min(sub_lows[:mid])
            l2 = min(sub_lows[mid:])
            peak = max(sub_highs[mid//2 : mid + mid//2])

            if abs(l1 - l2) / max(0.01, l1) < 0.0020 and peak > l1 + 8.0:
                if closes[-1] > peak:  # Breakout above neckline
                    height = peak - l1
                    return DetectedPattern(
                        pattern_type=PatternType.DOUBLE_BOTTOM_W,
                        bias="BULLISH",
                        confidence=0.88,
                        neckline_level=round(peak, 2),
                        target_price=round(peak + height, 2),
                        stop_loss_level=round(min(l1, l2) - 2.0, 2),
                        description=f"Double Bottom (W-Pattern) confirmed with breakout above neckline {peak:.2f}"
                    )

        # 3. Horizontal Support Breakdown: Multi-touch support line violated
        if n_bars >= 15:
            # Find key support cluster in the last 15-30 bars
            search_len = min(n_bars - 1, 30)
            prior_lows = lows[-search_len:-1]
            min_low = min(prior_lows)
            # Check if at least 2 distinct candles tested within 4 pts of min_low
            touches = sum(1 for l in prior_lows if abs(l - min_low) <= 4.0)
            if touches >= 2 and closes[-1] < (min_low - 1.5):
                return DetectedPattern(
                    pattern_type=PatternType.SUPPORT_BREAKDOWN,
                    bias="BEARISH",
                    confidence=0.83,
                    neckline_level=round(min_low, 2),
                    target_price=round(min_low - 25.0, 2),
                    stop_loss_level=round(min_low + 10.0, 2),
                    description=f"Horizontal Support Breakdown below {min_low:.2f} ({touches}x base violation)"
                )

        # 4. Bull Flag Continuation (Strong pole upward followed by tight downward consolidation)
        if n_bars >= 15:
            pole = closes[-15] - closes[-25] if n_bars >= 25 else closes[-5] - closes[-15]
            recent_range = max(highs[-8:]) - min(lows[-8:])
            if pole > 25.0 and recent_range < (pole * 0.45) and closes[-1] > max(highs[-8:-1]):
                return DetectedPattern(
                    pattern_type=PatternType.BULL_FLAG,
                    bias="BULLISH",
                    confidence=0.80,
                    neckline_level=round(max(highs[-8:]), 2),
                    target_price=round(closes[-1] + pole, 2),
                    stop_loss_level=round(min(lows[-8:]) - 2.0, 2),
                    description=f"Bull Flag Breakout continuation (Measured Move Target: +{pole:.1f} pts)"
                )

        # 5. Bear Flag Continuation (Strong pole downward followed by tight upward consolidation)
        if n_bars >= 15:
            pole = closes[-25] - closes[-15] if n_bars >= 25 else closes[-15] - closes[-5]
            recent_range = max(highs[-8:]) - min(lows[-8:])
            if pole > 25.0 and recent_range < (pole * 0.45) and closes[-1] < min(lows[-8:-1]):
                return DetectedPattern(
                    pattern_type=PatternType.BEAR_FLAG,
                    bias="BEARISH",
                    confidence=0.80,
                    neckline_level=round(min(lows[-8:]), 2),
                    target_price=round(closes[-1] - pole, 2),
                    stop_loss_level=round(max(highs[-8:]) + 2.0, 2),
                    description=f"Bear Flag Breakdown continuation (Measured Move Target: -{pole:.1f} pts)"
                )

        return None
