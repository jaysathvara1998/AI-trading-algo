"""
Unit Tests for M-Pattern (Double Top Reversal & Neckline Breakdown) AI Trading Skill.
Tests with real market data matching the user's 5-minute NIFTY 50 chart:
- Peak 1: 23,397.60
- Valley / Neckline: 23,383.55
- Peak 2: 23,396.00
- Breakdown: 23,376.00 (< 23,383.55) -> Trigger BUY PUT
"""

import pytest
import pandas as pd
from algo_vpin_v2.skills.m_pattern_skill import MPatternSkill, MPatternStage


def test_m_pattern_detection_and_breakdown():
    skill = MPatternSkill()

    # 5-minute candles matching user's chart
    bars = [
        # Base rally
        {"open": 23360.0, "high": 23375.0, "low": 23358.0, "close": 23372.0, "volume": 100000},
        {"open": 23372.0, "high": 23390.0, "low": 23370.0, "close": 23388.0, "volume": 150000},
        # Peak 1 (11:15 candle)
        {"open": 23388.0, "high": 23397.60, "low": 23380.0, "close": 23394.0, "volume": 200000},
        # Valley pullback (11:20 candle) -> Neckline @ 23,383.55
        {"open": 23394.0, "high": 23395.0, "low": 23383.55, "close": 23385.0, "volume": 120000},
        # Second push up to Peak 2 (11:25 candle) -> Peak 2 @ 23,396.00
        {"open": 23385.0, "high": 23396.00, "low": 23384.0, "close": 23395.0, "volume": 180000},
        # Rejection candle
        {"open": 23395.0, "high": 23396.00, "low": 23385.0, "close": 23386.0, "volume": 140000},
        # Decisive Breakdown candle (11:30 candle) -> Cuts through 23,383.55 neckline
        {"open": 23386.0, "high": 23387.0, "low": 23374.0, "close": 23376.0, "volume": 280000}
    ]

    metrics = skill.scan_m_pattern(bars)
    assert metrics is not None
    assert metrics.peak1_price == 23397.60
    assert metrics.valley_price == 23383.55
    assert metrics.peak2_price == 23396.00
    assert metrics.neckline_level == 23383.55
    assert metrics.stage == MPatternStage.BREAKDOWN_CONFIRMED

    # Expected measured move: height = 23397.60 - 23383.55 = 14.05
    # TP1 = 23383.55 - 14.05 = 23369.50
    assert abs(metrics.pattern_height - 14.05) < 0.05
    assert abs(metrics.target_1x - 23369.50) < 0.05
    assert metrics.target_1_618x < metrics.target_1x

    # Evaluate skill output
    res = skill.evaluate(bars)
    assert res.is_favorable is True
    assert res.signal == "BUY_PUT"
    assert res.confidence >= 0.85
    assert "M-Pattern (Double Top) confirmed" in res.reason


def test_m_pattern_forming_testing_neckline():
    skill = MPatternSkill()

    bars = [
        {"open": 23360.0, "high": 23375.0, "low": 23358.0, "close": 23372.0},
        {"open": 23372.0, "high": 23390.0, "low": 23370.0, "close": 23388.0},
        # Peak 1
        {"open": 23388.0, "high": 23400.0, "low": 23380.0, "close": 23395.0},
        # Valley @ 23385.0
        {"open": 23395.0, "high": 23395.0, "low": 23385.0, "close": 23387.0},
        # Peak 2 @ 23399.0
        {"open": 23387.0, "high": 23399.0, "low": 23386.0, "close": 23396.0},
        # Currently re-testing neckline at 23385.0 without breaking yet
        {"open": 23396.0, "high": 23396.0, "low": 23384.5, "close": 23385.5}
    ]

    metrics = skill.scan_m_pattern(bars)
    assert metrics is not None
    assert metrics.stage == MPatternStage.TESTING_NECKLINE

    res = skill.evaluate(bars)
    assert res.is_favorable is False
    assert res.signal == "ALERT_M_NECKLINE_TEST"


def test_m_pattern_rejected_when_peaks_diverge_too_much():
    skill = MPatternSkill(max_peak_diff_pct=0.0020)

    bars = [
        {"open": 23360.0, "high": 23375.0, "low": 23358.0, "close": 23372.0},
        # Peak 1 at 23380
        {"open": 23372.0, "high": 23380.0, "low": 23370.0, "close": 23378.0},
        # Valley at 23350
        {"open": 23378.0, "high": 23378.0, "low": 23350.0, "close": 23355.0},
        # Peak 2 at 23450 (70 pts difference! Not an M pattern)
        {"open": 23355.0, "high": 23450.0, "low": 23355.0, "close": 23440.0},
        {"open": 23440.0, "high": 23442.0, "low": 23340.0, "close": 23345.0}
    ]

    metrics = skill.scan_m_pattern(bars)
    assert metrics is None

    res = skill.evaluate(bars)
    assert res.is_favorable is False
    assert res.signal == "NEUTRAL"


def test_put_option_chart_m_breakdown_and_w_breakout():
    """
    Directly tests the user's uploaded NIFTY 22 SEP 23400 PUT 1-minute chart:
    1. White arrows: M-Pattern (UP -> MID DOWN -> MID UP -> DOWN breakdown below 55.50)
    2. Yellow arrows: W-Pattern (DOWN -> MID UP -> MID DOWN -> UP breakout above 55.50)
    """
    skill = MPatternSkill(min_pattern_depth_pts=3.0, max_peak_diff_pct=0.08)

    # 1. White arrows: 1-minute PUT chart M-pattern breakdown
    # Leg 1: UP from 48.0 to Peak 1 @ 68.50 (spanning multiple 1m bars)
    m_bars = [
        {"open": 48.0, "high": 52.0, "low": 47.5, "close": 51.5},
        {"open": 51.5, "high": 60.0, "low": 51.0, "close": 59.0},
        {"open": 59.0, "high": 68.50, "low": 58.0, "close": 67.0},  # Peak 1 = 68.50
        # Leg 2: MID DOWN to Valley @ 55.50
        {"open": 67.0, "high": 67.0, "low": 60.0, "close": 61.0},
        {"open": 61.0, "high": 61.5, "low": 55.50, "close": 56.0},  # Valley = 55.50
        # Leg 3: MID UP to Peak 2 @ 66.00
        {"open": 56.0, "high": 63.0, "low": 55.8, "close": 62.0},
        {"open": 62.0, "high": 66.00, "low": 61.0, "close": 65.0},  # Peak 2 = 66.00
        # Leg 4: DOWN breaking through 55.50 neckline down to 48.00
        {"open": 65.0, "high": 65.0, "low": 57.0, "close": 58.0},
        {"open": 58.0, "high": 58.0, "low": 48.0, "close": 49.0, "volume": 300000}
    ]

    m_metrics = skill.scan_m_pattern(m_bars)
    assert m_metrics is not None
    assert m_metrics.stage == MPatternStage.BREAKDOWN_CONFIRMED
    assert abs(m_metrics.neckline_level - 55.50) < 1.0
    assert m_metrics.target_1x <= 45.0  # 55.5 - (68.5 - 55.5) = 42.5

    # 2. Yellow arrows: W-Pattern reversal from 46.15
    # Leg 1: DOWN to Valley 1 @ 46.00
    # Leg 2: MID UP to Mid Peak / Neckline @ 55.50
    # Leg 3: MID DOWN to Valley 2 @ 45.00
    # Leg 4: UP breaking through 55.50 neckline to 65.00+
    w_bars = [
        {"open": 54.0, "high": 54.0, "low": 46.00, "close": 46.50},  # Valley 1 = 46.00
        {"open": 46.5, "high": 52.0, "low": 46.50, "close": 51.00},
        {"open": 51.0, "high": 55.50, "low": 50.00, "close": 55.00},  # Mid Peak = 55.50
        {"open": 55.0, "high": 55.0, "low": 45.00, "close": 45.50},  # Valley 2 = 45.00
        {"open": 45.5, "high": 54.0, "low": 45.00, "close": 53.00},
        {"open": 53.0, "high": 66.0, "low": 53.00, "close": 65.00, "volume": 500000} # Breakout > 55.50!
    ]

    w_metrics = skill.scan_w_pattern(w_bars)
    assert w_metrics is not None
    assert w_metrics.stage == MPatternStage.BREAKOUT_CONFIRMED
    assert abs(w_metrics.neckline_level - 55.50) < 1.0
    assert w_metrics.target_1x >= 65.0  # 55.5 + (55.5 - 45.0) = 66.0

