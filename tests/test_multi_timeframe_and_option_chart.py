"""
Unit Tests for Multi-Timeframe Pattern Scanner & Option Chart AI Skill
Validates:
1. Resampling 1-minute data into 3m, 5m, 15m
2. Multi-Timeframe Confluence (Macro 5m M-pattern + Micro 1m breakdown trigger)
3. Direct Option Chart Analysis (NIFTY 23400 PUT W-pattern breakout)
4. Cross-Validation between Spot Index M-breakdown and Option PE W-breakout
"""

import pytest
import pandas as pd
import numpy as np

from algo_vpin_v2.skills.multi_timeframe_scanner import MultiTimeframeScanner, MultiTimeframeResult
from algo_vpin_v2.skills.option_chart_skill import OptionChartSkill, OptionChartAnalysis
from algo_vpin_v2.skills.m_pattern_skill import MPatternStage


def test_resample_ohlcv_chunks():
    scanner = MultiTimeframeScanner()

    # Generate 15 one-minute bars
    bars_1m = []
    base = 23350.0
    for i in range(15):
        bars_1m.append({
            "open": base + i,
            "high": base + i + 2,
            "low": base + i - 1,
            "close": base + i + 1,
            "volume": 1000
        })
    df_1m = pd.DataFrame(bars_1m)

    df_5m = scanner.resample_ohlcv(df_1m, "5m")
    assert len(df_5m) == 3  # 15 1m bars resample into 3 5m bars
    assert df_5m["open"].iloc[0] == bars_1m[0]["open"]
    assert df_5m["close"].iloc[0] == bars_1m[4]["close"]
    assert df_5m["volume"].iloc[0] == 5000


def test_multi_timeframe_m_pattern_confluence():
    scanner = MultiTimeframeScanner(timeframes=["1m", "5m"])

    # Build 1-minute candles representing the user's NIFTY chart M-pattern
    # Leg 1: UP to Peak 1 @ 23397.60
    # Leg 2: MID DOWN to Valley @ 23383.55
    # Leg 3: MID UP to Peak 2 @ 23396.00
    # Leg 4: DOWN breakdown to 23376.00
    bars = []
    # Up to 23397.6
    bars.extend([
        {"open": 23365.0, "high": 23378.0, "low": 23363.0, "close": 23376.0, "volume": 10000},
        {"open": 23376.0, "high": 23388.0, "low": 23374.0, "close": 23385.0, "volume": 12000},
        {"open": 23385.0, "high": 23397.60, "low": 23382.0, "close": 23395.0, "volume": 18000}, # Peak 1
    ])
    # Down to 23383.55
    bars.extend([
        {"open": 23395.0, "high": 23395.0, "low": 23388.0, "close": 23389.0, "volume": 15000},
        {"open": 23389.0, "high": 23390.0, "low": 23383.55, "close": 23385.0, "volume": 14000}, # Valley
    ])
    # Up to 23396.00
    bars.extend([
        {"open": 23385.0, "high": 23392.0, "low": 23384.0, "close": 23390.0, "volume": 16000},
        {"open": 23390.0, "high": 23396.00, "low": 23389.0, "close": 23394.0, "volume": 20000}, # Peak 2
    ])
    # Down breaking 23383.55
    bars.extend([
        {"open": 23394.0, "high": 23394.0, "low": 23384.0, "close": 23384.5, "volume": 22000},
        {"open": 23384.5, "high": 23385.0, "low": 23374.0, "close": 23376.0, "volume": 35000}  # Breakdown
    ])
    df_1m = pd.DataFrame(bars)

    result = scanner.scan_confluence(df_1m)
    assert result.direction == "BUY_PUT"
    assert abs(result.neckline_level - 23383.55) < 1.0
    assert result.confluence_score >= 0.75


def test_option_chart_pe_w_breakout():
    opt_skill = OptionChartSkill()

    # User's uploaded 23400 PUT 1-minute chart:
    # Reversal W-pattern from 46.15:
    # Leg 1: DOWN to 46.00
    # Leg 2: MID UP to Mid Peak / Neckline @ 55.50
    # Leg 3: MID DOWN to 45.00
    # Leg 4: UP breakout above 55.50
    w_bars = [
        {"open": 54.0, "high": 54.0, "low": 46.00, "close": 46.50},
        {"open": 46.5, "high": 52.0, "low": 46.50, "close": 51.00},
        {"open": 51.0, "high": 55.50, "low": 50.00, "close": 55.00},  # Mid Neckline = 55.50
        {"open": 55.0, "high": 55.0, "low": 45.00, "close": 45.50},  # Valley 2 = 45.00
        {"open": 45.5, "high": 54.0, "low": 45.00, "close": 53.00},
        {"open": 53.0, "high": 66.0, "low": 53.00, "close": 65.00, "volume": 500000} # Breakout!
    ]

    analysis = opt_skill.analyze_option_chart(
        ohlcv=w_bars,
        symbol="NIFTY 22 SEP 23400 PUT",
        option_type="PE",
        strike=23400,
        current_premium=65.00
    )

    assert analysis.pattern_detected == "W_BREAKOUT"
    assert abs(analysis.neckline_premium - 55.50) < 1.0
    assert analysis.target_1x_premium >= 65.0

    # Cross-validation: Spot Index M-Breakdown + PE W-Breakout -> Max Conviction
    cross = opt_skill.cross_validate(spot_signal="BUY_PUT", option_analysis=analysis)
    assert cross.is_favorable is True
    assert cross.confidence >= 0.90
    assert "HIGH CONVICTION CONFLUENCE" in cross.reason


def test_anti_chase_exhaustion_guard():
    from algo_vpin_v2.mtf_analyst import MultiTimeframeAnalyst
    analyst = MultiTimeframeAnalyst()

    # Simulate spot dumping 40 points in 5 bars
    prices = [23400.0, 23390.0, 23380.0, 23370.0, 23360.0, 23355.0]
    for p in prices:
        analyst.update_bar(p)

    # Attempt to BUY PUT at the extreme bottom of the 45-point vertical dump
    is_ok, reason = analyst.validate_htf_entry_alignment("SELL", 23355.0)
    assert is_ok is False
    assert "5M_HTF_EXHAUSTION" in reason

