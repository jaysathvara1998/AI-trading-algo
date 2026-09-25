"""
Unit Test for Institutional Runner Mode (Trail Past Target)
Validates that when a winning trade hits Target 1:
1. It does NOT kill the trade immediately.
2. It elevates SL to guarantee minimum profit.
3. It dynamically ratchets the SL behind the highest peak price (e.g. 91.15 -> SL 87.65).
4. When price pulls back, it exits with extended runner profit instead of the conservative target!
"""

import pytest
from datetime import datetime
import pytz

from algo_vpin_v2.config import RiskConfig, ScalperConfig, TradeStrategyMode
from algo_vpin_v2.risk_manager import RiskManager, ActivePosition, PositionSide, ToxicityRegime


def test_runner_mode_trails_past_target_to_peak():
    tz = pytz.timezone("Asia/Kolkata")
    now = datetime(2026, 9, 22, 12, 55, 0, tzinfo=tz)

    scalper_cfg = ScalperConfig(
        trail_past_target=True,
        runner_trail_buffer_pts=3.5
    )
    rm = RiskManager(scalper_config=scalper_cfg)

    # Simulate the user's actual trade on 23400 PUT:
    # Entry @ 69.57, Initial SL @ 59.57, Target @ 84.50
    pos = ActivePosition(
        side=PositionSide.SHORT,
        entry_price=69.57,
        quantity=65,
        stop_loss=59.57,
        take_profit=84.50,
        entry_time=now,
        entry_vpin=0.18,
        entry_regime=ToxicityRegime.LOW_TOXICITY,
        symbol="NIFTY 22 SEP 23400 PUT",
        is_option=True,
        strategy_mode=TradeStrategyMode.SCALPER,
        initial_risk_pts=10.0
    )
    rm.current_position = pos

    # 1. Price hits Target 1 @ 85.00:
    # Instead of exiting, Runner Mode engages!
    exit_needed, reason = rm.check_exit_conditions(current_price=23330.0, option_premium=85.00, current_time=now)
    assert exit_needed is False
    assert pos.is_runner_active is True
    # Stop-Loss must be elevated to at least 85.00 - 3.5 = 81.50 (locking in profit!)
    assert pos.stop_loss >= 81.50
    assert pos.stop_loss > pos.entry_price

    # 2. Price surges further to 88.00:
    exit_needed, reason = rm.check_exit_conditions(current_price=23325.0, option_premium=88.00, current_time=now)
    assert exit_needed is False
    assert pos.stop_loss >= 84.50

    # 3. Price reaches day peak of 91.15 (from user's screenshot!):
    exit_needed, reason = rm.check_exit_conditions(current_price=23320.0, option_premium=91.15, current_time=now)
    assert exit_needed is False
    # SL ratcheted up to 91.15 - 3.5 = 87.65
    assert abs(pos.stop_loss - 87.65) < 0.05

    # 4. Price pulls back from 91.15 down to 87.00 (< 87.65):
    exit_needed, reason = rm.check_exit_conditions(current_price=23325.0, option_premium=87.00, current_time=now)
    assert exit_needed is True
    assert "RUNNER_TRAILING_HIT" in reason
    assert "Peak INR 91.15" in reason
    # Verified: captured ~18.08 pts of profit instead of exiting at 84.92 (+15 pts)!
