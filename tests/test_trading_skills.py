"""
Comprehensive Unit Tests for Chinmay Option Scalping Trading Skills
Validates:
1. PriceActionSkill (Hammers, Shooting Stars, Engulfing, Breakouts, S/R Retests)
2. StrikeSelectorSkill (ATM / ITM Delta sweet-spot, Strike Steps)
3. RiskRewardSkill (Minimum 1:1.5 R:R enforcement, Spot-anchored SL)
4. PositionGuardianSkill (Spot SL Shield, Breakeven at 1R, Trailing at 1.5R, TP, Disaster SL)
5. EnsembleBrain Integration with Modular Skills
"""

import pytest
from algo_vpin_v2.skills import (
    PriceActionSkill,
    StrikeSelectorSkill,
    RiskRewardSkill,
    PositionGuardianSkill,
    SkillResult
)
from algo_vpin_v2.garch_engine import DirectionalSignal
from algo_vpin_v2.ensemble_brain import EnsembleBrain, EnsembleDecision
from algo_vpin_v2.macro_features import MacroFeatureState


# ============================================================================
# 1. PRICE ACTION SKILL TESTS
# ============================================================================
def test_price_action_hammer_bullish():
    skill = PriceActionSkill()
    # Hammer: open 24000, close 24010, high 24012, low 23970
    # body = 10, lower_wick = 30 (3x body), upper_wick = 2
    res = skill.detect_pin_bar(open_p=24000.0, high_p=24012.0, low_p=23970.0, close_p=24010.0)
    assert res == "HAMMER_BULLISH"


def test_price_action_shooting_star_bearish():
    skill = PriceActionSkill()
    # Shooting Star: open 24010, close 24000, high 24050, low 23998
    # body = 10, upper_wick = 40 (4x body), lower_wick = 2
    res = skill.detect_pin_bar(open_p=24010.0, high_p=24050.0, low_p=23998.0, close_p=24000.0)
    assert res == "SHOOTING_STAR_BEARISH"


def test_price_action_bullish_engulfing():
    skill = PriceActionSkill()
    # Prev: open 24020, close 24000 (red 20 pts)
    # Curr: open 23998, close 24030 (green 32 pts enveloping prev)
    res = skill.detect_engulfing(prev_o=24020.0, prev_c=24000.0, curr_o=23998.0, curr_c=24030.0)
    assert res == "BULLISH_ENGULFING"


def test_price_action_eval_veto_on_adverse_pattern():
    skill = PriceActionSkill()
    # If intended direction is BUY, but candle prints a severe Shooting Star (bearish rejection)
    eval_res = skill.evaluate(
        open_p=24010.0, high_p=24050.0, low_p=23998.0, close_p=24000.0,
        intended_direction="BUY"
    )
    assert not eval_res.is_favorable
    assert "Bearish price action contradicts BUY setup" in eval_res.reason


# ============================================================================
# 2. STRIKE SELECTOR SKILL TESTS
# ============================================================================
def test_strike_selector_atm_and_itm():
    skill = StrikeSelectorSkill()
    # NIFTY spot = 24340 -> ATM = 24350
    atm_nifty = skill.resolve_recommended_strike("NIFTY", 24340.0, action="BUY", prefer_itm=False)
    assert atm_nifty == 24350.0

    # NIFTY BUY ITM -> 24350 - 50 = 24300
    itm_call = skill.resolve_recommended_strike("NIFTY", 24340.0, action="BUY", prefer_itm=True)
    assert itm_call == 24300.0

    # NIFTY SELL (PUT) ITM -> 24350 + 50 = 24400
    itm_put = skill.resolve_recommended_strike("NIFTY", 24340.0, action="SELL", prefer_itm=True)
    assert itm_put == 24400.0


def test_strike_selector_evaluates_delta():
    skill = StrikeSelectorSkill()
    eval_res = skill.evaluate(underlying="NIFTY", spot_price=24340.0, action="BUY")
    assert eval_res.is_favorable
    assert "24350 CE" in eval_res.signal
    assert eval_res.metadata["approx_delta"] >= 0.45


# ============================================================================
# 3. RISK-REWARD SKILL TESTS
# ============================================================================
def test_risk_reward_minimum_expectancy():
    skill = RiskRewardSkill(min_rr_ratio=1.50)
    
    # Good setup: Spot 24300, nearest resistance 24345 (+45 pts), candle low 24285 (-15 pts)
    # R:R = 45 / 17 ~ 2.65 -> Favorable!
    res_good = skill.evaluate(
        spot_entry=24300.0,
        action="BUY",
        option_entry=110.0,
        candle_low=24285.0,
        nearest_resistance=24345.0
    )
    assert res_good.is_favorable
    assert res_good.metadata["plan"].risk_reward_ratio >= 1.50

    # Bad setup: Spot 24300, resistance right overhead at 24310 (+10 pts), SL 24282 (-18 pts)
    # R:R = 10 / 18 ~ 0.55 -> Must be VETOED!
    res_bad = skill.evaluate(
        spot_entry=24300.0,
        action="BUY",
        option_entry=110.0,
        nearest_resistance=24310.0
    )
    assert not res_bad.is_favorable
    assert "Unfavorable R:R" in res_bad.reason


# ============================================================================
# 4. POSITION GUARDIAN SKILL TESTS
# ============================================================================
def test_guardian_spot_sl_shield_holds_on_wick():
    guardian = PositionGuardianSkill()
    # Entry at 100.0, SL at 90.0, Spot entry at 24300.0, Spot SL at 24282.0
    # Option price drops to 89.0 (below SL!), but Spot is still at 24295.0 (holding structure!)
    res = guardian.evaluate(
        symbol="NIFTY 24300 CE",
        is_call=True,
        entry_price=100.0,
        current_option_price=89.0,
        current_spot_price=24295.0,  # Spot intact!
        spot_sl=24282.0,
        current_sl=90.0,
        target_tp=120.0,
        initial_risk_pts=10.0,
        highest_price_seen=100.0
    )
    # Must NOT exit; shield is active
    assert res.metadata["action"].should_exit is False
    assert res.metadata["action"].is_shielded is True


def test_guardian_breakeven_jump():
    guardian = PositionGuardianSkill()
    # Entry 100.0, initial risk 10.0 pts.
    # Option runs to 111.0 (gain >= 1.0R)
    res = guardian.evaluate(
        symbol="NIFTY 24300 CE",
        is_call=True,
        entry_price=100.0,
        current_option_price=111.0,
        current_spot_price=24320.0,
        spot_sl=24282.0,
        current_sl=90.0,
        target_tp=120.0,
        initial_risk_pts=10.0,
        highest_price_seen=111.0
    )
    # SL stepped up to Breakeven (+1.5 pts = 101.5)
    assert res.metadata["action"].updated_sl == 101.5
    assert res.metadata["action"].trailing_tier == "Breakeven (+1.5 pts)"


def test_guardian_take_profit_hit():
    guardian = PositionGuardianSkill()
    res = guardian.evaluate(
        symbol="NIFTY 24300 CE",
        is_call=True,
        entry_price=100.0,
        current_option_price=121.5,
        current_spot_price=24345.0,
        spot_sl=24282.0,
        current_sl=101.5,
        target_tp=120.0,  # Hit!
        initial_risk_pts=10.0,
        highest_price_seen=121.5
    )
    assert res.metadata["action"].should_exit is True
    assert "TAKE_PROFIT_TRIGGERED" in res.metadata["action"].exit_reason


# ============================================================================
# 5. ENSEMBLE BRAIN INTEGRATION TEST
# ============================================================================
def test_ensemble_brain_consults_trading_skills():
    brain = EnsembleBrain()
    macro = MacroFeatureState(
        pdh=24400.0, pdl=24200.0, week_high=24500.0, week_low=24100.0,
        day_open=24300.0, ema_15m_fast=24310.0, ema_15m_slow=24300.0,
        dist_pdh_pct=0.005, dist_pdl_pct=0.005,
        week_range_position=0.50, trend_15m_bias=0
    )

    # When brain evaluates a BUY signal with a bearish shooting star candle
    dec = brain.evaluate(
        garch_signal=DirectionalSignal.BUY,
        price_delta=10.0,
        rolling_vol=0.01,
        garch_forecast=0.01,
        vpin=0.20,
        macro_state=macro,
        open_p=24310.0,
        high_p=24350.0,  # Long upper wick
        low_p=24298.0,
        close_p=24300.0  # Shooting star against BUY
    )
    # Brain must veto the trade because PriceActionSkill rejected it!
    assert dec.is_vetoed is True
    assert "SKILL VETO" in dec.reason
    assert dec.price_action_result is not None


def test_w_pattern_breakout_triggers_buy():
    """Test that a confirmed DOUBLE_BOTTOM_W overrides GARCH HOLD and triggers BUY CALL"""
    from algo_vpin_v2.pattern_engine import DetectedPattern, PatternType
    brain = EnsembleBrain()
    brain.config.xgb_min_prob_threshold = 0.99

    macro = MacroFeatureState(
        pdh=23500.0, pdl=23400.0, week_high=23600.0, week_low=23300.0,
        day_open=23420.0, ema_15m_fast=23440.0, ema_15m_slow=23430.0,
        dist_pdh_pct=0.01, dist_pdl_pct=0.01,
        week_range_position=0.50, trend_15m_bias=0
    )

    pattern = DetectedPattern(
        pattern_type=PatternType.DOUBLE_BOTTOM_W,
        bias="BULLISH",
        confidence=0.85,
        neckline_level=23456.0,
        target_price=23475.0,
        stop_loss_level=23430.0,
        description="Double Bottom (W-Pattern) confirmed with breakout above neckline 23456.00"
    )

    dec = brain.evaluate(
        garch_signal=DirectionalSignal.HOLD,  # GARCH is neutral
        price_delta=5.0,
        rolling_vol=0.01,
        garch_forecast=0.0,
        vpin=0.20,
        macro_state=macro,
        open_p=23450.0,
        high_p=23458.0,
        low_p=23449.0,
        close_p=23457.0,  # Breakout candle above neckline
        pattern_st=pattern
    )

    assert dec.final_action == DirectionalSignal.BUY
    assert dec.is_vetoed is False
    assert "Pattern Trigger" in dec.reason


def test_w_pattern_vetoes_lagging_sell():
    """Verify that a Bullish W-pattern HARD VETOES any lagging SELL signal from ML models"""
    from algo_vpin_v2.pattern_engine import DetectedPattern, PatternType
    brain = EnsembleBrain()
    brain.config.xgb_min_prob_threshold = 0.99

    macro = MacroFeatureState(
        pdh=23500.0, pdl=23400.0, week_high=23600.0, week_low=23300.0,
        day_open=23420.0, ema_15m_fast=23440.0, ema_15m_slow=23430.0,
        dist_pdh_pct=0.01, dist_pdl_pct=0.01,
        week_range_position=0.50, trend_15m_bias=0
    )

    w_pattern = DetectedPattern(
        pattern_type=PatternType.DOUBLE_BOTTOM_W,
        bias="BULLISH",
        confidence=0.88,
        neckline_level=23345.0,
        target_price=23370.0,
        stop_loss_level=23320.0,
        description="Double Bottom (W-Pattern) confirmed with breakout above neckline 23345.00"
    )

    # Statistical model (GARCH) says SELL, but W-Pattern is BULLISH
    dec = brain.evaluate(
        garch_signal=DirectionalSignal.SELL,  # Lagging sell
        price_delta=-2.0,
        rolling_vol=0.01,
        garch_forecast=-0.001,
        vpin=0.20,
        macro_state=macro,
        open_p=23340.0,
        high_p=23348.0,
        low_p=23338.0,
        close_p=23346.0,
        pattern_st=w_pattern
    )

    # SELL MUST BE VETOED and converted to BUY CALL
    assert dec.final_action == DirectionalSignal.BUY
    assert dec.is_vetoed is False


def test_descending_trendline_breakout():
    """Verify that a descending trendline breakout is detected and triggers BUY_CALL"""
    from algo_vpin_v2.skills.m_pattern_skill import MPatternSkill
    skill = MPatternSkill()

    # Create lower highs: Peak 1 @ 23414, Valley @ 23350, Peak 2 @ 23365, Valley @ 23320, then Breakout @ 23350
    bars = []
    # Drop from 23414
    for p in [23414, 23400, 23380, 23360, 23350]:
        bars.append({"open": p, "high": p + 2, "low": p - 2, "close": p})
    # Bounce to 23365
    for p in [23352, 23358, 23365]:
        bars.append({"open": p, "high": p + 2, "low": p - 2, "close": p})
    # Drop to 23320
    for p in [23355, 23340, 23328, 23320]:
        bars.append({"open": p, "high": p + 2, "low": p - 2, "close": p})
    # Breakout surge to 23355 (breaking descending trendline)
    for p in [23325, 23335, 23345, 23355]:
        bars.append({"open": p, "high": p + 2, "low": p - 2, "close": p})

    res = skill.evaluate(bars, current_price=23355.0)
    assert res.is_favorable is True
    assert res.signal == "BUY_CALL"

