"""
Comprehensive Unit Tests for Autonomous Self-Learning & Confirmation Watcher Architecture
Validates:
1. PatternConfirmationWatcher (Patience loop: FORMING -> CONFIRMATION_WATCHING -> CONFIRMED / INVALIDATED)
2. Rejection Wick / Fakeout Filter
3. Post-Trade Cooldown Guard (Blocks impulse top re-entries)
4. AutoTuningEngine (Dynamic parameter adjustments & adaptive_config.json persistence)
5. DynamicSkillRegistry (Zero-downtime hot-reloading of modular trading skills)
6. SkillGeneratorAgent (Sandbox syntax validation and deployment)
"""

import unittest
from pathlib import Path
from algo_vpin_v2.skills.pattern_confirmation_watcher import PatternConfirmationWatcher, PatternState
from algo_vpin_v2.skills.dynamic_skill_registry import DynamicSkillRegistry
from algo_vpin_v2.auto_tuner import AutoTuningEngine, AdaptiveConfigState
from algo_vpin_v2.agents.skill_generator_agent import SkillGeneratorAgent


class TestAutonomousSelfLearningArchitecture(unittest.TestCase):

    def setUp(self):
        self.watcher = PatternConfirmationWatcher()
        self.auto_tuner = AutoTuningEngine()
        self.registry = DynamicSkillRegistry()
        self.generator = SkillGeneratorAgent()

    def test_01_pattern_confirmation_lifecycle_success(self):
        """Test bullish breakout confirmation upon strong candle close above neckline"""
        # Register a forming W-Pattern breakout at 23400
        cand = self.watcher.register_forming_pattern(
            pattern_type="DOUBLE_BOTTOM_W",
            bias="BULLISH",
            trigger_level=23400.0,
            invalidation_level=23350.0,
            confidence=0.85,
            description="Double Bottom W Neckline"
        )
        self.assertEqual(cand.status.value, PatternState.CONFIRMATION_WATCHING.value)

        # Bar 1: Price closes below trigger (23395) -> Still watching
        state, msg = self.watcher.on_candle_close(
            open_p=23380.0, high_p=23398.0, low_p=23375.0, close_p=23395.0, volume=15000.0, current_bar=1
        )
        self.assertEqual(state.value, PatternState.CONFIRMATION_WATCHING.value)

        # Bar 2: Price closes strongly above trigger (23410) -> CONFIRMED!
        state, msg = self.watcher.on_candle_close(
            open_p=23395.0, high_p=23415.0, low_p=23390.0, close_p=23410.0, volume=35000.0, current_bar=2
        )
        self.assertEqual(state.value, PatternState.CONFIRMED.value)
        res = self.watcher.evaluate()
        self.assertTrue(res.is_favorable)
        self.assertEqual(res.signal, "BUY_CALL")

    def test_02_fakeout_rejection_wick_invalidation(self):
        """Test that a breakout with a long rejection wick is caught as a fakeout and invalidated"""
        self.watcher.register_forming_pattern(
            pattern_type="TRENDLINE_BREAKOUT_UP",
            bias="BULLISH",
            trigger_level=23400.0,
            invalidation_level=23360.0
        )
        # Candle spikes to 23450 but rejects back down, closing at 23402 with huge 48 pt upper wick
        state, msg = self.watcher.on_candle_close(
            open_p=23390.0, high_p=23450.0, low_p=23385.0, close_p=23402.0, volume=20000.0, current_bar=1
        )
        self.assertEqual(state.value, PatternState.INVALIDATED.value)
        self.assertIn("Fakeout", msg)

    def test_03_post_trade_cooldown_guard(self):
        """Test that post-trade cooldown blocks re-entries for N bars after taking profit"""
        self.watcher.set_post_trade_cooldown(symbol="NIFTY 23600 PUT", side="SHORT", bars=3, current_bar=10)
        self.assertTrue(self.watcher.is_in_cooldown(current_bar=10))
        self.assertTrue(self.watcher.is_in_cooldown(current_bar=11))
        self.assertTrue(self.watcher.is_in_cooldown(current_bar=12))
        # After 3 bars (bar 13) -> Cooldown expires
        self.assertFalse(self.watcher.is_in_cooldown(current_bar=13))

    def test_04_auto_tuner_persistence(self):
        """Test real-time parameter adaptation and JSON persistence"""
        initial_conf = self.auto_tuner.state.xgb_min_prob_threshold
        # Simulate a stop-loss trade
        self.auto_tuner.on_trade_completed(pnl_inr=-800.0, pnl_pct=-5.2, exit_reason="STOP_LOSS_HIT")
        self.assertGreaterEqual(self.auto_tuner.state.xgb_min_prob_threshold, initial_conf)
        self.assertTrue(self.auto_tuner.config_path.exists())

    def test_05_dynamic_skill_registry_discovery(self):
        """Test that DynamicSkillRegistry loads all active trading skills in algo_vpin_v2/skills/"""
        count = self.registry.discover_and_load_skills()
        self.assertGreaterEqual(count, 5)
        self.assertIsNotNone(self.registry.get_skill("PriceActionSkill"))
        self.assertIsNotNone(self.registry.get_skill("MPatternSkill"))
        self.assertIsNotNone(self.registry.get_skill("DailyReflectionSkill"))

    def test_06_skill_generator_agent_sandbox_compilation(self):
        """Test that SkillGeneratorAgent generates valid python code that passes sandbox compilation"""
        valid, msg = self.generator._validate_skill_code_sandbox(
            filename="test_sample_skill.py",
            code="""
from algo_vpin_v2.skills.base_skill import BaseTradingSkill, SkillResult
class TestSampleSkill(BaseTradingSkill):
    def __init__(self):
        super().__init__(name="TestSampleSkill")
    def evaluate(self, **kwargs):
        return SkillResult(is_favorable=True, confidence=0.8)
"""
        )
        self.assertTrue(valid)


if __name__ == "__main__":
    unittest.main()
