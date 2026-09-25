import unittest
from datetime import datetime
from pathlib import Path
import tempfile
import pytz

from algo_vpin_v2.shadow_learner import VirtualShadowLearner, VirtualShadowPosition
from algo_vpin_v2.garch_engine import DirectionalSignal


class TestVirtualShadowLearner(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.learner = VirtualShadowLearner(data_dir=Path(self.temp_dir.name))

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_start_and_exit_shadow_trade_tp(self):
        self.assertFalse(self.learner.is_tracking())
        self.learner.start_shadow_trade(
            signal=DirectionalSignal.BUY,
            symbol="SENSEX 24 SEP 74000 CALL",
            entry_price=300.0,
            entry_spot=74100.0,
            stop_loss=268.0,
            take_profit=350.0
        )
        self.assertTrue(self.learner.is_tracking())

        # Update price to hit TP
        res = self.learner.update_shadow_position(current_spot=74200.0, option_ltp=355.0)
        self.assertIsNotNone(res)
        self.assertFalse(self.learner.is_tracking())
        self.assertIn("SHADOW_TAKE_PROFIT_HIT", res["exit_reason"])
        self.assertGreater(res["pnl_inr"], 0)

    def test_shadow_trailing_stop(self):
        self.learner.start_shadow_trade(
            signal=DirectionalSignal.BUY,
            symbol="SENSEX 24 SEP 74000 CALL",
            entry_price=300.0,
            entry_spot=74100.0,
            stop_loss=268.0,
            take_profit=380.0
        )
        # Price surges to 335 (+35 pts profit -> activates breakeven floor 315 & trail 315)
        self.learner.update_shadow_position(current_spot=74150.0, option_ltp=335.0)
        pos = self.learner.active_shadow_position
        self.assertTrue(pos.is_runner_active)
        self.assertGreaterEqual(pos.current_trail_sl, 315.0)

        # Price drops below trail SL
        res = self.learner.update_shadow_position(current_spot=74120.0, option_ltp=310.0)
        self.assertIsNotNone(res)
        self.assertIn("SHADOW_TRAILING_SL_HIT", res["exit_reason"])
        self.assertGreaterEqual(res["pnl_inr"], 0)


if __name__ == "__main__":
    unittest.main()
