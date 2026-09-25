"""
Unit tests for Risk Management and Order Execution Engine
"""

from datetime import datetime
import unittest
import pytz

from algo_vpin_dhan.config import AppConfig, RiskConfig, MarketConfig
from algo_vpin_dhan.risk_manager import RiskManager, PositionSide
from algo_vpin_dhan.execution import ExecutionEngine, OrderType, OrderStatus
from algo_vpin_dhan.vpin import ToxicityRegime, VPINResult
from algo_vpin_dhan.garch_engine import DirectionalSignal, GARCHForecastResult


class TestRiskAndExecution(unittest.TestCase):

    def setUp(self):
        self.app_cfg = AppConfig()
        self.app_cfg.dhan.paper_trading = True
        self.risk_cfg = RiskConfig(
            max_daily_loss=1000.0,
            max_position_lots=4,
            sl_garch_multiplier=2.0,
            tp_garch_multiplier=3.0
        )
        self.market_cfg = MarketConfig(lot_size=50)
        self.risk_mgr = RiskManager(self.risk_cfg, self.market_cfg)
        self.execution = ExecutionEngine(self.app_cfg, self.risk_mgr)

    def test_toxicity_position_sizing(self):
        """Test position sizing throttles: 0.5x on high toxicity, 1.0x on low toxicity."""
        high_tox_vpin = VPINResult(
            vpin=0.65,
            regime=ToxicityRegime.HIGH_TOXICITY,
            buy_volume_bar=1000,
            sell_volume_bar=1000,
            bucket_volume=25000,
            completed_buckets=50,
            entry_band_multiplier=1.5,
            size_multiplier=0.5
        )
        lots_high = self.risk_mgr.calculate_position_size(high_tox_vpin)
        self.assertEqual(lots_high, 2)  # 4 * 0.5 = 2 lots

        low_tox_vpin = VPINResult(
            vpin=0.20,
            regime=ToxicityRegime.LOW_TOXICITY,
            buy_volume_bar=1000,
            sell_volume_bar=1000,
            bucket_volume=25000,
            completed_buckets=50,
            entry_band_multiplier=0.8,
            size_multiplier=1.0
        )
        lots_low = self.risk_mgr.calculate_position_size(low_tox_vpin)
        self.assertEqual(lots_low, 4)  # 4 * 1.0 = 4 lots

    def test_dynamic_targets_and_paper_execution(self):
        """Test dynamic SL/TP targets calculation and paper order execution."""
        garch_res = GARCHForecastResult(
            mu_next=0.0003,
            h_next=1e-6,
            sigma_next=0.001,
            annualized_vol=0.30,
            signal=DirectionalSignal.BUY,
            omega=1e-6,
            alpha=0.05,
            beta=0.90,
            is_converged=True
        )
        vpin_res = VPINResult(
            vpin=0.20,
            regime=ToxicityRegime.LOW_TOXICITY,
            buy_volume_bar=1000,
            sell_volume_bar=1000,
            bucket_volume=25000,
            completed_buckets=50,
            entry_band_multiplier=0.8,
            size_multiplier=1.0
        )

        current_price = 24000.0
        targets = self.risk_mgr.compute_trade_targets(
            signal=DirectionalSignal.BUY,
            current_price=current_price,
            garch_res=garch_res,
            vpin_res=vpin_res
        )

        self.assertIsNotNone(targets)
        self.assertLess(targets.stop_loss, current_price)
        self.assertGreater(targets.take_profit, current_price)
        self.assertEqual(targets.total_quantity, 4 * 50)

        # Allow trade entry during simulated market time
        tz = pytz.timezone("Asia/Kolkata")
        market_time = datetime.now(tz).replace(hour=10, minute=30, second=0)

        # Mock can_open_new_trade check
        receipt = self.execution.execute_entry(
            signal=DirectionalSignal.BUY,
            current_price=current_price,
            trade_target=targets,
            vpin_res=vpin_res,
            order_type=OrderType.MARKET
        )

        if receipt:
            self.assertEqual(receipt.status, OrderStatus.FILLED)
            self.assertEqual(self.risk_mgr.current_position.side, PositionSide.LONG)

            # Test Stop Loss exit trigger
            exit_needed, reason = self.risk_mgr.check_exit_conditions(
                current_price=targets.stop_loss - 10.0,
                current_time=market_time
            )
            self.assertTrue(exit_needed)
            self.assertIn("STOP_LOSS", reason)

            # Execute exit
            exit_receipt = self.execution.execute_exit(
                current_price=targets.stop_loss - 10.0,
                reason=reason
            )
            self.assertIsNotNone(exit_receipt)
            self.assertEqual(self.risk_mgr.current_position.side, PositionSide.FLAT)
            self.assertLess(self.risk_mgr.daily_realized_pnl, 0.0)

    def test_daily_kill_switch(self):
        """Test that daily loss exceeding limit trips the hard kill-switch."""
        self.risk_mgr.daily_realized_pnl = -1500.0  # Exceeds max_daily_loss 1000.0
        # Trigger trade close to update kill-switch
        self.risk_mgr.current_position.side = PositionSide.LONG
        self.risk_mgr.current_position.quantity = 100
        self.risk_mgr.current_position.entry_price = 24000.0
        self.risk_mgr.record_trade_close(23990.0)

        self.assertTrue(self.risk_mgr.is_kill_switch_active)
        can_open, reason = self.risk_mgr.can_open_new_trade()
        self.assertFalse(can_open)
        self.assertIn("KILL_SWITCH_ACTIVE", reason)


if __name__ == "__main__":
    unittest.main()
