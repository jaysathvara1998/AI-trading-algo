"""
End-to-end Integration Test for Quantitative Trading Engine
"""

import unittest
from algo_vpin_dhan.config import AppConfig
from algo_vpin_dhan.main import QuantitativeTradingEngine


class TestIntegrationPipeline(unittest.TestCase):

    def test_full_pipeline_simulation(self):
        """Runs the entire VPIN + GARCH + SVM quantitative trading engine across synthetic bars."""
        config = AppConfig()
        config.dhan.paper_trading = True
        config.garch.min_fit_samples = 30
        config.garch.rolling_window = 60
        config.svm.retrain_interval_bars = 10

        engine = QuantitativeTradingEngine(config)

        # Generate 250 synthetic 1-minute bars
        bars_df = engine.data_feed.generate_synthetic_bars(n_bars=250, seed=123)

        # Run simulation
        engine.run_simulation(bars_df)

        # Verify components executed
        self.assertTrue(engine.is_warmed_up)
        self.assertGreater(engine.total_bars_processed, 50)
        self.assertGreater(len(engine.vpin_calc.completed_buckets), 0)
        self.assertIsNotNone(engine.vpin_calc.latest_vpin)
        self.assertGreater(len(engine.garch_engine.returns), 30)
        self.assertTrue(engine.svm_filter.is_trained)


if __name__ == "__main__":
    unittest.main()
