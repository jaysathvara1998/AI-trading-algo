"""
Unit tests for Layer 2: GARCH(1,1) Volatility Forecasting Engine
"""

import unittest
import numpy as np

from algo_vpin_dhan.config import GARCHConfig
from algo_vpin_dhan.garch_engine import GARCHEngine, DirectionalSignal


class TestGARCHEngine(unittest.TestCase):

    def setUp(self):
        self.config = GARCHConfig(
            p=1,
            q=1,
            mean_model="AR",
            ar_lags=1,
            delta_1=0.00015,
            rolling_window=100,
            min_fit_samples=30
        )
        self.engine = GARCHEngine(self.config)

    def test_log_returns_and_warmup(self):
        """Test that log returns are recorded properly and forecast runs after warmup."""
        prices = [24000.0]
        for i in range(25):
            p = prices[-1] * (1.0 + np.random.normal(0, 0.001))
            prices.append(p)
            res = self.engine.add_bar(p)
            self.assertIsNone(res)  # Not enough samples yet (< 30)

        # Feed up to 40 samples
        for i in range(15):
            p = prices[-1] * (1.0 + np.random.normal(0, 0.001))
            prices.append(p)
            res = self.engine.add_bar(p)

        self.assertIsNotNone(res)
        self.assertGreater(res.h_next, 0.0)
        self.assertGreater(res.sigma_next, 0.0)
        self.assertIn(res.signal, [DirectionalSignal.BUY, DirectionalSignal.SELL, DirectionalSignal.HOLD])

    def test_directional_trigger_thresholds(self):
        """Test delta_1 directional threshold behavior."""
        # Strong upward drift series
        drift_prices = [24000.0]
        for i in range(50):
            p = drift_prices[-1] * 1.0005  # +5 bps every bar
            drift_prices.append(p)
            res = self.engine.add_bar(p)

        self.assertIsNotNone(res)
        self.assertGreater(res.mu_next, 0.0)


if __name__ == "__main__":
    unittest.main()
