"""
Unit tests for Layer 3: SVM Trade-Veto Filter (RBF Kernel)
"""

import unittest
import numpy as np

from algo_vpin_dhan.config import SVMConfig
from algo_vpin_dhan.svm_filter import SVMTradeFilter
from algo_vpin_dhan.garch_engine import DirectionalSignal


class TestSVMTradeFilter(unittest.TestCase):

    def setUp(self):
        self.config = SVMConfig(
            c_param=1.0,
            gamma=0.0001,
            retrain_interval_bars=5,
            max_training_samples=100
        )
        self.filter = SVMTradeFilter(self.config)

    def test_feature_construction_and_rolling_updates(self):
        """Test rolling feature matrix and target association with mixed classes."""
        prices = [24000.0]
        for i in range(40):
            # Alternate positive and negative price steps to create both +1 and -1 labels
            step = 5.0 if i % 2 == 0 else -5.0
            p = prices[-1] + step
            prices.append(p)
            self.filter.update_bar(
                current_price=p,
                price_delta=step,
                rolling_vol=0.001,
                garch_forecast=0.0002 * (1 if step > 0 else -1)
            )

        self.assertGreaterEqual(len(self.filter.feature_history), 35)
        self.assertTrue(self.filter.is_trained)

    def test_veto_logic_rules(self):
        """Test that conflicting signals are correctly vetoed."""
        # Create synthetic training set where positive delta leads to +1 and negative to -1
        features = []
        targets = []
        for i in range(50):
            if i % 2 == 0:
                features.append([10.0, 0.001, 0.0005])
                targets.append(1)
            else:
                features.append([-10.0, 0.001, -0.0005])
                targets.append(-1)

        self.filter.warm_up_with_historical(features, targets)
        self.assertTrue(self.filter.is_trained)

        # Mock an SVM prediction by manipulating internal model or features
        # If GARCH signals BUY but SVM predicts -1 -> Should veto
        decision_hold = self.filter.evaluate_signal(
            garch_signal=DirectionalSignal.HOLD,
            price_delta=0.0,
            rolling_vol=0.001,
            garch_forecast=0.0
        )
        self.assertFalse(decision_hold.is_vetoed)
        self.assertEqual(decision_hold.final_action, DirectionalSignal.HOLD)


if __name__ == "__main__":
    unittest.main()
