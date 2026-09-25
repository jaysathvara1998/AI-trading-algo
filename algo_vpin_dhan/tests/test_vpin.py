"""
Unit tests for Layer 1: BVC and VPIN Calculator
"""

import unittest
import numpy as np
import pandas as pd

from algo_vpin_dhan.config import VPINConfig
from algo_vpin_dhan.vpin import VPINCalculator, ToxicityRegime


class TestVPINCalculator(unittest.TestCase):

    def setUp(self):
        self.config = VPINConfig(
            n_buckets=10,
            sigma_window=20,
            delta_2=0.55,
            delta_3=0.25,
            default_bucket_volume=1000.0
        )
        self.calc = VPINCalculator(self.config)

    def test_bvc_volume_conservation(self):
        """Test that V^B + V^S == total bar volume within floating point precision."""
        price = 24050.0
        prev_price = 24000.0
        volume = 5000.0

        buy_v, sell_v = self.calc.classify_bar_bvc(price, prev_price, volume)
        self.assertAlmostEqual(buy_v + sell_v, volume, places=4)
        self.assertGreater(buy_v, sell_v)  # Price rose, so buy volume > sell volume

        # Test price fall
        price_drop = 23950.0
        buy_v_drop, sell_v_drop = self.calc.classify_bar_bvc(price_drop, price, volume)
        self.assertAlmostEqual(buy_v_drop + sell_v_drop, volume, places=4)
        self.assertGreater(sell_v_drop, buy_v_drop)  # Price fell, sell volume > buy volume

    def test_bucket_boundary_splitting(self):
        """Test that continuous volume is properly partitioned into exact buckets."""
        # Process a bar with volume equal to 2.5 buckets
        res = self.calc.process_bar(close_price=24100.0, volume=2500.0)
        self.assertEqual(len(self.calc.completed_buckets), 2)
        self.assertAlmostEqual(self.calc.curr_bucket_filled, 500.0, places=4)

        # Process another bar to finish the third bucket
        res2 = self.calc.process_bar(close_price=24120.0, volume=500.0)
        self.assertEqual(len(self.calc.completed_buckets), 3)
        self.assertAlmostEqual(self.calc.curr_bucket_filled, 0.0, places=4)

    def test_vpin_bounds_and_toxicity_regime(self):
        """Test VPIN calculation and regime categorization."""
        # Simulate high buy imbalance (high toxicity)
        for i in range(25):
            self.calc.process_bar(close_price=24000.0 + (i * 20.0), volume=1000.0)

        vpin = self.calc._compute_vpin()
        self.assertGreaterEqual(vpin, 0.0)
        self.assertLessEqual(vpin, 1.0)
        self.assertGreater(vpin, 0.50)  # Strong directional move yields high toxicity

        # Test regime outputs
        regime, entry_mult, size_mult = self.calc._classify_regime(0.60)
        self.assertEqual(regime, ToxicityRegime.HIGH_TOXICITY)
        self.assertEqual(size_mult, 0.5)
        self.assertEqual(entry_mult, 1.5)

        regime_low, entry_mult_low, size_mult_low = self.calc._classify_regime(0.20)
        self.assertEqual(regime_low, ToxicityRegime.LOW_TOXICITY)
        self.assertEqual(size_mult_low, 1.0)
        self.assertEqual(entry_mult_low, 0.8)


if __name__ == "__main__":
    unittest.main()
