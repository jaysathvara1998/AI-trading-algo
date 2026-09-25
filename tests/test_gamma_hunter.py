import unittest
from datetime import datetime
import pytz

from algo_vpin_v2.skills.expiry_gamma_hunter_skill import ExpiryGammaHunterSkill


class TestExpiryGammaHunterSkill(unittest.TestCase):
    def setUp(self):
        self.skill = ExpiryGammaHunterSkill()
        self.tz = pytz.timezone("Asia/Kolkata")

    def test_expiry_day_detection(self):
        # Thursday date (2026-09-24)
        thu_dt = self.tz.localize(datetime(2026, 9, 24, 14, 0))
        self.assertTrue(self.skill.is_expiry_day("SENSEX", thu_dt))
        self.assertTrue(self.skill.is_expiry_day("NIFTY", thu_dt))

        # Tuesday date (2026-09-22)
        tue_dt = self.tz.localize(datetime(2026, 9, 22, 14, 0))
        self.assertTrue(self.skill.is_expiry_day("NIFTY", tue_dt))
        self.assertFalse(self.skill.is_expiry_day("SENSEX", tue_dt))

    def test_gamma_window(self):
        in_win = self.tz.localize(datetime(2026, 9, 24, 13, 45))
        self.assertTrue(self.skill.is_gamma_window(in_win))

        out_win = self.tz.localize(datetime(2026, 9, 24, 10, 30))
        self.assertFalse(self.skill.is_gamma_window(out_win))

    def test_exponential_step_trailing(self):
        # Entry @ 10.0, Peak @ 15.0 (+50% -> Breakeven shield)
        sl, reason = self.skill.calculate_gamma_step_trailing(10.0, 15.0, 14.0)
        self.assertGreaterEqual(sl, 10.2)
        self.assertIn("GAMMA_BREAKEVEN_SHIELD", reason)

        # Entry @ 10.0, Peak @ 20.0 (+100% -> Tier 2 Lock +40%)
        sl, reason = self.skill.calculate_gamma_step_trailing(10.0, 20.0, 19.0)
        self.assertGreaterEqual(sl, 14.0)
        self.assertIn("GAMMA_TIER_2_LOCK", reason)

        # Entry @ 10.0, Peak @ 30.0 (+200% -> Tier 3 Lock +100%)
        sl, reason = self.skill.calculate_gamma_step_trailing(10.0, 30.0, 28.0)
        self.assertEqual(sl, 20.0)
        self.assertIn("GAMMA_TIER_3_LOCK", reason)

        # Entry @ 10.0, Peak @ 85.0 (+750% -> Super runner 20% peak trail)
        sl, reason = self.skill.calculate_gamma_step_trailing(10.0, 85.0, 80.0)
        self.assertEqual(sl, 68.0)
        self.assertIn("GAMMA_SUPER_RUNNER", reason)


if __name__ == "__main__":
    unittest.main()
