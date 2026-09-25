import unittest
from datetime import datetime
import pytz

from algo_vpin_v2.skills.tradingview_analyst_skill import TradingViewAnalystSkill


class TestTradingViewAnalystSkill(unittest.TestCase):
    def setUp(self):
        self.skill = TradingViewAnalystSkill()
        self.tz = pytz.timezone("Asia/Kolkata")

    def test_symbol_resolution(self):
        exch_n, sym_n = self.skill._resolve_tv_symbol("NIFTY")
        self.assertEqual(exch_n, "NSE")
        self.assertEqual(sym_n, "NIFTY")

        exch_s, sym_s = self.skill._resolve_tv_symbol("SENSEX")
        self.assertEqual(exch_s, "BSE")
        self.assertEqual(sym_s, "SENSEX")

        exch_b, sym_b = self.skill._resolve_tv_symbol("BANKNIFTY")
        self.assertEqual(exch_b, "NSE")
        self.assertEqual(sym_b, "BANKNIFTY")

    def test_evaluation_fallback_or_live(self):
        now = self.tz.localize(datetime(2026, 9, 24, 14, 0))
        res = self.skill.evaluate("NIFTY", current_time=now)
        self.assertIsNotNone(res)
        self.assertIn("signal", dir(res))
        self.assertGreaterEqual(res.confidence, 0.0)


if __name__ == "__main__":
    unittest.main()
