"""
Global Market Sentiment Engine for Algo VPIN v2.0 (AURA-v2)
Implements Pillar 1 of the Pre-9:15 AM Master Routine:
- Assesses GIFT Nifty & Global Indices Sentiment
- Analyzes Opening Gap vs Previous Day Close (PDC)
- Quantifies Global Macro Bias (BULLISH, NEUTRAL, BEARISH)
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional
import logging

logger = logging.getLogger("algo_vpin_v2.global_market")


class GlobalSentiment(Enum):
    STRONG_BULLISH = "STRONG_BULLISH"
    MILD_BULLISH = "MILD_BULLISH"
    NEUTRAL = "NEUTRAL"
    MILD_BEARISH = "MILD_BEARISH"
    STRONG_BEARISH = "STRONG_BEARISH"


@dataclass
class GlobalMarketState:
    sentiment: GlobalSentiment
    opening_gap_pts: float
    opening_gap_pct: float
    gift_nifty_bias: str
    us_futures_bias: str
    global_score: float  # -1.0 (Strong Bearish) to +1.0 (Strong Bullish)
    description: str


import json
import urllib.request


class GlobalMarketEngine:
    """
    Evaluates global pre-market cues and opening gap dynamics.
    """
    def __init__(self):
        self.last_state: Optional[GlobalMarketState] = None
        self.live_us_delta_pct: float = 0.0
        self.live_gift_delta_pct: float = 0.0
        self.mc_advances: int = 0
        self.mc_declines: int = 0
        self.mc_nifty_change_pct: float = 0.0
        self.fetch_live_global_indices()
        self.fetch_moneycontrol_indices()

    def fetch_moneycontrol_indices(self, symbol: str = "NIFTY") -> None:
        """
        Fetches live NIFTY 50 or SENSEX and market breadth cues directly from Moneycontrol Price API.
        """
        try:
            mc_code = "in%3BSEN" if "SENSEX" in symbol.upper() else "in%3BNSX"
            url = f"https://priceapi.moneycontrol.com/pricefeed/notapplicable/inidicesindia/{mc_code}"
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=4) as resp:
                data = json.loads(resp.read().decode())
                d = data.get("data", {})
                if d:
                    self.mc_advances = int(d.get("adv", 0))
                    self.mc_declines = int(d.get("decl", 0))
                    self.mc_nifty_change_pct = float(d.get("pricepercentchange", 0.0))
                    curr_p = float(d.get("pricecurrent", 0.0))
                    logger.info(f"[GlobalMarketEngine] Moneycontrol Live Feed: {symbol.upper()} @ {curr_p:,.2f} ({self.mc_nifty_change_pct:+.2f}%) | Adv: {self.mc_advances} | Dec: {self.mc_declines}")
        except Exception as e:
            logger.debug(f"[GlobalMarketEngine] Moneycontrol index fetch notice: {e}")

    def fetch_live_global_indices(self) -> None:
        """
        Fetches live Global index cues (US Dow / S&P Futures) via public quote endpoints.
        """
        try:
            # Query Dow Jones futures / index
            url = "https://query1.finance.yahoo.com/v8/finance/chart/%5EDJI?interval=1d&range=1d"
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=4) as resp:
                d = json.loads(resp.read().decode())
                meta = d.get("chart", {}).get("result", [{}])[0].get("meta", {})
                curr = meta.get("regularMarketPrice", 0.0)
                prev = meta.get("previousClose", curr)
                if prev > 0:
                    self.live_us_delta_pct = ((curr - prev) / prev) * 100.0
                    logger.info(f"[GlobalMarketEngine] Live Global Dow: {curr:,.2f} ({self.live_us_delta_pct:+.2f}%)")
        except Exception as e:
            logger.debug(f"[GlobalMarketEngine] Live global fetch notice: {e}")

    def evaluate_premarket_cues(
        self,
        current_spot: float,
        prev_close: float,
        gift_nifty_delta_pct: Optional[float] = None,
        us_futures_delta_pct: Optional[float] = None,
        symbol: str = "NIFTY"
    ) -> GlobalMarketState:
        """
        Evaluates pre-market sentiment from spot open vs prev_close and global indices.
        """
        if gift_nifty_delta_pct is None:
            gift_nifty_delta_pct = self.live_gift_delta_pct
        if us_futures_delta_pct is None:
            us_futures_delta_pct = self.live_us_delta_pct

        # Cross-Index Anomaly Guard (e.g. SENSEX 74,000 compared with NIFTY 23,000)
        if prev_close <= 0 or abs(current_spot - prev_close) / max(1.0, prev_close) > 0.30:
            if hasattr(self, "mc_nifty_change_pct") and self.mc_nifty_change_pct != 0.0:
                gap_pct = self.mc_nifty_change_pct
                gap_pts = current_spot * (gap_pct / 100.0)
            else:
                gap_pts = 0.0
                gap_pct = 0.0
        else:
            gap_pts = current_spot - prev_close
            gap_pct = (gap_pts / prev_close) * 100.0

        # Weighted sentiment score
        # Gap %: weight 0.6, GIFT Nifty: weight 0.25, US Futures: weight 0.15
        gap_normalized = max(-1.0, min(1.0, gap_pct / 0.75))  # +/- 0.75% is max scale
        gift_normalized = max(-1.0, min(1.0, gift_nifty_delta_pct / 0.75))
        us_normalized = max(-1.0, min(1.0, us_futures_delta_pct / 1.0))

        composite_score = (gap_normalized * 0.60) + (gift_normalized * 0.25) + (us_normalized * 0.15)

        if composite_score >= 0.40:
            sentiment = GlobalSentiment.STRONG_BULLISH
            desc = f"Strong Bullish Opening Gap (+{gap_pct:.2f}% | +{gap_pts:.1f} pts)"
        elif composite_score >= 0.12:
            sentiment = GlobalSentiment.MILD_BULLISH
            desc = f"Mild Bullish Bias (+{gap_pct:.2f}% | +{gap_pts:.1f} pts)"
        elif composite_score <= -0.40:
            sentiment = GlobalSentiment.STRONG_BEARISH
            desc = f"Strong Bearish Gap Down ({gap_pct:.2f}% | {gap_pts:.1f} pts)"
        elif composite_score <= -0.12:
            sentiment = GlobalSentiment.MILD_BEARISH
            desc = f"Mild Bearish Bias ({gap_pct:.2f}% | {gap_pts:.1f} pts)"
        else:
            sentiment = GlobalSentiment.NEUTRAL
            desc = f"Flat/Neutral Opening ({gap_pct:+.2f}% | {gap_pts:+.1f} pts)"

        gift_bias = "BULLISH" if gift_nifty_delta_pct > 0.1 else ("BEARISH" if gift_nifty_delta_pct < -0.1 else "FLAT")
        us_bias = "BULLISH" if us_futures_delta_pct > 0.1 else ("BEARISH" if us_futures_delta_pct < -0.1 else "FLAT")

        state = GlobalMarketState(
            sentiment=sentiment,
            opening_gap_pts=round(gap_pts, 2),
            opening_gap_pct=round(gap_pct, 2),
            gift_nifty_bias=gift_bias,
            us_futures_bias=us_bias,
            global_score=round(composite_score, 3),
            description=desc
        )
        self.last_state = state
        return state
