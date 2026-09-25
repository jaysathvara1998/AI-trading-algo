"""
NIFTY 50 Heavyweight Components Live Scanner for Algo VPIN v2.0 (AURA-v2)
Implements Pillar 3 of the Pre-9:15 AM Master Routine:
- Tracks the top 8 NIFTY heavyweights (~50%+ index weight):
  RELIANCE, HDFCBANK, ICICIBANK, INFY, TCS, LT, KOTAKBANK, AXISBANK
- Calculates Advance/Decline Breadth & Weighted Component Sentiment
- Detects Index vs Component Divergences (Fakeout / Trap Confirmation)
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional
import logging

logger = logging.getLogger("algo_vpin_v2.heavyweight_scanner")


class ComponentBias(Enum):
    STRONG_BULLISH = "STRONG_BULLISH"
    MILD_BULLISH = "MILD_BULLISH"
    NEUTRAL = "NEUTRAL"
    MILD_BEARISH = "MILD_BEARISH"
    STRONG_BEARISH = "STRONG_BEARISH"
    BEARISH_DIVERGENCE_TRAP = "BEARISH_DIVERGENCE_TRAP"
    BULLISH_DIVERGENCE_TRAP = "BULLISH_DIVERGENCE_TRAP"


@dataclass
class ComponentSnapshot:
    symbol: str
    weight: float
    ltp: float
    prev_close: float
    change_pct: float
    trend: str  # ADVANCE, DECLINE, FLAT


@dataclass
class HeavyweightBreadthState:
    advances: int
    declines: int
    unchanged: int
    adv_dec_ratio: float
    weighted_score: float  # -1.0 to +1.0
    bias: ComponentBias
    divergence_signal: Optional[str]
    lead_gainers: List[str]
    lead_losers: List[str]
    summary: str


import json
import urllib.request


class HeavyweightScanner:
    """
    Live tracker for core Nifty heavyweight stocks directly via Dhan live market feed and charts.
    """
    # Key NIFTY 50 Heavyweights, approximate weights, and Dhan NSE Security IDs
    HEAVYWEIGHT_CONFIG = {
        "HDFCBANK": {"weight": 0.115, "sec_id": "1333", "mc_code": "HDF01"},
        "RELIANCE": {"weight": 0.095, "sec_id": "2885", "mc_code": "RI"},
        "ICICIBANK": {"weight": 0.080, "sec_id": "4963", "mc_code": "ICI02"},
        "INFY": {"weight": 0.055, "sec_id": "1594", "mc_code": "IT"},
        "TCS": {"weight": 0.040, "sec_id": "11536", "mc_code": "TCS"},
        "LT": {"weight": 0.035, "sec_id": "11483", "mc_code": "LT"},
        "KOTAKBANK": {"weight": 0.030, "sec_id": "1922", "mc_code": "KMB"},
        "AXISBANK": {"weight": 0.030, "sec_id": "5900", "mc_code": "AB16"}
    }

    def __init__(self, tradehull_client=None):
        self.components: Dict[str, ComponentSnapshot] = {}
        self.last_breadth_state: Optional[HeavyweightBreadthState] = None
        self.tradehull_client = tradehull_client
        self._is_polling: bool = False
        self.poll_live_heavyweights()

    def set_tradehull_client(self, tradehull_client):
        self.tradehull_client = tradehull_client
        self.poll_live_heavyweights()

    def poll_live_heavyweights(self, synchronous: bool = False) -> None:
        """
        Polls live stock prices for Nifty heavyweights.
        Runs asynchronously in a background daemon thread by default to prevent blocking the tick loop.
        """
        if self._is_polling:
            return

        if synchronous:
            self._execute_poll()
        else:
            import threading
            t = threading.Thread(target=self._execute_poll, daemon=True, name="HeavyweightScannerPoll")
            t.start()

    def _execute_poll(self) -> None:
        self._is_polling = True
        try:
            from datetime import datetime
            import pytz
            tz = pytz.timezone("Asia/Kolkata")
            today_str = datetime.now(tz).strftime("%Y-%m-%d")

            # 1. Primary: Direct Dhan Live Intraday Data
            if self.tradehull_client is not None and hasattr(self.tradehull_client, "Dhan"):
                try:
                    dhan = self.tradehull_client.Dhan
                    for sym, cfg in self.HEAVYWEIGHT_CONFIG.items():
                        sec_id = cfg["sec_id"]
                        res = dhan.intraday_minute_data(
                            security_id=sec_id,
                            exchange_segment="NSE_EQ",
                            instrument_type="EQUITY",
                            from_date=today_str,
                            to_date=today_str
                        )
                        if isinstance(res, dict) and res.get("status") == "success" and "data" in res:
                            closes = res["data"].get("close", [])
                            opens = res["data"].get("open", [])
                            if closes:
                                ltp = float(closes[-1])
                                day_open = float(opens[0]) if opens else ltp
                                self.update_component_tick(sym, ltp, day_open)
                    if self.components:
                        return
                except Exception as e:
                    logger.debug(f"[HeavyweightScanner] Dhan stock poll error: {e}")

            # 2. Secondary Fallback: Moneycontrol Price API
            for sym, cfg in self.HEAVYWEIGHT_CONFIG.items():
                code = cfg["mc_code"]
                try:
                    url = f"https://priceapi.moneycontrol.com/pricefeed/nse/equitycash/{code}"
                    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                    with urllib.request.urlopen(req, timeout=2) as resp:
                        data = json.loads(resp.read().decode())
                        d = data.get("data", {})
                        curr = float(d.get("pricecurrent", 0.0))
                        prev = float(d.get("priceprevclose", curr))
                        if curr > 0:
                            self.update_component_tick(sym, curr, prev)
                except Exception:
                    pass
        finally:
            self._is_polling = False

    def update_component_tick(self, symbol: str, ltp: float, prev_close: float) -> None:
        """
        Updates a single component stock's price.
        """
        if prev_close <= 0:
            pct = 0.0
        else:
            pct = ((ltp - prev_close) / prev_close) * 100.0

        trend = "ADVANCE" if pct > 0.05 else ("DECLINE" if pct < -0.05 else "FLAT")
        weight = self.HEAVYWEIGHT_CONFIG.get(symbol, {}).get("weight", 0.03)

        self.components[symbol] = ComponentSnapshot(
            symbol=symbol,
            weight=weight,
            ltp=ltp,
            prev_close=prev_close,
            change_pct=pct,
            trend=trend
        )

    def evaluate_breadth(self, spot_trend_bias: int = 0, is_nifty_near_pdh: bool = False, is_nifty_near_pdl: bool = False) -> HeavyweightBreadthState:
        """
        Evaluates cumulative heavyweight breadth and checks for divergence against Index spot.
        """
        if not self.components:
            # Default state if components not individually streamed yet
            return HeavyweightBreadthState(
                advances=0,
                declines=0,
                unchanged=0,
                adv_dec_ratio=1.0,
                weighted_score=0.0,
                bias=ComponentBias.NEUTRAL,
                divergence_signal=None,
                lead_gainers=[],
                lead_losers=[],
                summary="Heavyweight breadth initializing..."
            )

        adv = sum(1 for c in self.components.values() if c.trend == "ADVANCE")
        dec = sum(1 for c in self.components.values() if c.trend == "DECLINE")
        unch = sum(1 for c in self.components.values() if c.trend == "FLAT")
        total = max(1, len(self.components))

        # Weighted calculation
        total_weight = sum(c.weight for c in self.components.values())
        weighted_delta = sum(c.weight * c.change_pct for c in self.components.values()) / max(0.01, total_weight)

        weighted_score = max(-1.0, min(1.0, weighted_delta / 0.80))
        ad_ratio = (adv + 0.1) / (dec + 0.1)

        # Divergence Check
        divergence = None
        if is_nifty_near_pdh and (dec > adv or weighted_score < -0.15):
            bias = ComponentBias.BEARISH_DIVERGENCE_TRAP
            divergence = "BEARISH_DIVERGENCE (NIFTY at PDH Highs, but Heavyweights Declining!)"
        elif is_nifty_near_pdl and (adv > dec or weighted_score > 0.15):
            bias = ComponentBias.BULLISH_DIVERGENCE_TRAP
            divergence = "BULLISH_DIVERGENCE (NIFTY at PDL Lows, but Heavyweights Advancing!)"
        elif weighted_score >= 0.35:
            bias = ComponentBias.STRONG_BULLISH
        elif weighted_score >= 0.10:
            bias = ComponentBias.MILD_BULLISH
        elif weighted_score <= -0.35:
            bias = ComponentBias.STRONG_BEARISH
        elif weighted_score <= -0.10:
            bias = ComponentBias.MILD_BEARISH
        else:
            bias = ComponentBias.NEUTRAL

        gainers = sorted([c.symbol for c in self.components.values() if c.trend == "ADVANCE"], key=lambda s: self.components[s].change_pct, reverse=True)[:3]
        losers = sorted([c.symbol for c in self.components.values() if c.trend == "DECLINE"], key=lambda s: self.components[s].change_pct)[:3]

        summary = f"Adv: {adv}/{total} | Dec: {dec}/{total} | W-Score: {weighted_score:+.2f} ({bias.value})"
        if divergence:
            summary += f" [!] {divergence}"

        state = HeavyweightBreadthState(
            advances=adv,
            declines=dec,
            unchanged=unch,
            adv_dec_ratio=round(ad_ratio, 2),
            weighted_score=round(weighted_score, 3),
            bias=bias,
            divergence_signal=divergence,
            lead_gainers=gainers,
            lead_losers=losers,
            summary=summary
        )
        self.last_breadth_state = state
        return state
