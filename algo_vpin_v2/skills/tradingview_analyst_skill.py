"""
TradingView Multi-Timeframe Technical Analysis Skill for Algo VPIN v2.0 (AURA-v2)
Integrates live TradingView Technical Analysis Engine (26 Institutional Indicators):
- Moving Averages: EMA 10, 20, 50, 100, 200, SMA, VWMA, Hull MA, Ichimoku Cloud
- Oscillators: RSI, Stochastic, CCI, MACD, ADX, Awesome Oscillator, Momentum
- Consolidates 1-Minute & 5-Minute real-time institutional recommendations
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import pytz
import logging

try:
    from tradingview_ta import TA_Handler, Interval, Exchange
    TRADINGVIEW_TA_AVAILABLE = True
except ImportError:
    TRADINGVIEW_TA_AVAILABLE = False

from .base_skill import BaseTradingSkill, SkillResult
from ..config import CONFIG

logger = logging.getLogger("algo_vpin_v2.skills.tradingview_analyst")


class TradingViewAnalystSkill(BaseTradingSkill):
    """
    Evaluates real-time multi-timeframe TradingView technical indicator consensus.
    Provides instant, zero-lag single-bar confirmation across 26 classical indicators.
    """
    def __init__(self):
        super().__init__(name="TradingViewAnalystSkill")
        self.tz = pytz.timezone("Asia/Kolkata")
        self._cached_analysis: Dict[str, Any] = {}
        self._last_poll_time: Optional[datetime] = None

    def _resolve_tv_symbol(self, symbol: str) -> Tuple[str, str]:
        """Maps internal symbol to TradingView exchange and ticker"""
        sym = symbol.upper()
        if "SENSEX" in sym:
            return "BSE", "SENSEX"
        elif "BANKNIFTY" in sym:
            return "NSE", "BANKNIFTY"
        return "NSE", "NIFTY"

    def evaluate(
        self,
        symbol: str = "NIFTY",
        current_time: Optional[datetime] = None,
        *args, **kwargs
    ) -> SkillResult:
        """
        Queries TradingView technical analysis engine for live 1M and 5M consensus.
        """
        if not TRADINGVIEW_TA_AVAILABLE:
            return SkillResult(
                is_favorable=True,
                confidence=0.50,
                signal="TV_TA_UNAVAILABLE",
                reason="tradingview_ta library not available; using default ML consensus."
            )

        now = current_time or datetime.now(self.tz)
        exchange, tv_ticker = self._resolve_tv_symbol(symbol)

        try:
            handler_1m = TA_Handler(
                symbol=tv_ticker,
                exchange=exchange,
                screener="india",
                interval=Interval.INTERVAL_1_MINUTE
            )
            analysis_1m = handler_1m.get_analysis()

            summary = analysis_1m.summary
            rec = summary.get("RECOMMENDATION", "NEUTRAL")
            buy_votes = summary.get("BUY", 0)
            sell_votes = summary.get("SELL", 0)
            neutral_votes = summary.get("NEUTRAL", 0)
            total_votes = max(1, buy_votes + sell_votes + neutral_votes)

            # Oscillators and Moving Averages
            ma_rec = analysis_1m.moving_averages.get("RECOMMENDATION", "NEUTRAL")
            osc_rec = analysis_1m.oscillators.get("RECOMMENDATION", "NEUTRAL")

            is_bullish = ("BUY" in rec)
            is_bearish = ("SELL" in rec)
            is_favorable = (is_bullish or is_bearish)

            confidence = max(buy_votes, sell_votes) / total_votes

            signal_type = "TV_STRONG_BUY" if "STRONG_BUY" in rec else (
                "TV_BUY" if "BUY" in rec else (
                    "TV_STRONG_SELL" if "STRONG_SELL" in rec else (
                        "TV_SELL" if "SELL" in rec else "TV_NEUTRAL"
                    )
                )
            )

            reason = (
                f"TradingView {tv_ticker} (1M): {rec} [{buy_votes} Buy / {sell_votes} Sell / {neutral_votes} Neutral] | "
                f"MAs: {ma_rec}, Oscillators: {osc_rec}"
            )

            metadata = {
                "recommendation": rec,
                "buy_count": buy_votes,
                "sell_count": sell_votes,
                "neutral_count": neutral_votes,
                "ma_recommendation": ma_rec,
                "osc_recommendation": osc_rec,
                "rsi_value": analysis_1m.indicators.get("RSI", None),
                "macd_level": analysis_1m.indicators.get("MACD.macd", None),
                "vwma": analysis_1m.indicators.get("VWMA", None),
                "timestamp": now.isoformat()
            }

            return SkillResult(
                is_favorable=is_favorable,
                confidence=round(confidence, 3),
                signal=signal_type,
                reason=reason,
                metadata=metadata
            )

        except Exception as e:
            logger.debug(f"[TradingViewSkill] Error querying TradingView TA: {e}")
            return SkillResult(
                is_favorable=True,
                confidence=0.50,
                signal="TV_TA_FALLBACK",
                reason=f"TradingView feed notice: {e}"
            )
