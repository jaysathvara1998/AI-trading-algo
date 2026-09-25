"""
Multi-Timeframe Pattern Confluence Scanner
(Macro Structure + Micro Trigger Execution)

Resamples high-resolution 1-minute market data into multiple higher timeframes
(3m, 5m, 15m, 1h), runs wave-based M-Pattern and W-Pattern recognition on each,
and computes institutional multi-timeframe confluence:
- Macro (15m / 5m): Establishes the major structural double top/bottom & key neckline.
- Micro (1m / 3m): Times the exact neckline breakout/breakdown candle to enter with minimal slippage.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd

from .m_pattern_skill import MPatternSkill, MPatternMetrics, MPatternStage, PatternType
from .base_skill import BaseTradingSkill, SkillResult


@dataclass
class MultiTimeframeResult:
    is_confluent: bool
    direction: str                     # "BUY_PUT" (Bearish M), "BUY_CALL" (Bullish W), or "NEUTRAL"
    primary_timeframe: str             # e.g. "5m"
    trigger_timeframe: str             # e.g. "1m"
    confluence_score: float            # 0.0 to 1.0
    neckline_level: float
    target_1x: float
    target_1_618x: float
    stop_loss: float
    tf_metrics: Dict[str, Optional[MPatternMetrics]] = field(default_factory=dict)
    summary: str = ""


class MultiTimeframeScanner(BaseTradingSkill):
    """
    Evaluates wave-based price action patterns across multiple timeframes simultaneously.
    """
    def __init__(
        self,
        name: str = "MultiTimeframeScanner",
        timeframes: Optional[List[str]] = None
    ):
        super().__init__(name=name)
        self.timeframes = timeframes or ["1m", "3m", "5m", "15m"]
        self.pattern_skill = MPatternSkill()

    def resample_ohlcv(self, df_1m: pd.DataFrame, target_tf: str) -> pd.DataFrame:
        """
        Resamples a 1-minute OHLCV DataFrame into higher timeframes (3m, 5m, 15m, 1h).
        Handles both datetime-indexed and sequential bar DataFrames.
        """
        if df_1m.empty or len(df_1m) < 2:
            return df_1m

        # Parse rule
        tf_map = {"1m": 1, "3m": 3, "5m": 5, "15m": 15, "1h": 60, "60m": 60}
        step = tf_map.get(target_tf.lower(), 1)
        if step == 1 or len(df_1m) < step:
            return df_1m.copy()

        # If datetime index is present, use pandas resample
        if isinstance(df_1m.index, pd.DatetimeIndex):
            rule_str = f"{step}min" if step < 60 else "1h"
            res = df_1m.resample(rule_str).agg({
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum"
            }).dropna()
            return res

        # Otherwise, aggregate sequentially by chunks of `step` bars
        n = len(df_1m)
        records = []
        for i in range(0, n, step):
            chunk = df_1m.iloc[i : i + step]
            if len(chunk) == 0:
                continue
            records.append({
                "open": float(chunk["open"].iloc[0]),
                "high": float(chunk["high"].max()),
                "low": float(chunk["low"].min()),
                "close": float(chunk["close"].iloc[-1]),
                "volume": float(chunk["volume"].sum()) if "volume" in chunk else 1.0
            })

        return pd.DataFrame(records)

    def scan_confluence(
        self,
        df_1m: pd.DataFrame,
        current_price: Optional[float] = None
    ) -> MultiTimeframeResult:
        """
        Scans all configured timeframes for pattern alignment.
        """
        curr_p = current_price if (current_price and current_price > 0) else (
            float(df_1m["close"].iloc[-1]) if not df_1m.empty else 0.0
        )

        tf_metrics: Dict[str, Optional[MPatternMetrics]] = {}
        tf_signals: Dict[str, str] = {}

        # 1. Scan each timeframe
        for tf in self.timeframes:
            df_tf = self.resample_ohlcv(df_1m, tf) if tf != "1m" else df_1m.copy()
            if len(df_tf) < 5:
                tf_metrics[tf] = None
                tf_signals[tf] = "NEUTRAL"
                continue

            # Check M-pattern first
            m_res = self.pattern_skill.scan_m_pattern(df_tf, current_price=curr_p)
            if m_res and m_res.stage in (MPatternStage.BREAKDOWN_CONFIRMED, MPatternStage.TESTING_NECKLINE):
                tf_metrics[tf] = m_res
                tf_signals[tf] = "BEARISH_M"
                continue

            # Check W-pattern next
            w_res = self.pattern_skill.scan_w_pattern(df_tf, current_price=curr_p)
            if w_res and w_res.stage in (MPatternStage.BREAKOUT_CONFIRMED, MPatternStage.TESTING_NECKLINE):
                tf_metrics[tf] = w_res
                tf_signals[tf] = "BULLISH_W"
                continue

            tf_metrics[tf] = None
            tf_signals[tf] = "NEUTRAL"

        # 2. Analyze Confluence
        bearish_tfs = [tf for tf, sig in tf_signals.items() if sig == "BEARISH_M"]
        bullish_tfs = [tf for tf, sig in tf_signals.items() if sig == "BULLISH_W"]

        # Case A: Bearish M-Pattern Alignment
        if len(bearish_tfs) >= 2 or ("1m" in bearish_tfs and any(tf in bearish_tfs for tf in ["3m", "5m", "15m"])):
            primary_tf = "5m" if "5m" in bearish_tfs else ("15m" if "15m" in bearish_tfs else bearish_tfs[0])
            m = tf_metrics[primary_tf] or tf_metrics[bearish_tfs[0]]
            score = min(0.98, 0.70 + (len(bearish_tfs) * 0.10))
            summary = (
                f"Multi-Timeframe M-Breakdown Confluence across {bearish_tfs}! "
                f"Macro {primary_tf} Neckline: {m.neckline_level:.1f}, "
                f"Target 1x: {m.target_1x:.1f}, SL: {m.stop_loss:.1f}"
            )
            return MultiTimeframeResult(
                is_confluent=True,
                direction="BUY_PUT",
                primary_timeframe=primary_tf,
                trigger_timeframe="1m" if "1m" in bearish_tfs else primary_tf,
                confluence_score=round(score, 2),
                neckline_level=m.neckline_level,
                target_1x=m.target_1x,
                target_1_618x=m.target_1_618x,
                stop_loss=m.stop_loss,
                tf_metrics=tf_metrics,
                summary=summary
            )

        # Case B: Bullish W-Pattern Alignment
        if len(bullish_tfs) >= 2 or ("1m" in bullish_tfs and any(tf in bullish_tfs for tf in ["3m", "5m", "15m"])):
            primary_tf = "5m" if "5m" in bullish_tfs else ("15m" if "15m" in bullish_tfs else bullish_tfs[0])
            w = tf_metrics[primary_tf] or tf_metrics[bullish_tfs[0]]
            score = min(0.98, 0.70 + (len(bullish_tfs) * 0.10))
            summary = (
                f"Multi-Timeframe W-Breakout Confluence across {bullish_tfs}! "
                f"Macro {primary_tf} Neckline: {w.neckline_level:.1f}, "
                f"Target 1x: {w.target_1x:.1f}, SL: {w.stop_loss:.1f}"
            )
            return MultiTimeframeResult(
                is_confluent=True,
                direction="BUY_CALL",
                primary_timeframe=primary_tf,
                trigger_timeframe="1m" if "1m" in bullish_tfs else primary_tf,
                confluence_score=round(score, 2),
                neckline_level=w.neckline_level,
                target_1x=w.target_1x,
                target_1_618x=w.target_1_618x,
                stop_loss=w.stop_loss,
                tf_metrics=tf_metrics,
                summary=summary
            )

        # Single timeframe trigger fallback
        if len(bearish_tfs) == 1:
            tf = bearish_tfs[0]
            m = tf_metrics[tf]
            return MultiTimeframeResult(
                is_confluent=False,
                direction="BUY_PUT",
                primary_timeframe=tf,
                trigger_timeframe=tf,
                confluence_score=0.75,
                neckline_level=m.neckline_level,
                target_1x=m.target_1x,
                target_1_618x=m.target_1_618x,
                stop_loss=m.stop_loss,
                tf_metrics=tf_metrics,
                summary=f"Single-timeframe M-Pattern detected on {tf} (Neckline: {m.neckline_level:.1f})"
            )

        if len(bullish_tfs) == 1:
            tf = bullish_tfs[0]
            w = tf_metrics[tf]
            return MultiTimeframeResult(
                is_confluent=False,
                direction="BUY_CALL",
                primary_timeframe=tf,
                trigger_timeframe=tf,
                confluence_score=0.75,
                neckline_level=w.neckline_level,
                target_1x=w.target_1x,
                target_1_618x=w.target_1_618x,
                stop_loss=w.stop_loss,
                tf_metrics=tf_metrics,
                summary=f"Single-timeframe W-Pattern detected on {tf} (Neckline: {w.neckline_level:.1f})"
            )

        return MultiTimeframeResult(
            is_confluent=False,
            direction="NEUTRAL",
            primary_timeframe="",
            trigger_timeframe="",
            confluence_score=0.50,
            neckline_level=0.0,
            target_1x=0.0,
            target_1_618x=0.0,
            stop_loss=0.0,
            tf_metrics=tf_metrics,
            summary="No multi-timeframe pattern confluence."
        )

    def evaluate(
        self,
        ohlcv: Union[pd.DataFrame, List[dict]],
        current_price: Optional[float] = None,
        intended_direction: Optional[str] = None
    ) -> SkillResult:
        """
        Returns standardized SkillResult for the EnsembleBrain.
        """
        df_1m = ohlcv if isinstance(ohlcv, pd.DataFrame) else pd.DataFrame(ohlcv)
        res = self.scan_confluence(df_1m, current_price=current_price)

        if res.is_confluent and res.direction in ("BUY_PUT", "BUY_CALL"):
            return SkillResult(
                is_favorable=True,
                confidence=res.confluence_score,
                signal=res.direction,
                reason=res.summary,
                metadata={
                    "primary_tf": res.primary_timeframe,
                    "trigger_tf": res.trigger_timeframe,
                    "neckline": res.neckline_level,
                    "target_1x": res.target_1x,
                    "target_1_618x": res.target_1_618x,
                    "stop_loss": res.stop_loss
                }
            )

        return SkillResult(
            is_favorable=False,
            confidence=res.confluence_score,
            signal=res.direction,
            reason=res.summary
        )
