"""
Options Premium Chart Skill
(Direct Analysis of Option CE & PE Candlestick Charts)

Trades and confirms patterns directly on Option Premium charts:
- PE Chart Analysis: Detects W-Pattern Breakouts (surging put premium when spot dumps)
- CE Chart Analysis: Detects W-Pattern Breakouts (surging call premium when spot rallies)
- Cross-Asset Synchronization:
  When Spot Index forms an M-Pattern Breakdown AND Option PE Chart forms a W-Pattern Breakout,
  the trade conviction is maximum (>95%) with zero directional divergence.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd

from .m_pattern_skill import MPatternSkill, MPatternMetrics, MPatternStage
from .base_skill import BaseTradingSkill, SkillResult


@dataclass
class OptionChartAnalysis:
    symbol: str
    option_type: str                  # "CE" or "PE"
    strike: int
    current_premium: float
    pattern_detected: Optional[str]   # "W_BREAKOUT", "M_BREAKDOWN", or None
    stage: MPatternStage
    neckline_premium: float
    target_1x_premium: float
    stop_loss_premium: float
    confidence: float
    reason: str


class OptionChartSkill(BaseTradingSkill):
    """
    Skill for analyzing option candlestick charts directly and cross-verifying with Spot Index.
    """
    def __init__(self, name: str = "OptionChartSkill"):
        super().__init__(name=name)
        # Option charts typically have 2-5 pt minimum swings
        self.pattern_engine = MPatternSkill(
            name="OptionPatternEngine",
            min_pattern_depth_pts=2.5,
            max_peak_diff_pct=0.08,   # Up to 8% difference between option tops/bottoms
            min_swing_pct=0.02        # 2% minimum swing on option premium
        )

    def analyze_option_chart(
        self,
        ohlcv: Union[pd.DataFrame, List[dict]],
        symbol: str = "NIFTY_OPT",
        option_type: str = "PE",
        strike: int = 23400,
        current_premium: Optional[float] = None
    ) -> OptionChartAnalysis:
        """
        Analyzes an option premium candlestick chart for M-breakdown or W-breakout.
        """
        df = ohlcv if isinstance(ohlcv, pd.DataFrame) else pd.DataFrame(ohlcv)
        curr_p = current_premium if current_premium else float(df["close"].iloc[-1])

        # 1. Check for W-pattern breakout on option premium (e.g. Put surging)
        w_met = self.pattern_engine.scan_w_pattern(df, current_price=curr_p)
        if w_met and w_met.stage == MPatternStage.BREAKOUT_CONFIRMED:
            return OptionChartAnalysis(
                symbol=symbol,
                option_type=option_type,
                strike=strike,
                current_premium=round(curr_p, 2),
                pattern_detected="W_BREAKOUT",
                stage=w_met.stage,
                neckline_premium=w_met.neckline_level,
                target_1x_premium=w_met.target_1x,
                stop_loss_premium=w_met.stop_loss,
                confidence=round(0.80 + (w_met.quality_score * 0.15), 2),
                reason=(
                    f"Option {option_type} {strike} confirmed W-Pattern Breakout above {w_met.neckline_level:.1f}! "
                    f"Target: {w_met.target_1x:.1f} | SL: {w_met.stop_loss:.1f}"
                )
            )

        # 2. Check for M-pattern breakdown on option premium (e.g. decaying option)
        m_met = self.pattern_engine.scan_m_pattern(df, current_price=curr_p)
        if m_met and m_met.stage == MPatternStage.BREAKDOWN_CONFIRMED:
            return OptionChartAnalysis(
                symbol=symbol,
                option_type=option_type,
                strike=strike,
                current_premium=round(curr_p, 2),
                pattern_detected="M_BREAKDOWN",
                stage=m_met.stage,
                neckline_premium=m_met.neckline_level,
                target_1x_premium=m_met.target_1x,
                stop_loss_premium=m_met.stop_loss,
                confidence=round(0.80 + (m_met.quality_score * 0.15), 2),
                reason=(
                    f"Option {option_type} {strike} confirmed M-Pattern Breakdown below {m_met.neckline_level:.1f}! "
                    f"Target: {m_met.target_1x:.1f} | SL: {m_met.stop_loss:.1f}"
                )
            )

        return OptionChartAnalysis(
            symbol=symbol,
            option_type=option_type,
            strike=strike,
            current_premium=round(curr_p, 2),
            pattern_detected=None,
            stage=MPatternStage.NONE,
            neckline_premium=0.0,
            target_1x_premium=0.0,
            stop_loss_premium=0.0,
            confidence=0.50,
            reason="No confirmed M or W pattern on option chart."
        )

    def cross_validate(
        self,
        spot_signal: str,                     # "BUY_PUT" (Spot M-Breakdown) or "BUY_CALL" (Spot W-Breakout)
        option_analysis: OptionChartAnalysis
    ) -> SkillResult:
        """
        Cross-validates Spot Index signal with Option Chart Analysis.
        """
        # Synchronized Pair 1: Bearish Spot (M-Breakdown) + Bullish Put (PE W-Breakout)
        if spot_signal == "BUY_PUT" and option_analysis.option_type == "PE":
            if option_analysis.pattern_detected == "W_BREAKOUT":
                return SkillResult(
                    is_favorable=True,
                    confidence=0.95,
                    signal="BUY_PUT",
                    reason=(
                        f"HIGH CONVICTION CONFLUENCE! Spot Index confirmed M-Pattern breakdown AND "
                        f"{option_analysis.symbol} confirmed W-Pattern breakout above {option_analysis.neckline_premium:.1f}!"
                    ),
                    metadata={
                        "strike": option_analysis.strike,
                        "premium_entry": option_analysis.current_premium,
                        "premium_tp1": option_analysis.target_1x_premium,
                        "premium_sl": option_analysis.stop_loss_premium
                    }
                )
            elif option_analysis.pattern_detected == "M_BREAKDOWN":
                return SkillResult(
                    is_favorable=False,
                    confidence=0.70,
                    signal="NEUTRAL",
                    reason=f"DIVERGENCE VETO: Spot is bearish but PE option chart is breaking down ({option_analysis.neckline_premium:.1f})."
                )

        # Synchronized Pair 2: Bullish Spot (W-Breakout) + Bullish Call (CE W-Breakout)
        if spot_signal == "BUY_CALL" and option_analysis.option_type == "CE":
            if option_analysis.pattern_detected == "W_BREAKOUT":
                return SkillResult(
                    is_favorable=True,
                    confidence=0.95,
                    signal="BUY_CALL",
                    reason=(
                        f"HIGH CONVICTION CONFLUENCE! Spot Index confirmed W-Pattern breakout AND "
                        f"{option_analysis.symbol} confirmed W-Pattern breakout above {option_analysis.neckline_premium:.1f}!"
                    ),
                    metadata={
                        "strike": option_analysis.strike,
                        "premium_entry": option_analysis.current_premium,
                        "premium_tp1": option_analysis.target_1x_premium,
                        "premium_sl": option_analysis.stop_loss_premium
                    }
                )

        # Default fallback
        return SkillResult(
            is_favorable=True,
            confidence=0.80,
            signal=spot_signal,
            reason=f"Spot signal {spot_signal} accepted with option premium at {option_analysis.current_premium:.1f}."
        )

    def evaluate(
        self,
        ohlcv: Union[pd.DataFrame, List[dict]],
        current_price: Optional[float] = None,
        intended_direction: Optional[str] = None
    ) -> SkillResult:
        analysis = self.analyze_option_chart(ohlcv, current_premium=current_price)
        if analysis.pattern_detected == "W_BREAKOUT":
            return SkillResult(
                is_favorable=True,
                confidence=analysis.confidence,
                signal="BUY_CALL" if analysis.option_type == "CE" else "BUY_PUT",
                reason=analysis.reason,
                metadata={
                    "neckline": analysis.neckline_premium,
                    "target_1x": analysis.target_1x_premium,
                    "stop_loss": analysis.stop_loss_premium
                }
            )
        return SkillResult(is_favorable=False, confidence=0.50, signal="NEUTRAL", reason=analysis.reason)
