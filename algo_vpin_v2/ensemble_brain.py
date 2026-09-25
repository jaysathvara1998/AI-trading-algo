"""
Layer 3: Joint Ensemble Brain (SVM RBF + XGBoost Trees + Deep ANN Neural Network + Macro Level Filter)
Combines:
1. SVM RBF Kernel Non-Linear Boundary
2. XGBoost Gradient Boosted Tree Probabilities
3. Deep Artificial Neural Network (ANN) Multi-Class Multi-Layer Perceptron
4. Macro Level Veto (Blocks buying at Weekly High / selling at Weekly Low)
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
import logging
import numpy as np

from .config import EnsembleConfig
from .garch_engine import DirectionalSignal
from .svm_filter import SVMTradeFilter
from .xgboost_brain import XGBoostBrain
from .ann_brain import ANNBrain
from .macro_features import MacroFeatureState
from .skills import (
    PriceActionSkill,
    StrikeSelectorSkill,
    RiskRewardSkill,
    PositionGuardianSkill,
    MPatternSkill,
    MultiTimeframeScanner,
    OptionChartSkill,
    ExpiryGammaHunterSkill,
    TradingViewAnalystSkill,
    DynamicSkillRegistry,
    SkillResult,
    RiskRewardPlan,
    GuardianAction
)

try:
    from vision_chart_brain.visual_predictor import VisualPredictor
except ImportError:
    VisualPredictor = None

logger = logging.getLogger("algo_vpin_v2.ensemble_brain")


@dataclass
class EnsembleDecision:
    """Consolidated decision from the multi-model Brain"""
    is_vetoed: bool
    final_action: DirectionalSignal
    svm_prediction: int
    xgb_prediction: int
    xgb_confidence: float
    reason: str
    macro_state: Optional[MacroFeatureState] = None
    
    # Tri-Brain Extensions (ANN Deep Neural Network)
    ann_prediction: int = 0
    ann_confidence: float = 0.50
    tri_is_vetoed: bool = False
    tri_final_action: DirectionalSignal = DirectionalSignal.HOLD
    tri_reason: str = ""

    # Quad-Brain Extensions (Vision 2D CNN Chart Brain)
    vision_action: str = "HOLD"
    vision_confidence: float = 0.50
    vision_probs: Optional[Dict[str, float]] = None
    vision_numeric_signal: int = 0

    # Chinmay Scalping Skills Extensions
    price_action_result: Optional[SkillResult] = None
    strike_result: Optional[SkillResult] = None
    risk_reward_result: Optional[SkillResult] = None


class EnsembleBrain:
    """
    Joint Machine Learning Brain managing SVM, XGBoost, Deep ANN Neural Network, and 2D CNN Vision Brain,
    empowered with Chinmay's modular Trading Skills (Price Action, Strike Selector, Risk-Reward, Position Guardian).
    """
    def __init__(self, config: Optional[EnsembleConfig] = None):
        self.config = config or EnsembleConfig()
        self.svm = SVMTradeFilter()
        self.xgb = XGBoostBrain(self.config)
        self.ann = ANNBrain(self.config, hidden_layer_sizes=(64, 32))
        self.ann.model_path = self.config.models_dir / "ann_model.joblib" if hasattr(self.config, "models_dir") else self.ann.model_path
        self.vision = VisualPredictor() if VisualPredictor is not None else None

        # Modular Trading Skills (Chinmay Methodology)
        self.price_action_skill = PriceActionSkill()
        self.strike_selector_skill = StrikeSelectorSkill()
        self.risk_reward_skill = RiskRewardSkill()
        self.position_guardian_skill = PositionGuardianSkill()
        self.m_pattern_skill = MPatternSkill()
        self.mtf_scanner = MultiTimeframeScanner()
        self.option_chart_skill = OptionChartSkill()
        self.gamma_hunter_skill = ExpiryGammaHunterSkill()
        self.tradingview_skill = TradingViewAnalystSkill()
        self.dynamic_registry = DynamicSkillRegistry()

    def load_all_models(self, underlying: Optional[str] = None) -> Tuple[bool, bool, bool]:
        """Loads pre-trained brains (SVM, XGBoost, Deep ANN, and Vision CNN) from disk, with asset-specific fallback"""
        from pathlib import Path
        models_dir = Path(__file__).resolve().parent / "models"
        root_dir = Path(__file__).resolve().parent.parent
        
        asset = (underlying or "").strip().lower()
        svm_path = None
        xgb_path = None
        ann_path = None

        if asset in ["nifty", "sensex", "banknifty"]:
            p_svm = models_dir / f"svm_model_{asset}.joblib"
            p_xgb = models_dir / f"xgb_model_{asset}.joblib"
            p_ann = models_dir / f"ann_model_{asset}.joblib"
            if p_svm.exists(): svm_path = p_svm
            if p_xgb.exists(): xgb_path = p_xgb
            if p_ann.exists(): ann_path = p_ann

        svm_ok = self.svm.load_model(svm_path)
        xgb_ok = self.xgb.load_model(xgb_path)
        ann_ok = self.ann.load_model(ann_path)

        # Asset-aware Vision Model loader
        if VisualPredictor is not None:
            v_path = None
            if asset in ["nifty", "sensex", "banknifty"]:
                pv = root_dir / "vision_chart_brain" / f"vision_cnn_{asset}.pt"
                if pv.exists():
                    v_path = str(pv)
            if v_path is not None:
                self.vision = VisualPredictor(model_path=v_path)
            elif self.vision is None:
                self.vision = VisualPredictor()

        return svm_ok, xgb_ok, ann_ok

    def update_bar(
        self,
        current_price: float,
        price_delta: float,
        rolling_vol: float,
        garch_forecast: float,
        vpin: float,
        macro_state: MacroFeatureState
    ):
        """Feeds new bar into SVM, XGBoost, and ANN models"""
        # 1. Update SVM (3 micro features)
        self.svm.update_bar(
            current_price=current_price,
            price_delta=price_delta,
            rolling_vol=rolling_vol,
            garch_forecast=garch_forecast
        )

        # 2. Update XGBoost (8 multi-timeframe features)
        xgb_vec = self.xgb.build_feature_vector(
            price_delta=price_delta,
            rolling_vol=rolling_vol,
            garch_forecast=garch_forecast,
            vpin=vpin,
            dist_pdh=macro_state.dist_pdh_pct,
            dist_pdl=macro_state.dist_pdl_pct,
            week_pos=macro_state.week_range_position,
            trend_15m=macro_state.trend_15m_bias
        )
        self.xgb.update_bar(current_price=current_price, feature_vector=xgb_vec)

        # 3. Update Deep ANN Neural Network
        self.ann.update_bar(
            current_price=current_price,
            price_delta=price_delta,
            rolling_vol=rolling_vol,
            garch_forecast=garch_forecast,
            vpin=vpin,
            dist_pdh=macro_state.dist_pdh_pct,
            dist_pdl=macro_state.dist_pdl_pct,
            week_pos=macro_state.week_range_position,
            trend_15m=macro_state.trend_15m_bias
        )

    def train_models(self) -> Tuple[bool, bool, bool]:
        """Forces training across all 3 machine learning models"""
        svm_trained = self.svm.train_model()
        xgb_trained = self.xgb.train_model()
        ann_trained = self.ann.train() if hasattr(self.ann, "train") else False
        return svm_trained, xgb_trained, self.ann.is_trained

    def evaluate(
        self,
        garch_signal: DirectionalSignal,
        price_delta: float,
        rolling_vol: float,
        garch_forecast: float,
        vpin: float,
        macro_state: MacroFeatureState,
        recent_bars: Optional[List[Dict[str, float]]] = None,
        smc_state: Optional[Any] = None,
        # Chinmay Trading Skills parameters
        open_p: Optional[float] = None,
        high_p: Optional[float] = None,
        low_p: Optional[float] = None,
        close_p: Optional[float] = None,
        prev_open: Optional[float] = None,
        prev_close: Optional[float] = None,
        recent_highs: Optional[List[float]] = None,
        recent_lows: Optional[List[float]] = None,
        key_levels: Optional[List[float]] = None,
        underlying: str = "NIFTY",
        current_option_price: Optional[float] = None,
        pattern_st: Optional[Any] = None
    ) -> EnsembleDecision:
        """
        Evaluates 4-Brain Consensus (SVM + XGBoost + Deep ANN + 2D CNN Vision) with SMC & Macro Guardrails.
        """
        # Run Vision 2D CNN Inference if available
        v_action = "HOLD"
        v_conf = 0.50
        v_probs = {"CALL": 0.33, "PUT": 0.33, "HOLD": 0.34}
        v_num = 0

        if self.vision is not None and recent_bars and len(recent_bars) >= 15:
            try:
                v_res = self.vision.predict_chart(recent_bars)
                v_action = v_res.get("action", "HOLD")
                v_conf = v_res.get("confidence", 0.50)
                v_probs = v_res.get("probabilities", v_probs)
                v_num = v_res.get("numeric_signal", 0)
            except Exception as e:
                logger.debug(f"[Vision Brain Inference Error]: {e}")

        # Direct Wave-Based M/W & Trendline Skill Check
        if (pattern_st is None or getattr(pattern_st, "bias", "") == "NEUTRAL") and recent_bars and len(recent_bars) >= 8:
            try:
                wave_res = self.m_pattern_skill.evaluate(recent_bars, current_price=current_price)
                if wave_res.is_favorable and wave_res.signal in ("BUY_CALL", "BUY_PUT"):
                    bias = "BULLISH" if wave_res.signal == "BUY_CALL" else "BEARISH"
                    from .pattern_engine import DetectedPattern, PatternType
                    p_type = PatternType.DOUBLE_BOTTOM_W if bias == "BULLISH" else PatternType.DOUBLE_TOP_M
                    pattern_st = DetectedPattern(
                        pattern_type=p_type,
                        bias=bias,
                        confidence=wave_res.confidence,
                        neckline_level=wave_res.metadata.get("neckline", current_price),
                        target_price=wave_res.metadata.get("target_1x", current_price),
                        stop_loss_level=wave_res.metadata.get("stop_loss", current_price),
                        description=wave_res.reason
                    )
            except Exception as e:
                logger.debug(f"Direct MPatternSkill evaluation: {e}")

        # Classical Chart Pattern check (Double Bottom W, Double Top M, Trendline Breakout)
        has_chart_pattern = (
            pattern_st is not None 
            and getattr(pattern_st, "confidence", 0.0) >= 0.70 
            and getattr(pattern_st, "bias", "") in ("BULLISH", "BEARISH")
        )

        effective_signal = garch_signal
        pattern_override_note = ""

        # PATTERN SOVEREIGNTY: Structural patterns (W/M/Trendlines) override lagging statistical models
        if has_chart_pattern:
            pattern_bias = getattr(pattern_st, "bias", "")
            p_desc = getattr(pattern_st, "description", "")
            p_type_val = getattr(getattr(pattern_st, "pattern_type", ""), "value", "CHART_PATTERN")

            if pattern_bias == "BULLISH":
                # VETO ANY BEARISH / SELL (BUY PUT) SETUP
                if effective_signal == DirectionalSignal.SELL:
                    logger.warning(f"[CHART PATTERN VETO] Bearish SELL vetoed by {p_type_val} ({p_desc})! Trend is reversing Bullish.")
                effective_signal = DirectionalSignal.BUY
                pattern_override_note = f" [Pattern Trigger: {p_desc}]"
                logger.info(f"[Brain Pattern Authority] {p_type_val} established BULLISH (BUY CALL) dominance!")

            elif pattern_bias == "BEARISH":
                # VETO ANY BULLISH / BUY (BUY CALL) SETUP
                if effective_signal == DirectionalSignal.BUY:
                    logger.warning(f"[CHART PATTERN VETO] Bullish BUY vetoed by {p_type_val} ({p_desc})! Trend is reversing Bearish.")
                
                # Special Guard: If trapped in tight compression or rejected at trendline ceiling
                if "RESISTANCE" in str(p_type_val).upper() or "CEILING" in str(p_desc).upper():
                    logger.warning(f"[TRENDLINE CEILING VETO] BUY CALL strictly prohibited under descending trendline ceiling ({p_desc}).")
                    effective_signal = DirectionalSignal.HOLD

                effective_signal = DirectionalSignal.SELL if effective_signal != DirectionalSignal.HOLD else DirectionalSignal.HOLD
                pattern_override_note = f" [Pattern Trigger: {p_desc}]"
                logger.info(f"[Brain Pattern Authority] {p_type_val} established BEARISH dominance / Trendline Ceiling Guard!")

        if effective_signal == DirectionalSignal.HOLD:
            return EnsembleDecision(
                is_vetoed=False,
                final_action=DirectionalSignal.HOLD,
                svm_prediction=0,
                xgb_prediction=0,
                xgb_confidence=0.50,
                reason="GARCH momentum is neutral (HOLD)",
                macro_state=macro_state,
                ann_prediction=0,
                ann_confidence=0.50,
                tri_is_vetoed=False,
                tri_final_action=DirectionalSignal.HOLD,
                tri_reason="GARCH momentum is neutral (HOLD)",
                vision_action=v_action,
                vision_confidence=v_conf,
                vision_probs=v_probs,
                vision_numeric_signal=v_num
            )

        # SMC Parameters Extraction
        smc_t_val = 0
        smc_bos = 0
        smc_choch = 0
        smc_range_pos = 0.5
        fvg_bias = 0
        sweep_bias = 0

        if smc_state is not None:
            smc_t_val = 1 if getattr(smc_state.trend, "value", "") == "BULLISH_STRUCTURE" else (-1 if getattr(smc_state.trend, "value", "") == "BEARISH_STRUCTURE" else 0)
            smc_bos = getattr(smc_state, "bos_direction", 0)
            smc_choch = getattr(smc_state, "choch_direction", 0)
            smc_range_pos = getattr(smc_state, "dealing_range_pos", 0.5)
            if hasattr(smc_state, "active_fvgs") and smc_state.active_fvgs:
                fvg_bias = 1 if getattr(smc_state.active_fvgs[-1].gap_type, "value", "") == "BULLISH_FVG" else -1
            if hasattr(smc_state, "liquidity_sweep") and smc_state.liquidity_sweep:
                sweep_bias = 1 if "BULLISH" in smc_state.liquidity_sweep else -1

        # --- RULE 1: Macro & SMC Dealing Range Guardrails ---
        # Allow intraday breakout expansions; only block when at extreme multi-day + intraday exhaustion (>99%)
        if garch_signal == DirectionalSignal.BUY and smc_range_pos >= 0.99 and macro_state.week_range_position >= 0.99:
            return EnsembleDecision(
                is_vetoed=True,
                final_action=DirectionalSignal.HOLD,
                svm_prediction=1,
                xgb_prediction=1,
                xgb_confidence=0.50,
                reason=f"SMC/MACRO VETO: Price at extreme Exhaustion ceiling (Range Pos: {smc_range_pos*100:.0f}%)",
                macro_state=macro_state,
                ann_prediction=0,
                ann_confidence=0.50,
                tri_is_vetoed=True,
                tri_final_action=DirectionalSignal.HOLD,
                tri_reason=f"SMC/MACRO VETO: Price at extreme Exhaustion ceiling (Range Pos: {smc_range_pos*100:.0f}%)",
                vision_action=v_action,
                vision_confidence=v_conf,
                vision_probs=v_probs,
                vision_numeric_signal=v_num
            )

        if garch_signal == DirectionalSignal.SELL and smc_range_pos <= 0.01 and macro_state.week_range_position <= 0.01:
            return EnsembleDecision(
                is_vetoed=True,
                final_action=DirectionalSignal.HOLD,
                svm_prediction=-1,
                xgb_prediction=-1,
                xgb_confidence=0.50,
                reason=f"SMC/MACRO VETO: Price at extreme Exhaustion floor (Range Pos: {smc_range_pos*100:.0f}%)",
                macro_state=macro_state,
                ann_prediction=0,
                ann_confidence=0.50,
                tri_is_vetoed=True,
                tri_final_action=DirectionalSignal.HOLD,
                tri_reason=f"SMC/MACRO VETO: Price at extreme Exhaustion floor (Range Pos: {smc_range_pos*100:.0f}%)",
                vision_action=v_action,
                vision_confidence=v_conf,
                vision_probs=v_probs,
                vision_numeric_signal=v_num
            )

        # --- MODEL PREDICTIONS ---
        # 1. SVM RBF Prediction
        svm_feat = self.svm.build_feature_vector(price_delta, rolling_vol, garch_forecast)
        svm_pred, svm_dist = self.svm.predict(svm_feat)

        # 2. XGBoost Tree Prediction (14-dim with SMC)
        xgb_vec = self.xgb.build_feature_vector(
            price_delta=price_delta,
            rolling_vol=rolling_vol,
            garch_forecast=garch_forecast,
            vpin=vpin,
            dist_pdh=macro_state.dist_pdh_pct,
            dist_pdl=macro_state.dist_pdl_pct,
            week_pos=macro_state.week_range_position,
            trend_15m=macro_state.trend_15m_bias,
            smc_trend=smc_t_val,
            smc_bos=smc_bos,
            smc_choch=smc_choch,
            smc_range_pos=smc_range_pos,
            smc_fvg_bias=fvg_bias,
            smc_sweep_bias=sweep_bias
        )
        xgb_pred, xgb_conf = self.xgb.predict(xgb_vec)

        # 3. ANN Deep Neural Network Prediction (14-dim with SMC)
        ann_vec = self.ann.build_feature_vector(
            price_delta=price_delta,
            rolling_vol=rolling_vol,
            garch_forecast=garch_forecast,
            vpin=vpin,
            dist_pdh=macro_state.dist_pdh_pct,
            dist_pdl=macro_state.dist_pdl_pct,
            week_pos=macro_state.week_range_position,
            trend_15m=macro_state.trend_15m_bias,
            smc_trend=smc_t_val,
            smc_bos=smc_bos,
            smc_choch=smc_choch,
            smc_range_pos=smc_range_pos,
            smc_fvg_bias=fvg_bias,
            smc_sweep_bias=sweep_bias
        )
        ann_pred, ann_conf, _ = self.ann.predict(ann_vec)

        # --- TRACK 1: Standard Dual Consensus (SVM + XGB) ---
        dual_veto = False
        dual_action = effective_signal
        dual_reason = f"APPROVED by Standard Dual-Brain (SVM + XGB){pattern_override_note}"

        if effective_signal == DirectionalSignal.BUY:
            if svm_pred == -1 and not has_chart_pattern:
                dual_veto = True
                dual_action = DirectionalSignal.HOLD
                dual_reason = "VETO by SVM: BUY signal but SVM predicted Bearish (-1)"
            elif xgb_pred == -1 and xgb_conf >= self.config.xgb_min_prob_threshold and not has_chart_pattern:
                dual_veto = True
                dual_action = DirectionalSignal.HOLD
                dual_reason = f"VETO by XGBoost: BUY signal but XGBoost confidence is {xgb_conf*100:.1f}% Bearish"
        elif effective_signal == DirectionalSignal.SELL:
            if svm_pred == 1 and not has_chart_pattern:
                dual_veto = True
                dual_action = DirectionalSignal.HOLD
                dual_reason = "VETO by SVM: SELL signal but SVM predicted Bullish (+1)"
            elif xgb_pred == 1 and xgb_conf >= self.config.xgb_min_prob_threshold and not has_chart_pattern:
                dual_veto = True
                dual_action = DirectionalSignal.HOLD
                dual_reason = f"VETO by XGBoost: SELL signal but XGBoost confidence is {xgb_conf*100:.1f}% Bullish"

        # --- TRACK 2: Tri-Brain Consensus (SVM + XGB + ANN Dual Alignment) ---
        tri_veto = False
        tri_action = effective_signal
        tri_reason = f"APPROVED by Tri-Brain Consensus (ANN + XGB + SVM){pattern_override_note}"

        if effective_signal == DirectionalSignal.BUY:
            if dual_veto or (ann_pred == -1 and ann_conf >= 0.55 and not has_chart_pattern):
                tri_veto = True
                tri_action = DirectionalSignal.HOLD
                tri_reason = f"VETO by Tri-Brain: Opposing Bearish ANN={ann_pred} ({ann_conf*100:.0f}%), DualVeto={dual_veto}"
        elif effective_signal == DirectionalSignal.SELL:
            if dual_veto or (ann_pred == 1 and ann_conf >= 0.55 and not has_chart_pattern):
                tri_veto = True
                tri_action = DirectionalSignal.HOLD
                tri_reason = f"VETO by Tri-Brain: Opposing Bullish ANN={ann_pred} ({ann_conf*100:.0f}%), DualVeto={dual_veto}"

        # --- CHINMAY TRADING SKILLS ENFORCEMENT ---
        pa_result = None
        strike_result = None
        rr_result = None

        target_act = dual_action if dual_action != DirectionalSignal.HOLD else tri_action
        if target_act in (DirectionalSignal.BUY, DirectionalSignal.SELL) and close_p is not None:
            act_str = "BUY" if target_act == DirectionalSignal.BUY else "SELL"

            # 1. Price Action Skill
            if open_p is not None and high_p is not None and low_p is not None:
                pa_result = self.price_action_skill.evaluate(
                    open_p=open_p,
                    high_p=high_p,
                    low_p=low_p,
                    close_p=close_p,
                    prev_open=prev_open,
                    prev_close=prev_close,
                    recent_highs=recent_highs,
                    recent_lows=recent_lows,
                    key_levels=key_levels,
                    intended_direction=act_str,
                    chart_pattern=pattern_st
                )
                if not pa_result.is_favorable:
                    dual_veto = True
                    dual_action = DirectionalSignal.HOLD
                    dual_reason = f"SKILL VETO (PriceAction): {pa_result.reason}"
                    tri_veto = True
                    tri_action = DirectionalSignal.HOLD
                    tri_reason = f"SKILL VETO (PriceAction): {pa_result.reason}"

            # 2. Risk-Reward Skill (Min 1:1.5 R:R requirement)
            if not dual_veto or not tri_veto:
                opt_p = current_option_price if current_option_price is not None else 100.0
                rr_result = self.risk_reward_skill.evaluate(
                    spot_entry=close_p,
                    action=act_str,
                    option_entry=opt_p,
                    candle_high=high_p,
                    candle_low=low_p,
                    nearest_resistance=key_levels[-1] if key_levels else None,
                    nearest_support=key_levels[0] if key_levels else None
                )
                if not rr_result.is_favorable:
                    dual_veto = True
                    dual_action = DirectionalSignal.HOLD
                    dual_reason = f"SKILL VETO (RiskReward): {rr_result.reason}"
                    tri_veto = True
                    tri_action = DirectionalSignal.HOLD
                    tri_reason = f"SKILL VETO (RiskReward): {rr_result.reason}"

            # 3. Strike Selector Skill
            strike_result = self.strike_selector_skill.evaluate(
                underlying=underlying,
                spot_price=close_p,
                action=act_str
            )

            # 4. Evaluate Dynamically Loaded Skills from Registry
            if not dual_veto or not tri_veto:
                dynamic_evals = self.dynamic_registry.evaluate_all(
                    open_p=open_p, high_p=high_p, low_p=low_p, close_p=close_p,
                    recent_bars=recent_bars, intended_direction=act_str
                )
                for skill_name, dyn_res in dynamic_evals.items():
                    if not dyn_res.is_favorable:
                        dual_veto = True
                        dual_action = DirectionalSignal.HOLD
                        dual_reason = f"DYNAMIC SKILL VETO ({skill_name}): {dyn_res.reason}"
                        tri_veto = True
                        tri_action = DirectionalSignal.HOLD
                        tri_reason = f"DYNAMIC SKILL VETO ({skill_name}): {dyn_res.reason}"
                        break

        return EnsembleDecision(
            is_vetoed=dual_veto,
            final_action=dual_action,
            svm_prediction=svm_pred,
            xgb_prediction=xgb_pred,
            xgb_confidence=xgb_conf,
            reason=dual_reason,
            macro_state=macro_state,
            ann_prediction=ann_pred,
            ann_confidence=ann_conf,
            tri_is_vetoed=tri_veto,
            tri_final_action=tri_action,
            tri_reason=tri_reason,
            vision_action=v_action,
            vision_confidence=v_conf,
            vision_probs=v_probs,
            vision_numeric_signal=v_num,
            price_action_result=pa_result,
            strike_result=strike_result,
            risk_reward_result=rr_result
        )
