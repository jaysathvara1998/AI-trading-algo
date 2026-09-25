"""
Algo VPIN v2.0 - Multi-Timeframe Quantitative Options Trading Engine
Orchestrates:
1. Macro Feature Engine: PDH, PDL, Weekly Range Position, 15-min Trend
2. Layer 1: Bulk Volume Classification & VPIN Toxicity (Volume Clock)
3. Layer 2: Rolling GARCH(1,1) Direction & Volatility Forecaster
4. Layer 3: Ensemble Brain (SVM RBF + XGBoost Trees Consensus)
5. Risk Manager: ₹50k Capital Allocation, Volatility Stops, 15:24 IST Cutoff
6. Execution: Real-Time Dhan Options Buying Router
"""

import sys
from pathlib import Path

# Add project root directory to path for standalone script execution
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Ensure robust UTF-8 console output on Windows with immediate flush
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)


import asyncio
import argparse
from collections import deque
from datetime import datetime
import logging
import pytz
import numpy as np
import pandas as pd

from algo_vpin_v2.config import AppConfig, CONFIG, InstrumentMode, ExecutionMode, TradeStrategyMode
from algo_vpin_v2.data_feed import DhanDataFeed, BarOHLCV
from algo_vpin_v2.vpin import VPINCalculator, ToxicityRegime
from algo_vpin_v2.garch_engine import GARCHEngine, DirectionalSignal
from algo_vpin_v2.macro_features import MacroFeatureEngine
from algo_vpin_v2.ensemble_brain import EnsembleBrain
from algo_vpin_v2.mtf_analyst import MultiTimeframeAnalyst
from algo_vpin_v2.risk_manager import RiskManager, PositionSide
from algo_vpin_v2.execution import OrderExecutionRouter, OrderType
from algo_vpin_v2.humanoid_agent import HumanoidTraderAgent, MarketPhase
from algo_vpin_v2.skills import (
    DailyReflectionSkill, PatternConfirmationWatcher, PatternState, 
    DynamicSkillRegistry, PreTradeSanitySkill, PostTradeDebriefSkill
)
from algo_vpin_v2.auto_tuner import AutoTuningEngine
from algo_vpin_v2.agents import SkillGeneratorAgent
from algo_vpin_v2.smc_engine import SMCEngine
from algo_vpin_v2.global_market_engine import GlobalMarketEngine, GlobalMarketState
from algo_vpin_v2.news_filter import NewsFilter, NewsEventState
from algo_vpin_v2.heavyweight_scanner import HeavyweightScanner, HeavyweightBreadthState
from algo_vpin_v2.pattern_engine import PatternRecognitionEngine, DetectedPattern
from algo_vpin_v2.shadow_learner import VirtualShadowLearner


class FlushStreamHandler(logging.StreamHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s",
    handlers=[FlushStreamHandler(sys.stdout)]
)
logger = logging.getLogger("algo_vpin_v2.main")


class AlgoVPINRunner:
    def __init__(self, config: AppConfig = CONFIG):
        self.config = config
        self.tz = pytz.timezone("Asia/Kolkata")
        
        # Humanoid Cognitive Agent & Pipeline Components
        self.humanoid = HumanoidTraderAgent(agent_name="AURA-v2 (Humanoid Quant)")
        self.data_feed = DhanDataFeed(self.config)
        self.vpin_calc = VPINCalculator(self.config.vpin)
        self.garch_engine = GARCHEngine(self.config.garch)
        self.macro_engine = MacroFeatureEngine()
        self.smc_engine = SMCEngine()
        self.pattern_engine = PatternRecognitionEngine()
        self.global_market_engine = GlobalMarketEngine()
        self.news_filter = NewsFilter()
        self.heavyweight_scanner = HeavyweightScanner(self.data_feed.tradehull_client)
        self.ensemble_brain = EnsembleBrain(self.config.ensemble)
        self.mtf_analyst = MultiTimeframeAnalyst()

        # Autonomous Self-Learning & Trading Skills (marian2js/trading-skills enhanced)
        self.auto_tuner = AutoTuningEngine()
        self.pattern_watcher = PatternConfirmationWatcher()
        self.skill_registry = DynamicSkillRegistry()
        self.skill_generator = SkillGeneratorAgent()
        self.daily_reflection_skill = DailyReflectionSkill()
        self.pre_trade_sanity_skill = PreTradeSanitySkill()
        self.post_trade_debrief_skill = PostTradeDebriefSkill()
        self.shadow_learner = VirtualShadowLearner()

        # Apply adaptive config parameters
        self.config.risk.max_daily_trades = self.auto_tuner.state.max_daily_trades
        self.config.ensemble.xgb_min_prob_threshold = self.auto_tuner.state.xgb_min_prob_threshold

        self.risk_manager = RiskManager(self.config.risk, self.config.market, track_name="Track 1: Dual-Brain")
        self.execution = OrderExecutionRouter(
            self.config,
            self.risk_manager,
            self.data_feed.tradehull_client,
            track_label="Track 1: Dual-Brain",
            data_feed=self.data_feed
        )

        # Track 2: Tri-Brain with Deep Neural Network (Side-by-Side Comparison Default: True)
        self.compare_ann: bool = True
        self.enable_tri_brain_comparison()
        self.total_bars_processed: int = 0
        self.is_warmed_up: bool = False
        self.recent_volumes = deque(maxlen=10)
        self.recent_bars_deque = deque(maxlen=30)
        self.opt_bars_tracker: dict = {}

    def enable_tri_brain_comparison(self):
        """Initializes second independent risk manager and execution router for Track 2 (ANN Tri-Brain)"""
        self.compare_ann = True
        self.tri_risk_manager = RiskManager(self.config.risk, self.config.market, track_name="Track 2: Tri-Brain ANN")
        self.tri_execution = OrderExecutionRouter(
            self.config,
            self.tri_risk_manager,
            self.data_feed.tradehull_client,
            track_label="Track 2: Tri-Brain ANN",
            data_feed=self.data_feed
        )
        logger.info("[COMPARISON ENGINE] Track 2 (ANN Tri-Brain) enabled for side-by-side live evaluation.")

    def warm_up(self, historical_bars: pd.DataFrame = None):
        logger.info(f"Initializing v2 system warm-up phase ({self.config.market.underlying})...")
        if historical_bars is None or historical_bars.empty:
            historical_bars = self.data_feed.fetch_historical_bars(days=5, symbol=self.config.market.underlying)

        logger.info(f"Ingesting {len(historical_bars)} historical bars for macro & brain warm-up...")

        # Seed recent bars deque for Vision Brain
        if not historical_bars.empty:
            for _, r in historical_bars.tail(30).iterrows():
                self.recent_bars_deque.append({
                    "open": float(r.get("open", r["close"])),
                    "high": float(r.get("high", r["close"])),
                    "low": float(r.get("low", r["close"])),
                    "close": float(r["close"]),
                    "volume": float(r.get("volume", 0.0))
                })

        # 1. Initialize Macro Engine
        self.macro_engine.initialize_from_history(historical_bars)

        # 2. Calibrate VPIN Bucket Volume V
        if len(historical_bars) >= 375:
            avg_daily_vol = float(historical_bars["volume"].sum()) / (len(historical_bars) / 375.0)
            self.vpin_calc.update_bucket_volume(avg_daily_vol)
            logger.info(f"Calibrated adaptive bucket volume V = {self.vpin_calc.bucket_volume:.1f}")

        # 3. Load pre-trained models if available (asset-aware for NIFTY / SENSEX)
        svm_ok, xgb_ok, ann_ok = self.ensemble_brain.load_all_models(self.config.market.underlying)
        if svm_ok and xgb_ok:
            logger.info(f"[Ensemble Brain] Models loaded from disk for {self.config.market.underlying}: SVM={svm_ok}, XGBoost={xgb_ok}, ANN={ann_ok}. Fast warm-up engaged.")

        # Sequential bar ingestion without repetitive intermediate retraining
        for idx, row in historical_bars.iterrows():
            ts = row.get("timestamp", pd.Timestamp.now(tz=self.tz))
            close_p = float(row["close"])
            open_p = float(row.get("open", close_p))
            high_p = float(row.get("high", close_p))
            low_p = float(row.get("low", close_p))
            vol = float(row["volume"])

            macro_st = self.macro_engine.update_1min_bar(close_p, high_p, low_p)
            vpin_res = self.vpin_calc.process_bar(close_p, vol, ts)
            garch_res = self.garch_engine.add_bar(close_p, ts)
            smc_st = self.smc_engine.update_bar(open_p, high_p, low_p, close_p, vol)

            price_delta = close_p - (historical_bars.iloc[idx-1]["close"] if idx > 0 else close_p)
            vol_roll = garch_res.sigma_next if garch_res else 0.001
            garch_pred = garch_res.mu_next if garch_res else 0.0

            # Feed features into brain history (14-dim SMC)
            smc_t_val = 1 if getattr(smc_st.trend, "value", "") == "BULLISH_STRUCTURE" else (-1 if getattr(smc_st.trend, "value", "") == "BEARISH_STRUCTURE" else 0)
            fvg_bias = 0
            if smc_st.active_fvgs:
                fvg_bias = 1 if getattr(smc_st.active_fvgs[-1].gap_type, "value", "") == "BULLISH_FVG" else -1
            sweep_bias = 0
            if smc_st.liquidity_sweep:
                sweep_bias = 1 if "BULLISH" in smc_st.liquidity_sweep else -1

            svm_feat = self.ensemble_brain.svm.build_feature_vector(price_delta, vol_roll, garch_pred)
            xgb_feat = self.ensemble_brain.xgb.build_feature_vector(
                price_delta=price_delta,
                rolling_vol=vol_roll,
                garch_forecast=garch_pred,
                vpin=vpin_res.vpin,
                dist_pdh=macro_st.dist_pdh_pct,
                dist_pdl=macro_st.dist_pdl_pct,
                week_pos=macro_st.week_range_position,
                trend_15m=macro_st.trend_15m_bias,
                smc_trend=smc_t_val,
                smc_bos=smc_st.bos_direction,
                smc_choch=smc_st.choch_direction,
                smc_range_pos=smc_st.dealing_range_pos,
                smc_fvg_bias=fvg_bias,
                smc_sweep_bias=sweep_bias
            )
            self.ensemble_brain.svm.feature_history.append(svm_feat)
            self.ensemble_brain.svm.target_history.append(1 if price_delta >= 0 else -1)
            self.ensemble_brain.xgb.feature_history.append(xgb_feat)
            self.ensemble_brain.xgb.target_history.append(1 if price_delta >= 0 else 0)

            ann_lbl = 2 if price_delta > 2.5 else (0 if price_delta < -2.5 else 1)
            self.ensemble_brain.ann.feature_history.append(xgb_feat)
            self.ensemble_brain.ann.target_history.append(ann_lbl)

        # If models were NOT pre-trained on disk, train them now
        if not (svm_ok and xgb_ok and ann_ok):
            svm_tr, xgb_tr, ann_tr = self.ensemble_brain.train_models()
            logger.info(f"Warm-up complete. Brain models trained: SVM={svm_tr}, XGBoost={xgb_tr}, ANN={ann_tr}. Completed VPIN buckets: {len(self.vpin_calc.completed_buckets)}")
        else:
            logger.info(f"Warm-up complete. High-precision pre-trained brains active for {self.config.market.underlying}. Completed VPIN buckets: {len(self.vpin_calc.completed_buckets)}")

        # Initialize Pre-Market Journal (Pillar 4)
        last_close = float(historical_bars.iloc[-1]["close"]) if not historical_bars.empty else 0.0
        pdh_val = self.macro_engine.pdh or last_close
        pdl_val = self.macro_engine.pdl or last_close
        pdc_val = float(historical_bars.iloc[-375]["close"]) if len(historical_bars) >= 375 else last_close
        
        global_st = self.global_market_engine.evaluate_premarket_cues(
            current_spot=last_close,
            prev_close=pdc_val,
            symbol=self.config.market.underlying
        )
        news_st = self.news_filter.check_news_risk()
        hw_st = self.heavyweight_scanner.evaluate_breadth()
        
        self.risk_manager.initialize_premarket_journal(
            pdh=pdh_val,
            pdl=pdl_val,
            pdc=pdc_val,
            global_sentiment=global_st.sentiment.value,
            heavyweight_bias=hw_st.bias.value,
            news_risk_status=news_st.risk_level.value
        )
        if self.compare_ann and self.tri_risk_manager:
            self.tri_risk_manager.initialize_premarket_journal(
                pdh=pdh_val,
                pdl=pdl_val,
                pdc=pdc_val,
                global_sentiment=global_st.sentiment.value,
                heavyweight_bias=hw_st.bias.value,
                news_risk_status=news_st.risk_level.value
            )

        # Generate Pre-Market Strategic Plan from Gemini AI + Key Levels
        strat_plan = self.daily_reflection_skill.generate_premarket_strategic_plan(
            symbol=self.config.market.underlying,
            current_spot=last_close,
            pdh=pdh_val,
            pdl=pdl_val,
            pdc=pdc_val,
            global_sentiment=global_st.sentiment.value,
            heavyweight_bias=hw_st.bias.value
        )
        morning_brief = self.daily_reflection_skill.get_morning_briefing()
        active_skills = self.skill_registry.get_active_skills()

        logger.info("=" * 80)
        logger.info("       STARTUP MARKET ANALYSIS, MEMORY BRIEFING & ACTIVE SKILLS         ")
        logger.info("=" * 80)
        logger.info(f"  MARKET REGIME  : Underlying={self.config.market.underlying} | Spot={last_close:.1f} | PDC={pdc_val:.1f}")
        logger.info(f"  KEY LEVELS     : PDH={pdh_val:.1f} | PDL={pdl_val:.1f}")
        logger.info(f"  SENTIMENT CUES : Global={global_st.sentiment.value} | Heavyweights={hw_st.bias.value}")
        logger.info("-" * 80)
        logger.info(f"  PRIOR MEMORY   : {morning_brief}")
        logger.info(f"  STRATEGIC PLAN : {strat_plan}")
        logger.info("-" * 80)
        logger.info(f"  ACTIVE SKILLS ({len(active_skills)} Registered & Ready):")
        for sk_name in active_skills.keys():
            logger.info(f"    [✓] {sk_name}")
        logger.info("=" * 80)

        self.is_warmed_up = True

    def _record_bar_to_csv(self, bar: BarOHLCV, vpin_res, garch_res, ensemble_dec, macro_st):
        try:
            date_str = datetime.now(self.tz).strftime("%Y%m%d")
            data_dir = ROOT_DIR / "algo_vpin_v2" / "data"
            data_dir.mkdir(parents=True, exist_ok=True)
            csv_path = data_dir / f"{self.config.market.underlying.lower()}_v2_1min_{date_str}.csv"

            row = {
                "timestamp": str(bar.timestamp),
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume,
                "vpin": round(vpin_res.vpin, 4),
                "vpin_regime": vpin_res.regime.value,
                "garch_mu": round(garch_res.mu_next, 6),
                "garch_vol": round(garch_res.annualized_vol, 4),
                "garch_signal": garch_res.signal.value,
                "svm_pred": ensemble_dec.svm_prediction,
                "xgb_pred": ensemble_dec.xgb_prediction,
                "xgb_conf": round(ensemble_dec.xgb_confidence, 4),
                "week_pos": round(macro_st.week_range_position, 3),
                "trend_15m": macro_st.trend_15m_bias,
                "decision": "VETO" if ensemble_dec.is_vetoed else "APPROVED"
            }
            df_row = pd.DataFrame([row])
            df_row.to_csv(csv_path, mode="a", header=not csv_path.exists(), index=False)
        except Exception as e:
            logger.debug(f"Could not record bar to CSV: {e}")

    def process_incoming_bar(self, bar: BarOHLCV):
        self.total_bars_processed += 1
        curr_price = bar.close
        high_price = bar.high
        low_price = bar.low
        timestamp = bar.timestamp

        # --- STEP 1: Multi-Timeframe Momentum Update & Exit Check ---
        mtf_st = self.mtf_analyst.update_bar(curr_price, timestamp=timestamp)
        pos = self.risk_manager.current_position
        curr_opt_premium = None
        if pos.side != PositionSide.FLAT and pos.is_option and pos.symbol:
            curr_opt_premium = self.data_feed.get_ltp(pos.symbol, curr_price)

        # Check Multi-Bar Cumulative Counter-Trend (e.g. 5m spot surge against PUT or dump against CALL)
        holding_call = pos.is_option and pos.symbol and ("CALL" in pos.symbol or "CE" in pos.symbol)
        holding_put = pos.is_option and pos.symbol and ("PUT" in pos.symbol or "PE" in pos.symbol)
        is_counter, counter_msg = self.mtf_analyst.check_cumulative_counter_trend(
            holding_call=holding_call,
            holding_put=holding_put,
            threshold_pts=20.0
        )

        exit_needed, exit_reason = self.risk_manager.check_exit_conditions(
            current_price=curr_price,
            option_premium=curr_opt_premium,
            current_time=timestamp.to_pydatetime() if hasattr(timestamp, "to_pydatetime") else None,
            counter_trend_reason=counter_msg if is_counter else None
        )
        if exit_needed:
            exit_exec_price = curr_opt_premium if (pos.is_option and curr_opt_premium) else curr_price
            self.execution.execute_exit(current_price=exit_exec_price, reason=exit_reason)

        # --- STEP 2: Macro Levels & Microstructure & Smart Money Concepts (SMC) & 4 Pillars ---
        macro_st = self.macro_engine.update_1min_bar(curr_price, high_price, low_price)
        vpin_res = self.vpin_calc.process_bar(close_price=curr_price, volume=bar.volume, timestamp=timestamp)
        garch_res = self.garch_engine.add_bar(close_price=curr_price, timestamp=timestamp)
        smc_st = self.smc_engine.update_bar(bar.open, bar.high, bar.low, bar.close, bar.volume)
        pattern_st = self.pattern_engine.update_bar(bar.open, bar.high, bar.low, bar.close, bar.volume)

        # 4 Pre-Market Master Pillars Updates
        global_st = self.global_market_engine.evaluate_premarket_cues(
            current_spot=curr_price,
            prev_close=self.macro_engine.pdc or curr_price,
            symbol=self.config.market.underlying
        )
        news_st = self.news_filter.check_news_risk(
            now=timestamp.to_pydatetime() if hasattr(timestamp, "to_pydatetime") else None
        )
        is_near_pdh = abs(macro_st.dist_pdh_pct) < 0.0015
        is_near_pdl = abs(macro_st.dist_pdl_pct) < 0.0015
        if self.total_bars_processed % 3 == 0:
            self.heavyweight_scanner.poll_live_heavyweights()
        heavyweight_st = self.heavyweight_scanner.evaluate_breadth(
            spot_trend_bias=macro_st.trend_15m_bias,
            is_nifty_near_pdh=is_near_pdh,
            is_nifty_near_pdl=is_near_pdl
        )

        # Append bar to recent history for Vision CNN
        self.recent_bars_deque.append({
            "open": bar.open,
            "high": bar.high,
            "low": bar.low,
            "close": bar.close,
            "volume": bar.volume
        })

        if garch_res is None:
            return

        price_delta = curr_price - (self.garch_engine.prices[-2] if len(self.garch_engine.prices) >= 2 else curr_price)
        rolling_vol = garch_res.sigma_next
        garch_forecast = garch_res.mu_next

        prev_open = self.recent_bars_deque[-2]["open"] if len(self.recent_bars_deque) >= 2 else bar.open
        prev_close = self.recent_bars_deque[-2]["close"] if len(self.recent_bars_deque) >= 2 else bar.close
        recent_h = [b["high"] for b in self.recent_bars_deque]
        recent_l = [b["low"] for b in self.recent_bars_deque]
        key_lvl = [macro_st.pdl, macro_st.pdh, round((macro_st.pdh + macro_st.pdl) / 2.0, 2)] if macro_st else []

        # --- STEP 3: Ensemble Brain Evaluation (SVM + XGBoost + ANN + Vision CNN + SMC + Chinmay Skills) ---
        ensemble_dec = self.ensemble_brain.evaluate(
            garch_signal=garch_res.signal,
            price_delta=price_delta,
            rolling_vol=rolling_vol,
            garch_forecast=garch_forecast,
            vpin=vpin_res.vpin,
            macro_state=macro_st,
            recent_bars=list(self.recent_bars_deque),
            smc_state=smc_st,
            open_p=bar.open,
            high_p=bar.high,
            low_p=bar.low,
            close_p=bar.close,
            prev_open=prev_open,
            prev_close=prev_close,
            recent_highs=recent_h,
            recent_lows=recent_l,
            key_levels=key_lvl,
            underlying=self.config.market.underlying,
            pattern_st=pattern_st
        )

        # Update Brain Training Dataset
        self.ensemble_brain.update_bar(
            current_price=curr_price,
            price_delta=price_delta,
            rolling_vol=rolling_vol,
            garch_forecast=garch_forecast,
            vpin=vpin_res.vpin,
            macro_state=macro_st
        )

        # Telemetry logging on every minute
        pos_str = f"OPEN [{pos.symbol} @ INR {pos.entry_price:.2f}]" if pos.side != PositionSide.FLAT else "FLAT"
        brain_status = 'VETO' if ensemble_dec.is_vetoed else ('APPROVED' if garch_res.signal != DirectionalSignal.HOLD else 'NEUTRAL')
        
        v_probs = ensemble_dec.vision_probs or {}
        p_c = v_probs.get("CALL", 0.0) * 100
        p_p = v_probs.get("PUT", 0.0) * 100
        p_h = v_probs.get("HOLD", 0.0) * 100
        vision_str = f"{ensemble_dec.vision_action} ({ensemble_dec.vision_confidence*100:.1f}% | C:{p_c:.0f}%, P:{p_p:.0f}%, H:{p_h:.0f}%)"

        playbook = self.humanoid.get_tactical_playbook(timestamp.to_pydatetime() if hasattr(timestamp, "to_pydatetime") else None)
        
        # AI Dynamic Strategy Mode Evaluation (Scalper vs Swing)
        chosen_mode, mode_reason = self.humanoid.evaluate_execution_mode(
            ensemble_dec=ensemble_dec,
            heavyweight_st=heavyweight_st,
            vpin_res=vpin_res,
            smc_st=smc_st,
            configured_mode=self.config.risk.execution_mode
        )

        thought = self.humanoid.synthesize_thought(
            curr_price=curr_price,
            vpin_res=vpin_res,
            garch_res=garch_res,
            ensemble_dec=ensemble_dec,
            macro_st=macro_st,
            smc_st=smc_st,
            global_st=global_st,
            news_st=news_st,
            heavyweight_st=heavyweight_st,
            pattern_st=pattern_st,
            vision_prediction=1 if "CALL" in (ensemble_dec.vision_action or "") else (-1 if "PUT" in (ensemble_dec.vision_action or "") else 0),
            vision_confidence=ensemble_dec.vision_confidence,
            position_side_str=pos_str,
            execution_mode_desc=f"{chosen_mode.value} MODE: {mode_reason}"
        )

        # Render 4-Pillars Cognitive HUD with Dynamic Mode Tag
        self.humanoid.render_humanoid_hud(
            timestamp=timestamp.to_pydatetime() if hasattr(timestamp, "to_pydatetime") else datetime.now(self.tz),
            curr_price=curr_price,
            vpin_res=vpin_res,
            garch_res=garch_res,
            ensemble_dec=ensemble_dec,
            pos_str=pos_str,
            day_pnl=self.risk_manager.daily_realized_pnl,
            vision_str=vision_str,
            global_st=global_st,
            news_st=news_st,
            heavyweight_st=heavyweight_st,
            smc_st=smc_st,
            journal=self.risk_manager.premarket_journal,
            execution_mode_tag=f"{chosen_mode.value} [{mode_reason}]"
        )
        self._record_bar_to_csv(bar, vpin_res, garch_res, ensemble_dec, macro_st)

        # Check Emergency Signal Reversal on Track 1 open position
        if pos.side != PositionSide.FLAT:
            raw_rev_signal = ensemble_dec.final_action if not ensemble_dec.is_vetoed else garch_res.signal
            rev_exit, rev_reason = self.risk_manager.check_exit_conditions(
                current_price=curr_price,
                option_premium=curr_opt_premium,
                current_time=timestamp.to_pydatetime() if hasattr(timestamp, "to_pydatetime") else None,
                reversal_signal=raw_rev_signal
            )
            if rev_exit:
                exit_exec_price = curr_opt_premium if (pos.is_option and curr_opt_premium) else curr_price
                self.execution.execute_exit(current_price=exit_exec_price, reason=rev_reason)

        # Rolling Volume tracking & Available Capital Query
        self.recent_volumes.append(bar.volume)
        mean_vol = np.mean(self.recent_volumes) if len(self.recent_volumes) > 0 else bar.volume
        vol_ratio = (bar.volume / mean_vol) if mean_vol > 0 else 1.0
        avail_capital = self.execution.get_available_capital()

        # --- MARKET HOURS TIME GATING (09:15 IST to 15:15 IST) & COOLDOWN GUARD ---
        bar_dt = timestamp.to_pydatetime() if hasattr(timestamp, "to_pydatetime") else datetime.now(self.tz)
        if bar_dt.tzinfo is None:
            bar_dt = self.tz.localize(bar_dt)
        m_open = bar_dt.replace(hour=self.config.market.market_open_hour, minute=self.config.market.market_open_minute, second=0, microsecond=0)
        m_cutoff = bar_dt.replace(hour=self.config.market.last_entry_hour, minute=self.config.market.last_entry_minute, second=0, microsecond=0)
        is_market_active = (m_open <= bar_dt <= m_cutoff)

        if self.pattern_watcher.is_in_cooldown(self.total_bars_processed):
            if is_market_active:
                logger.info(f"[COOLDOWN GUARD] Re-entry locked on {self.pattern_watcher.cooldown_symbol}. Allowing pullback to develop before new entry.")
            is_market_active = False

        if not is_market_active:
            if bar_dt < m_open:
                logger.debug(f"[PRE-MARKET] Current time {bar_dt.strftime('%H:%M:%S')} is before market open {m_open.strftime('%H:%M:%S')}. Entries paused.")
            elif bar_dt > m_cutoff:
                logger.debug(f"[POST-ENTRY CUTOFF] Current time {bar_dt.strftime('%H:%M:%S')} is past cutoff {m_cutoff.strftime('%H:%M:%S')}. Entries paused.")

        # --- STEP 4A: Order Execution on Track 1 (Standard Dual-Brain) ---
        is_option = (self.config.market.instrument_mode == InstrumentMode.OPTIONS_BUYING)
        target_symbol = None
        opt_premium = None

        if is_market_active and not ensemble_dec.is_vetoed and ensemble_dec.final_action in (DirectionalSignal.BUY, DirectionalSignal.SELL):
            action_str = "BUY" if ensemble_dec.final_action == DirectionalSignal.BUY else "SELL"
            
            # --- 5-MINUTE HTF ALIGNMENT CHECK ---
            htf_ok, htf_reason = self.mtf_analyst.validate_htf_entry_alignment(action_str, curr_price)
            if not htf_ok:
                logger.warning(f"[HTF VETO] [TRACK 1 DUAL] Entry blocked: {htf_reason}")
            else:
                if is_option:
                    target_symbol = self.data_feed.resolve_option_strike(action_str, curr_price)
                    opt_premium = self.data_feed.get_ltp(target_symbol, curr_price)
                    exec_price = opt_premium
                    logger.info(f"[TRACK 1 DUAL] Signal {ensemble_dec.final_action.value} -> Contract: {target_symbol} | Live Premium: INR {opt_premium:.2f}")
                else:
                    exec_price = curr_price

                t1_unanimous = (ensemble_dec.svm_prediction == ensemble_dec.xgb_prediction and ensemble_dec.xgb_prediction != 0)
                targets = self.risk_manager.compute_trade_targets(
                    signal=ensemble_dec.final_action,
                    current_price=curr_price,
                    garch_res=garch_res,
                    vpin_res=vpin_res,
                    is_option=is_option,
                    option_premium=opt_premium,
                    available_cash=avail_capital,
                    strategy_mode=chosen_mode,
                    candle_high=high_price,
                    candle_low=low_price
                )


                if targets:
                    if is_option and target_symbol and opt_premium:
                        opt_bars = self.opt_bars_tracker.setdefault(target_symbol, [])
                        opt_bars.append({"open": opt_premium, "high": opt_premium, "low": opt_premium, "close": opt_premium, "volume": 1000})
                        if len(opt_bars) > 30:
                            self.opt_bars_tracker[target_symbol] = opt_bars[-30:]
                        if len(opt_bars) >= 6:
                            opt_an = self.ensemble_brain.option_chart_skill.analyze_option_chart(
                                self.opt_bars_tracker[target_symbol],
                                symbol=target_symbol,
                                option_type="CE" if ("CALL" in target_symbol or "CE" in target_symbol) else "PE",
                                current_premium=opt_premium
                            )
                            if opt_an.pattern_detected == "M_BREAKDOWN":
                                logger.warning(f"[OPTION CHART VETO] [TRACK 1] {target_symbol} is in M-Pattern Breakdown! Entry vetoed.")
                                targets = None

                    if targets:
                        # Pre-Trade Sanity Verification (marian2js/trading-skills)
                        sanity_res = self.pre_trade_sanity_skill.evaluate(
                            intended_direction=action_str,
                            entry_price=exec_price,
                            stop_loss=targets.stop_loss,
                            take_profit=targets.take_profit,
                            current_spot=curr_price,
                            recent_bars=list(self.recent_bars_deque)
                        )
                        if not sanity_res.is_favorable:
                            logger.warning(f"[PRE-TRADE SANITY VETO] [TRACK 1] {sanity_res.reason}")
                            targets = None

                    if targets and is_option and (not target_symbol or "ATM_" in str(target_symbol)):
                        logger.warning(f"[ENTRY BLOCKED] Option symbol resolution failed: {target_symbol}. Skipping entry.")
                    elif targets:
                        self.risk_manager.current_position.entry_time = datetime.now(self.tz)
                        self.execution.execute_entry(
                            signal=ensemble_dec.final_action,
                            current_price=exec_price,
                            trade_target=targets,
                            vpin_res=vpin_res,
                            order_type=OrderType.MARKET,
                            target_symbol=target_symbol,
                            is_unanimous=t1_unanimous,
                            volume_ratio=vol_ratio
                        )
                elif is_market_active and target_symbol and opt_premium and not self.shadow_learner.is_tracking() and pos.side == PositionSide.FLAT:
                    # Daily trade limit or capital guard reached: Initiate Ghost/Shadow Virtual Trade
                    is_sensex = "SENSEX" in str(self.config.market.underlying).upper()
                    sh_sl_pts = 32.0 if is_sensex else 10.0
                    sh_tp_pts = 48.0 if is_sensex else 15.0
                    sh_sl = max(1.0, round(opt_premium - sh_sl_pts, 2))
                    sh_tp = round(opt_premium + sh_tp_pts, 2)
                    self.shadow_learner.start_shadow_trade(
                        signal=ensemble_dec.final_action,
                        symbol=target_symbol,
                        entry_price=opt_premium,
                        entry_spot=curr_price,
                        stop_loss=sh_sl,
                        take_profit=sh_tp,
                        current_time=timestamp.to_pydatetime() if hasattr(timestamp, "to_pydatetime") else datetime.now(self.tz),
                        is_option=is_option
                    )

        # --- STEP 4B: Order Execution on Track 2 (Tri-Brain ANN) if Enabled ---
        if self.compare_ann and self.tri_risk_manager and self.tri_execution:
            tri_pos = self.tri_risk_manager.current_position
            tri_opt_prem = None
            if tri_pos.side != PositionSide.FLAT and tri_pos.is_option and tri_pos.symbol:
                tri_opt_prem = self.data_feed.get_ltp(tri_pos.symbol, curr_price)

            # Track 2 Reversals & Exits
            if tri_pos.side != PositionSide.FLAT:
                tri_is_counter, tri_counter_msg = self.mtf_analyst.check_cumulative_counter_trend(
                    holding_call=(tri_pos.is_option and tri_pos.symbol and ("CALL" in tri_pos.symbol or "CE" in tri_pos.symbol)),
                    holding_put=(tri_pos.is_option and tri_pos.symbol and ("PUT" in tri_pos.symbol or "PE" in tri_pos.symbol)),
                    threshold_pts=20.0
                )
                tri_exit_needed, tri_exit_reason = self.tri_risk_manager.check_exit_conditions(
                    current_price=curr_price,
                    option_premium=tri_opt_prem,
                    current_time=timestamp.to_pydatetime() if hasattr(timestamp, "to_pydatetime") else None,
                    counter_trend_reason=tri_counter_msg if tri_is_counter else None
                )
                if tri_exit_needed:
                    tri_exit_p = tri_opt_prem if (tri_pos.is_option and tri_opt_prem) else curr_price
                    self.tri_execution.execute_exit(current_price=tri_exit_p, reason=f"[TRACK 2] {tri_exit_reason}")

            # Track 2 Entry
            if is_market_active and not ensemble_dec.tri_is_vetoed and ensemble_dec.tri_final_action in (DirectionalSignal.BUY, DirectionalSignal.SELL):
                tri_action_str = "BUY" if ensemble_dec.tri_final_action == DirectionalSignal.BUY else "SELL"
                
                # --- 5-MINUTE HTF ALIGNMENT CHECK ---
                tri_htf_ok, tri_htf_reason = self.mtf_analyst.validate_htf_entry_alignment(tri_action_str, curr_price)
                if not tri_htf_ok:
                    logger.warning(f"[HTF VETO] [TRACK 2 TRI-BRAIN] Entry blocked: {tri_htf_reason}")
                else:
                    if target_symbol is None:
                        target_symbol = self.data_feed.resolve_option_strike(tri_action_str, curr_price)
                    if opt_premium is None and is_option:
                        opt_premium = self.data_feed.get_ltp(target_symbol, curr_price)
                    tri_exec_p = opt_premium if is_option else curr_price

                    t2_unanimous = (ensemble_dec.svm_prediction == ensemble_dec.xgb_prediction == ensemble_dec.ann_prediction and ensemble_dec.ann_prediction != 0)
                    tri_targets = self.tri_risk_manager.compute_trade_targets(
                        signal=ensemble_dec.tri_final_action,
                        current_price=curr_price,
                        garch_res=garch_res,
                        vpin_res=vpin_res,
                        is_option=is_option,
                        option_premium=opt_premium,
                        available_cash=avail_capital,
                        strategy_mode=chosen_mode,
                        candle_high=high_price,
                        candle_low=low_price
                    )
                    if tri_targets and is_option and target_symbol and opt_premium:
                        if len(self.opt_bars_tracker.get(target_symbol, [])) >= 6:
                            opt_an = self.ensemble_brain.option_chart_skill.analyze_option_chart(
                                self.opt_bars_tracker[target_symbol],
                                symbol=target_symbol,
                                option_type="CE" if ("CALL" in target_symbol or "CE" in target_symbol) else "PE",
                                current_premium=opt_premium
                            )
                            if opt_an.pattern_detected == "M_BREAKDOWN":
                                logger.warning(f"[OPTION CHART VETO] [TRACK 2] {target_symbol} is in M-Pattern Breakdown! Entry vetoed.")
                                tri_targets = None

                    if tri_targets:
                        # Pre-Trade Sanity Verification (marian2js/trading-skills)
                        tri_sanity = self.pre_trade_sanity_skill.evaluate(
                            intended_direction=tri_action_str,
                            entry_price=tri_exec_p,
                            stop_loss=tri_targets.stop_loss,
                            take_profit=tri_targets.take_profit,
                            current_spot=curr_price,
                            recent_bars=list(self.recent_bars_deque)
                        )
                        if not tri_sanity.is_favorable:
                            logger.warning(f"[PRE-TRADE SANITY VETO] [TRACK 2] {tri_sanity.reason}")
                            tri_targets = None

                    if tri_targets and target_symbol and "ATM_" not in str(target_symbol):
                        self.tri_risk_manager.current_position.entry_time = datetime.now(self.tz)
                        self.tri_execution.execute_entry(
                            signal=ensemble_dec.tri_final_action,
                            current_price=tri_exec_p,
                            trade_target=tri_targets,
                            vpin_res=vpin_res,
                            order_type=OrderType.MARKET,
                            target_symbol=target_symbol,
                            is_unanimous=t2_unanimous,
                            volume_ratio=vol_ratio
                        )

    def print_comparison_report(self):
        """Prints post-market head-to-head performance comparison between Track 1 and Track 2"""
        logger.info("=" * 80)
        logger.info("       LIVE HEAD-TO-HEAD MODEL COMPARISON SUMMARY       ")
        logger.info("=" * 80)
        logger.info(f"  TRACK 1 (Dual-Brain SVM + XGB)  : {len(self.execution.order_book)} Orders | Net Realized PnL: {self.risk_manager.daily_realized_pnl:+.2f} INR")
        if self.tri_execution and self.tri_risk_manager:
            logger.info(f"  TRACK 2 (Tri-Brain ANN+XGB+SVM) : {len(self.tri_execution.order_book)} Orders | Net Realized PnL: {self.tri_risk_manager.daily_realized_pnl:+.2f} INR")
        
        # Shadow / Ghost Learner Insights
        sh_insights = self.shadow_learner.analyze_session_allocation()
        if sh_insights.get("phase_stats"):
            logger.info("-" * 80)
            logger.info("   VIRTUAL SHADOW LEARNER: SESSION TRADE DISTRIBUTION INSIGHTS   ")
            logger.info("-" * 80)
            for phase, st in sh_insights["phase_stats"].items():
                logger.info(f"  * {phase:<30}: {st['total_trades']} Trades | Win Rate: {st['win_rate']*100:.1f}% | Net: INR {st['net_pnl']:+,.2f}")
        logger.info("=" * 80)

        # Trigger Automated EOD Audit & Gemini AI Reflection Skill
        try:
            audit = self.daily_reflection_skill.run_end_of_day_audit()
            if audit:
                logger.info(f"[SELF-LEARNING] {audit.get('summary_takeaway', '')}")
                ai_ref = audit.get("ai_reflection", "")
                if ai_ref:
                    logger.info("-" * 80)
                    logger.info("          GEMINI AI AGENT POST-MARKET REFLECTION & CRITIQUE          ")
                    logger.info("-" * 80)
                    for line in ai_ref.splitlines():
                        logger.info(f"  {line}")
                    logger.info("-" * 80)

                # Autonomous Skill Generation / Enhancement
                chop_losses = audit.get("quant_metrics", {}).get("chop_losses", 0)
                if chop_losses >= 2:
                    logger.info("[SKILL EVOLUTION] Detected multiple chop losses today. Synthesizing Chop Guardian Skill...")
                    ok, msg = self.skill_generator.generate_and_deploy_skill(
                        skill_name="ChopFilterSkill",
                        description="Vetoes entries during low-momentum tight consolidation ranges",
                        detection_logic_rules="Reject entry when rolling 5-bar range is less than 15 pts and ADX is below 20",
                        registry=self.skill_registry
                    )
                    if ok:
                        logger.info(f"[SKILL EVOLUTION] {msg}")

        except Exception as e:
            logger.debug(f"[SELF-LEARNING] Notice: {e}")

    def run_shutdown_self_learning(self):
        """
        Triggered on graceful shutdown or user exit (Ctrl+C):
        Runs post-session evaluation, updates adaptive weights, evolves skills, and requests Gemini reflection.
        """
        if getattr(self, "_shutdown_completed", False):
            return
        self._shutdown_completed = True

        logger.info("\n" + "=" * 80)
        logger.info("   [SHUTDOWN] TRIGGERING AUTO SELF-LEARNING, SKILL EVOLUTION & GEMINI AI REFLECTION...   ")
        logger.info("=" * 80)
        self.print_comparison_report()

    def run_simulation(self, bars_df: pd.DataFrame):
        logger.info(f"Starting v2 simulation run for {self.config.market.underlying} across {len(bars_df)} bars...")
        self.warm_up(bars_df.head(150))

        for idx, row in bars_df.iloc[150:].iterrows():
            bar = BarOHLCV(
                timestamp=row.get("timestamp", pd.Timestamp.now(tz=self.tz)),
                open=float(row["open"]),
                high=float(row.get("high", row["close"])),
                low=float(row.get("low", row["close"])),
                close=float(row["close"]),
                volume=float(row["volume"]),
                symbol=self.config.market.symbol
            )
            self.process_incoming_bar(bar)

        self.print_comparison_report()

    async def run_live_session(self):
        """
        Executes the full Live Intraday Lifecycle with Real-Time Terminal HUD and Telegram Notifications.
        """
        self.warm_up()

        logger.info(f"[LIVE RUNNER] Starting real-time Dhan polling loop ({self.config.market.underlying})...")
        last_bar_minute = -1

        try:
            while True:
                now_ist = datetime.now(self.tz)

                # Check if Market Closed (Past 15:30 IST)
                if (now_ist.hour > self.config.market.market_close_hour) or \
                   (now_ist.hour == self.config.market.market_close_hour and now_ist.minute >= self.config.market.market_close_minute):
                    logger.info(f"Market closed for the day ({now_ist.strftime('%H:%M')} IST). Generating post-session report...")
                    self.print_comparison_report()
                    break

                # Batch poll live Spot Index price and active option positions in 1 atomic tick request
                symbols_to_poll = [self.config.market.symbol]
                pos = self.risk_manager.current_position
                if pos.side != PositionSide.FLAT and pos.symbol:
                    symbols_to_poll.append(pos.symbol)
                if self.compare_ann and self.tri_risk_manager and self.tri_risk_manager.current_position.side != PositionSide.FLAT:
                    tri_sym = self.tri_risk_manager.current_position.symbol
                    if tri_sym and tri_sym not in symbols_to_poll:
                        symbols_to_poll.append(tri_sym)
                if self.shadow_learner.is_tracking() and self.shadow_learner.active_shadow_position:
                    sh_sym = self.shadow_learner.active_shadow_position.symbol
                    if sh_sym and sh_sym not in symbols_to_poll:
                        symbols_to_poll.append(sh_sym)

                quotes = self.data_feed.get_multiple_ltp(symbols_to_poll)
                curr_spot = quotes.get(self.config.market.symbol, 0.0)
                if curr_spot <= 0:
                    curr_spot = self.data_feed.get_ltp(self.config.market.symbol)
                if curr_spot <= 0:
                    logger.warning(f"Failed to fetch live LTP for {self.config.market.symbol}. Retrying...")
                    await asyncio.sleep(3)
                    continue

                # --- FAST GUARDIAN: Check active Track 1 position ---
                if pos.side != PositionSide.FLAT and pos.symbol:
                    try:
                        curr_opt_ltp = quotes.get(pos.symbol, 0.0)
                        if curr_opt_ltp <= 0:
                            curr_opt_ltp = self.data_feed.get_ltp(pos.symbol, curr_spot)
                        exit_needed, exit_reason = self.risk_manager.check_exit_conditions(
                            current_price=curr_spot,
                            option_premium=curr_opt_ltp,
                            current_time=now_ist
                        )
                        if exit_needed:
                            logger.info(f"[FAST GUARDIAN EXIT v2] [TRACK 1] Instant trigger: {exit_reason} @ LTP INR {curr_opt_ltp:.2f}")
                            self.execution.execute_exit(current_price=curr_opt_ltp, reason=exit_reason)
                    except Exception as e:
                        logger.debug(f"Fast guardian check Track 1: {e}")

                # --- FAST GUARDIAN: Check active Track 2 position if enabled ---
                if self.compare_ann and self.tri_risk_manager and self.tri_execution:
                    tri_pos = self.tri_risk_manager.current_position
                    if tri_pos.side != PositionSide.FLAT and tri_pos.symbol:
                        try:
                            tri_opt_ltp = quotes.get(tri_pos.symbol, 0.0)
                            if tri_opt_ltp <= 0:
                                tri_opt_ltp = self.data_feed.get_ltp(tri_pos.symbol, curr_spot)
                            tri_exit_needed, tri_exit_reason = self.tri_risk_manager.check_exit_conditions(
                                current_price=curr_spot,
                                option_premium=tri_opt_ltp,
                                current_time=now_ist
                            )
                            if tri_exit_needed:
                                logger.info(f"[FAST GUARDIAN EXIT v2] [TRACK 2] Instant trigger: {tri_exit_reason} @ LTP INR {tri_opt_ltp:.2f}")
                                self.tri_execution.execute_exit(current_price=tri_opt_ltp, reason=tri_exit_reason)
                        except Exception as e:
                            logger.debug(f"Fast guardian check Track 2: {e}")

                # --- FAST GUARDIAN: Check active Virtual Shadow Position (Ghost Learning) ---
                if self.shadow_learner.is_tracking() and self.shadow_learner.active_shadow_position:
                    try:
                        sh_sym = self.shadow_learner.active_shadow_position.symbol
                        sh_opt_ltp = quotes.get(sh_sym, 0.0)
                        if sh_opt_ltp <= 0:
                            sh_opt_ltp = self.data_feed.get_ltp(sh_sym, curr_spot)
                        self.shadow_learner.update_shadow_position(
                            current_spot=curr_spot,
                            option_ltp=sh_opt_ltp,
                            current_time=now_ist
                        )
                    except Exception as e:
                        logger.debug(f"Fast guardian check Shadow Position: {e}")

                # --- 1-MINUTE ENGINE: Process Bar when minute rolls over ---
                current_minute = now_ist.minute
                if current_minute != last_bar_minute:
                    last_bar_minute = current_minute
                    try:
                        bar = BarOHLCV(
                            timestamp=pd.Timestamp(now_ist),
                            open=curr_spot,
                            high=curr_spot,
                            low=curr_spot,
                            close=curr_spot,
                            volume=25000.0,
                            symbol=self.config.market.symbol
                        )
                        self.process_incoming_bar(bar)
                    except Exception as e:
                        logger.error(f"Error in 1-minute polling cycle: {e}")

                # Sleep interval (8s if in trade, 10s if flat) to strictly respect Dhan API limits
                in_any_trade = (pos.side != PositionSide.FLAT) or (self.compare_ann and self.tri_risk_manager and self.tri_risk_manager.current_position.side != PositionSide.FLAT)
                sleep_duration = 8 if in_any_trade else 10
                await asyncio.sleep(sleep_duration)
        except (KeyboardInterrupt, asyncio.CancelledError):
            self.run_shutdown_self_learning()


# Alias for backward compatibility
QuantitativeTradingEngineV2 = AlgoVPINRunner


def main():
    parser = argparse.ArgumentParser(description="Algo VPIN v2.0 Multi-Timeframe Quantitative Options Engine")
    parser.add_argument("--mode", choices=["live", "simulation"], default="live", help="Run mode: live or simulation (default: live)")
    parser.add_argument("--symbol", choices=["NIFTY", "SENSEX"], default="NIFTY", help="Underlying Index (NIFTY or SENSEX)")
    parser.add_argument("--strategy-mode", choices=["auto", "scalper", "swing"], default="auto", help="Strategy execution mode: auto (Brain selects), scalper (1:1.5 RR + 0.5R Trail), or swing (Multi-target runner)")
    parser.add_argument("--bars", type=int, default=500, help="Number of bars for simulation")
    parser.add_argument("--no-compare", action="store_true", help="Disable Track 2 comparison (comparison is ON by default)")
    args = parser.parse_args()

    engine = AlgoVPINRunner()
    if args.symbol:
        engine.config.market.set_symbol(args.symbol)

    if args.strategy_mode == "scalper":
        engine.config.risk.execution_mode = ExecutionMode.SCALPER_ONLY
        logger.info("[MODE CONFIG] Enforced Pure SCALPER Mode (1:1.5 RR + 0.5R Step Trailing)")
    elif args.strategy_mode == "swing":
        engine.config.risk.execution_mode = ExecutionMode.SWING_ONLY
        logger.info("[MODE CONFIG] Enforced Pure INSTITUTIONAL SWING Runner Mode")
    else:
        engine.config.risk.execution_mode = ExecutionMode.AUTO_BRAIN_SELECT
        logger.info("[MODE CONFIG] Enabled AI Dynamic Mode (Brain dynamically selects Scalp vs Swing)")

    if args.no_compare:
        engine.compare_ann = False
        engine.tri_risk_manager = None
        engine.tri_execution = None
        logger.info("[MODE CONFIG] Track 2 comparison disabled.")
    else:
        logger.info("[MODE CONFIG] Track 1 & Track 2 Parallel Comparison ACTIVE.")

    try:
        if args.mode == "simulation":
            engine.config.telegram.enabled = False
            CONFIG.telegram.enabled = False
            start_price = 24000.0 if engine.config.market.underlying == "NIFTY" else 79000.0
            bars_df = engine.data_feed.generate_synthetic_bars(n_bars=args.bars, start_price=start_price)
            engine.run_simulation(bars_df)
        else:
            try:
                asyncio.run(engine.run_live_session())
            except (KeyboardInterrupt, SystemExit, asyncio.CancelledError):
                engine.run_shutdown_self_learning()
    except (KeyboardInterrupt, SystemExit):
        engine.run_shutdown_self_learning()


if __name__ == "__main__":
    main()
