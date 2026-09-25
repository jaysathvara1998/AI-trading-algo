"""
Autonomous Self-Tuning Engine for Algo VPIN v2.0
Dynamically auto-adjusts risk rules, hyperparameter cutoffs, and chop-zone filters
without requiring manual config edits.
Persists dynamic settings to `data/adaptive_config.json`.
"""

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger("algo_vpin_v2.auto_tuner")


@dataclass
class AdaptiveConfigState:
    max_daily_trades: int = 12
    xgb_min_prob_threshold: float = 0.75
    post_trade_cooldown_bars: int = 3
    avoid_equilibrium_chop: bool = True
    equilibrium_buffer_pts: float = 12.0
    instant_sl_cut: bool = True
    last_updated: str = ""
    learning_notes: str = ""


class AutoTuningEngine:
    """
    Autonomous Self-Calibrator:
    1. Adjusts thresholds in real-time based on closed trade feedback
    2. Syncs with DailyReflectionSkill at market close
    3. Persists state to `data/adaptive_config.json`
    """
    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or (Path(__file__).resolve().parent / "data")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.config_path = self.data_dir / "adaptive_config.json"
        self.state = self.load_adaptive_config()

    def load_adaptive_config(self) -> AdaptiveConfigState:
        """Loads adaptive parameters from JSON on disk"""
        if self.config_path.exists():
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return AdaptiveConfigState(**data)
            except Exception as e:
                logger.debug(f"[AutoTuner] Could not load adaptive config: {e}")
        
        default_state = AdaptiveConfigState()
        self.save_adaptive_config(default_state)
        return default_state

    def save_adaptive_config(self, state: Optional[AdaptiveConfigState] = None):
        """Saves current adaptive parameters to disk"""
        if state is not None:
            self.state = state
        try:
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(asdict(self.state), f, indent=2)
            logger.info(f"[AutoTuner] Persisted adaptive settings to {self.config_path}")
        except Exception as e:
            logger.error(f"[AutoTuner] Error saving adaptive config: {e}")

    def on_trade_completed(self, pnl_inr: float, pnl_pct: float, exit_reason: str):
        """
        Real-time feedback loop: Triggered after every single closed position.
        """
        if pnl_inr > 0:
            logger.info(f"[AutoTuner] Winning trade (+₹{pnl_inr:.2f}). Activating 3-bar post-trade cooldown buffer.")
            self.state.post_trade_cooldown_bars = 3
        else:
            logger.warning(f"[AutoTuner] Loss trade (-₹{abs(pnl_inr):.2f} | {exit_reason[:30]}). Tightening filter.")
            if "STOP_LOSS" in exit_reason.upper():
                # Raise threshold slightly to filter noisy chop
                self.state.xgb_min_prob_threshold = min(0.80, self.state.xgb_min_prob_threshold + 0.02)
                self.state.avoid_equilibrium_chop = True
        
        self.save_adaptive_config()

    def sync_with_eod_audit(self, audit_data: Dict[str, Any]):
        """
        End-of-Day synchronization with DailyReflectionSkill audit results.
        """
        if not audit_data:
            return

        adapted = audit_data.get("adapted_rules", {})
        win_rate = audit_data.get("win_rate", 0.5)

        self.state.max_daily_trades = adapted.get("max_daily_trades", 5)
        self.state.xgb_min_prob_threshold = adapted.get("xgb_min_prob_threshold", 0.75)
        self.state.avoid_equilibrium_chop = adapted.get("avoid_chop_zone", True)
        self.state.instant_sl_cut = adapted.get("instant_sl_cut", True)
        self.state.last_updated = audit_data.get("last_audit_date", "")
        self.state.learning_notes = audit_data.get("summary_takeaway", "")

        self.save_adaptive_config()
        logger.info(f"[AutoTuner] Synchronized EOD learnings: WinRate={win_rate*100:.0f}%, MaxTrades={self.state.max_daily_trades}, MinConf={self.state.xgb_min_prob_threshold*100:.0f}%")
