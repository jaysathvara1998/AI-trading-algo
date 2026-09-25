"""
Layer 3: XGBoost Machine Learning Brain for Algo VPIN v2.0
Fits Gradient Boosted Decision Trees on multi-timeframe feature vectors,
evaluates non-linear interaction rules, and predicts directional breakout probabilities.
"""

from pathlib import Path
from typing import List, Optional, Tuple, Dict
import logging
import joblib
import numpy as np
import xgboost as xgb

from .config import EnsembleConfig
from .garch_engine import DirectionalSignal

logger = logging.getLogger("algo_vpin_v2.xgboost_brain")


class XGBoostBrain:
    """
    Gradient Boosted Decision Tree classifier for directional trade validation.
    """
    def __init__(self, config: Optional[EnsembleConfig] = None):
        self.config = config or EnsembleConfig()
        self.model: Optional[xgb.XGBClassifier] = None
        self.is_trained: bool = False
        
        self.feature_history: List[List[float]] = []
        self.target_history: List[int] = []
        self.feature_names = [
            "bar_delta", "rolling_vol", "garch_forecast", "vpin",
            "dist_pdh", "dist_pdl", "week_pos", "trend_15m",
            "smc_trend", "smc_bos", "smc_choch", "smc_range_pos",
            "smc_fvg_bias", "smc_sweep_bias"
        ]
        
        self._pending_feature: Optional[List[float]] = None
        self._pending_price: Optional[float] = None
        self.bars_since_retrain: int = 0

    def build_feature_vector(
        self,
        price_delta: float,
        rolling_vol: float,
        garch_forecast: float,
        vpin: float,
        dist_pdh: float,
        dist_pdl: float,
        week_pos: float,
        trend_15m: int,
        smc_trend: int = 0,
        smc_bos: int = 0,
        smc_choch: int = 0,
        smc_range_pos: float = 0.5,
        smc_fvg_bias: int = 0,
        smc_sweep_bias: int = 0
    ) -> List[float]:
        """Constructs 14-dimensional multi-timeframe & Smart Money Concepts feature vector"""
        return [
            float(price_delta),
            float(rolling_vol),
            float(garch_forecast),
            float(vpin),
            float(dist_pdh),
            float(dist_pdl),
            float(week_pos),
            float(trend_15m),
            float(smc_trend),
            float(smc_bos),
            float(smc_choch),
            float(smc_range_pos),
            float(smc_fvg_bias),
            float(smc_sweep_bias)
        ]

    def update_bar(
        self,
        current_price: float,
        feature_vector: List[float]
    ):
        """
        Updates XGBoost training set when a new 1-min bar arrives.
        """
        if self._pending_feature is not None and self._pending_price is not None:
            forward_ret = current_price - self._pending_price
            # Label: 1 for Bullish, 0 for Bearish (XGBoost binary format)
            target = 1 if forward_ret >= 0 else 0
            self.feature_history.append(self._pending_feature)
            self.target_history.append(target)

            if len(self.feature_history) > self.config.max_history_samples:
                self.feature_history = self.feature_history[-self.config.max_history_samples:]
                self.target_history = self.target_history[-self.config.max_history_samples:]

        self._pending_feature = feature_vector
        self._pending_price = current_price

        self.bars_since_retrain += 1
        if self.bars_since_retrain >= self.config.retrain_interval:
            self.train_model()

    def train_model(self) -> bool:
        """Trains the XGBoost Classifier on historical feature vectors"""
        if len(self.feature_history) < 30:
            return False

        unique_classes = set(self.target_history)
        if len(unique_classes) < 2:
            return False

        X = np.array(self.feature_history, dtype=np.float32)
        y = np.array(self.target_history, dtype=np.int32)

        # Symmetric Class Balancing: prevents upward/downward bias from historical drift
        n_pos = int(np.sum(y == 1))
        n_neg = int(np.sum(y == 0))
        scale_weight = float(n_neg) / float(max(1, n_pos)) if n_pos > 0 else 1.0

        try:
            model = xgb.XGBClassifier(
                n_estimators=self.config.xgb_n_estimators,
                max_depth=self.config.xgb_max_depth,
                learning_rate=self.config.xgb_learning_rate,
                scale_pos_weight=scale_weight,
                eval_metric="logloss",
                tree_method="hist",
                random_state=42
            )
            model.fit(X, y)
            self.model = model
            self.is_trained = True
            self.bars_since_retrain = 0
            self.save_model()
            return True
        except Exception as e:
            logger.warning(f"XGBoost training failed: {e}")
            return False

    def predict(self, feature_vector: List[float]) -> Tuple[int, float]:
        """
        Generates directional prediction (+1 / -1) and confidence probability (0.0 to 1.0).
        """
        if not self.is_trained or self.model is None:
            return 1, 0.50

        x_arr = np.array([feature_vector], dtype=np.float32)
        try:
            proba = self.model.predict_proba(x_arr)[0]
            prob_up = float(proba[1])
            pred = 1 if prob_up >= 0.50 else -1
            confidence = prob_up if pred == 1 else (1.0 - prob_up)
            return pred, confidence
        except Exception:
            return 1, 0.50

    def get_feature_importances(self) -> Dict[str, float]:
        """Returns feature importance distribution from trained decision trees"""
        if not self.is_trained or self.model is None:
            return {}
        try:
            importances = self.model.feature_importances_
            return {name: float(round(imp, 4)) for name, imp in zip(self.feature_names, importances)}
        except Exception:
            return {}

    def save_model(self, file_path: Optional[Path] = None):
        """Saves trained XGBoost brain to disk"""
        if not self.is_trained or self.model is None:
            return
        try:
            target_path = file_path or (Path(__file__).parent / "models" / "xgb_model.joblib")
            target_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump({
                "model": self.model,
                "feature_history": self.feature_history,
                "target_history": self.target_history,
                "is_trained": self.is_trained
            }, target_path)
            logger.info(f"[XGBoost Brain] Saved model to {target_path.name}")
        except Exception as e:
            logger.warning(f"Could not save XGBoost model: {e}")

    def load_model(self, file_path: Optional[Path] = None) -> bool:
        """Loads pre-trained XGBoost brain from disk"""
        target_path = file_path or (Path(__file__).parent / "models" / "xgb_model.joblib")
        if not target_path.exists():
            return False
        try:
            payload = joblib.load(target_path)
            self.model = payload["model"]
            raw_feats = payload.get("feature_history", [])
            # Only keep history with matching dimensions
            self.feature_history = [f for f in raw_feats if len(f) == len(self.feature_names)]
            self.target_history = payload.get("target_history", [])[:len(self.feature_history)]
            self.is_trained = payload.get("is_trained", True)
            logger.info(f"[XGBoost Brain] Loaded pre-trained model ({len(self.feature_history)} samples)")
            return True
        except Exception as e:
            logger.warning(f"Could not load XGBoost model: {e}")
            return False
