"""
Layer 3: SVM Trade-Veto Filter (RBF Kernel) for Algo VPIN v2.0
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
import logging
import joblib
import numpy as np
import pandas as pd
from sklearn.svm import SVC

from .config import EnsembleConfig
from .garch_engine import DirectionalSignal

logger = logging.getLogger("algo_vpin_v2.svm_filter")


@dataclass
class SVMVetoDecision:
    is_vetoed: bool
    svm_prediction: int
    garch_signal: DirectionalSignal
    final_action: DirectionalSignal
    reason: str
    decision_margin: float
    model_trained: bool


class SVMTradeFilter:
    def __init__(self, config: Optional[EnsembleConfig] = None):
        cfg = config or EnsembleConfig()
        self.kernel = cfg.svm_kernel
        self.c_param = cfg.svm_c
        self.gamma = cfg.svm_gamma
        self.retrain_interval = cfg.retrain_interval
        self.max_samples = cfg.max_history_samples

        self.model: Optional[SVC] = None
        self.is_trained: bool = False
        self.feature_history: List[List[float]] = []
        self.target_history: List[int] = []
        self.price_history: List[float] = []

        self._pending_feature: Optional[List[float]] = None
        self._pending_price: Optional[float] = None
        self.bars_since_retrain: int = 0

    def build_feature_vector(self, price_delta: float, rolling_volatility: float, garch_forecast: float) -> List[float]:
        return [float(price_delta), float(rolling_volatility), float(garch_forecast)]

    def update_bar(self, current_price: float, price_delta: float, rolling_vol: float, garch_forecast: float):
        if self._pending_feature is not None and self._pending_price is not None:
            forward_ret = current_price - self._pending_price
            target = 1 if forward_ret >= 0 else -1
            self.feature_history.append(self._pending_feature)
            self.target_history.append(target)

            if len(self.feature_history) > self.max_samples:
                self.feature_history = self.feature_history[-self.max_samples:]
                self.target_history = self.target_history[-self.max_samples:]

        self._pending_feature = self.build_feature_vector(price_delta, rolling_vol, garch_forecast)
        self._pending_price = current_price
        self.price_history.append(current_price)

        self.bars_since_retrain += 1
        if self.bars_since_retrain >= self.retrain_interval:
            self.train_model()

    def train_model(self) -> bool:
        if len(self.feature_history) < 30:
            return False
        unique_classes = set(self.target_history)
        if len(unique_classes) < 2:
            return False

        X = np.array(self.feature_history, dtype=np.float64)
        y = np.array(self.target_history, dtype=np.int32)

        try:
            model = SVC(kernel=self.kernel, C=self.c_param, gamma=self.gamma, decision_function_shape="ovr")
            model.fit(X, y)
            self.model = model
            self.is_trained = True
            self.bars_since_retrain = 0
            self.save_model()
            return True
        except Exception:
            return False

    def save_model(self, file_path: Optional[Path] = None):
        if not self.is_trained or self.model is None:
            return
        try:
            target_path = file_path or (Path(__file__).parent / "models" / "svm_model.joblib")
            target_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump({
                "model": self.model,
                "feature_history": self.feature_history,
                "target_history": self.target_history,
                "is_trained": self.is_trained
            }, target_path)
            logger.info(f"[SVM Engine] Saved trained model to {target_path.name}")
        except Exception as e:
            logger.warning(f"Could not save SVM model: {e}")

    def load_model(self, file_path: Optional[Path] = None) -> bool:
        target_path = file_path or (Path(__file__).parent / "models" / "svm_model.joblib")
        if not target_path.exists():
            return False
        try:
            payload = joblib.load(target_path)
            self.model = payload["model"]
            self.feature_history = payload.get("feature_history", [])
            self.target_history = payload.get("target_history", [])
            self.is_trained = payload.get("is_trained", True)
            logger.info(f"[SVM Engine] Loaded pre-trained SVM model ({len(self.feature_history)} samples)")
            return True
        except Exception as e:
            logger.warning(f"Could not load SVM model: {e}")
            return False

    def predict(self, feature_vector: List[float]) -> Tuple[int, float]:
        if not self.is_trained or self.model is None:
            return 1, 0.0
        x_arr = np.array([feature_vector], dtype=np.float64)
        try:
            pred = int(self.model.predict(x_arr)[0])
            distance = float(self.model.decision_function(x_arr)[0])
            return pred, distance
        except Exception:
            return 1, 0.0
