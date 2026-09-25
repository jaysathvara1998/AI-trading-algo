"""
Layer 4: Artificial Neural Network (ANN) Deep Brain for Algo VPIN v2.0
Implements a Multi-Layer Perceptron (MLP) Deep Neural Network:
- Input Layer: 8-Dimensional Microstructure + Volatility + Macro Features
- Hidden Layers: Layer 1 (64 Neurons + ReLU), Layer 2 (32 Neurons + ReLU)
- Output Layer: Softmax Probability Distribution [P(PUT), P(HOLD), P(CALL)]
"""

from pathlib import Path
from typing import List, Optional, Tuple, Dict
import logging
import joblib
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from .config import EnsembleConfig
from .garch_engine import DirectionalSignal

logger = logging.getLogger("algo_vpin_v2.ann_brain")


class ANNBrain:
    """
    Artificial Neural Network (ANN / MLP) Classifier for deep non-linear pattern recognition.
    """
    def __init__(self, config: Optional[EnsembleConfig] = None, hidden_layer_sizes: Tuple[int, ...] = (64, 32)):
        self.config = config or EnsembleConfig()
        self.hidden_layer_sizes = hidden_layer_sizes
        
        self.scaler = StandardScaler()
        self.model: Optional[MLPClassifier] = None
        self.is_trained: bool = False
        
        self.feature_history: List[List[float]] = []
        self.target_history: List[int] = []  # 0: PUT, 1: HOLD, 2: CALL
        self.feature_names = [
            "bar_delta", "rolling_vol", "garch_forecast", "vpin",
            "dist_pdh", "dist_pdl", "week_pos", "trend_15m",
            "smc_trend", "smc_bos", "smc_choch", "smc_range_pos",
            "smc_fvg_bias", "smc_sweep_bias"
        ]
        
        self._pending_feature: Optional[List[float]] = None
        self._pending_price: Optional[float] = None
        self.bars_since_retrain: int = 0
        self.underlying: str = "NIFTY"
        self.model_path = Path("ann_model.joblib")

    def set_underlying(self, underlying: str):
        """Configures asset-specific thresholds for SENSEX / NIFTY"""
        self.underlying = (underlying or "NIFTY").upper()

    def get_move_threshold(self) -> float:
        """Dynamic threshold to ignore 1-second bid-ask noise"""
        if "SENSEX" in self.underlying:
            return 20.0
        elif "BANKNIFTY" in self.underlying:
            return 14.0
        return 6.0

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
    ):
        """
        Labels past bar t-1 based on asset-scaled price movement to t.
        0: PUT / Drop (< -threshold pts)
        1: HOLD (Noise / Flat)
        2: CALL / Rally (> +threshold pts)
        """
        threshold = self.get_move_threshold()

        if self._pending_feature is not None and self._pending_price is not None:
            future_return = current_price - self._pending_price
            
            if future_return > threshold:
                label = 2  # CALL
            elif future_return < -threshold:
                label = 0  # PUT
            else:
                label = 1  # HOLD (Filtered out noise)

            self.feature_history.append(self._pending_feature)
            self.target_history.append(label)

            # Cap sliding memory
            max_samples = 4000
            if len(self.feature_history) > max_samples:
                self.feature_history = self.feature_history[-max_samples:]
                self.target_history = self.target_history[-max_samples:]

        self._pending_feature = self.build_feature_vector(
            price_delta, rolling_vol, garch_forecast, vpin, dist_pdh, dist_pdl, week_pos, trend_15m,
            smc_trend, smc_bos, smc_choch, smc_range_pos, smc_fvg_bias, smc_sweep_bias
        )
        self._pending_price = current_price
        self.bars_since_retrain += 1

        if len(self.feature_history) >= 100 and (not self.is_trained or self.bars_since_retrain >= 20):
            self.train()

    def train(self):
        """Trains the Deep Artificial Neural Network with Adam optimizer, strong L2 penalty, and early stopping."""
        if len(self.feature_history) < 60:
            return

        X = np.array(self.feature_history, dtype=np.float32)
        y = np.array(self.target_history, dtype=np.int64)

        # Ensure all 3 classes exist in sample batch
        unique_classes = np.unique(y)
        if len(unique_classes) < 2:
            return

        try:
            X_scaled = self.scaler.fit_transform(X)
            
            self.model = MLPClassifier(
                hidden_layer_sizes=self.hidden_layer_sizes,
                activation='relu',
                solver='adam',
                alpha=0.02,  # Strong L2 Ridge Regularization to prevent noise overfitting
                batch_size=min(64, len(X)),
                learning_rate='adaptive',
                learning_rate_init=0.003,
                max_iter=300,
                early_stopping=True,
                n_iter_no_change=20,
                random_state=42
            )
            self.model.fit(X_scaled, y)
            self.is_trained = True
            self.bars_since_retrain = 0
            self.save_model()
            logger.info(f"[ANN Neural Brain] Successfully trained deep network on {len(X)} samples. Iterations: {self.model.n_iter_}")
        except Exception as e:
            logger.error(f"[ANN Neural Brain] Training error: {e}")

    def predict(self, feature_vector: List[float]) -> Tuple[int, float, Dict[str, float]]:
        """
        Forward propagation through the Artificial Neural Network.
        Returns:
            - signal: -1 (PUT), 0 (HOLD), +1 (CALL)
            - confidence: probability score (0.0 to 1.0)
            - probs_dict: {'PUT': p0, 'HOLD': p1, 'CALL': p2}
        """
        if not self.is_trained or self.model is None:
            return 0, 0.50, {'PUT': 0.33, 'HOLD': 0.34, 'CALL': 0.33}

        try:
            X = np.array([feature_vector], dtype=np.float32)
            X_scaled = self.scaler.transform(X)
            probs = self.model.predict_proba(X_scaled)[0]
            classes = self.model.classes_

            prob_map = {0: 0.0, 1: 0.0, 2: 0.0}
            for cls_idx, cls_label in enumerate(classes):
                prob_map[int(cls_label)] = float(probs[cls_idx])

            put_p = prob_map[0]
            hold_p = prob_map[1]
            call_p = prob_map[2]

            probs_dict = {'PUT': round(put_p, 4), 'HOLD': round(hold_p, 4), 'CALL': round(call_p, 4)}

            if call_p > put_p and call_p > hold_p and call_p >= 0.50:
                return +1, call_p, probs_dict
            elif put_p > call_p and put_p > hold_p and put_p >= 0.50:
                return -1, put_p, probs_dict
            else:
                return 0, max(hold_p, 0.50), probs_dict
        except Exception as e:
            logger.debug(f"[ANN Neural Brain] Forward pass exception: {e}")
            return 0, 0.50, {'PUT': 0.33, 'HOLD': 0.34, 'CALL': 0.33}

    def save_model(self, path: Optional[Path] = None):
        target = path or self.model_path
        if self.is_trained and self.model is not None:
            try:
                joblib.dump({"model": self.model, "scaler": self.scaler}, target)
            except Exception as e:
                logger.debug(f"[ANN Neural Brain] Save model error: {e}")

    def load_model(self, path: Optional[Path] = None) -> bool:
        target = path or self.model_path
        if target.exists():
            try:
                data = joblib.load(target)
                self.model = data["model"]
                self.scaler = data["scaler"]
                self.is_trained = True
                logger.info("[ANN Neural Brain] Loaded pre-trained Deep Neural Network from disk.")
                return True
            except Exception as e:
                logger.debug(f"[ANN Neural Brain] Could not load model: {e}")
        return False
