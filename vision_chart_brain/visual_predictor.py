"""
Real-Time Visual Predictor Bridge for Algo VPIN
Ingests real-time OHLCV bars, renders an in-memory 128x128 candlestick snapshot,
and executes 2D CNN inference to output visual pattern confirmation:
  - BUY CALL: Bullish candlestick structure & breakout momentum
  - BUY PUT: Bearish breakdown & rejection wick structure
  - HOLD: Consolidation trap, indecision, or adverse wick
"""

import os
from pathlib import Path
from typing import Dict, Tuple, Optional, Union, List
import numpy as np
import pandas as pd
import logging

from .chart_renderer import FastChartRenderer

logger = logging.getLogger("vision_chart_brain.predictor")


class VisualPredictor:
    def __init__(self, model_path: Optional[str] = None, window_bars: int = 20):
        self.window_bars = window_bars
        self.renderer = FastChartRenderer(image_size=128, window_bars=window_bars)
        self.model = None
        self.device = "cpu"

        default_model = Path(__file__).resolve().parent / "vision_cnn_model.pt"
        target_path = model_path or str(default_model)
        
        self._load_model(target_path)

    def _load_model(self, model_path: str):
        try:
            import torch
            from .cnn_vision_model import CandlestickCNN
            
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            if os.path.exists(model_path):
                self.model = CandlestickCNN(num_classes=3)
                state_dict = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                self.model.to(self.device)
                self.model.eval()
                logger.info(f"[Visual Predictor] Loaded trained 2D CNN model from {model_path} on {self.device}")
            else:
                logger.warning(f"[Visual Predictor] Model file {model_path} not found. Please train model first.")
        except Exception as e:
            logger.warning(f"[Visual Predictor] Error loading PyTorch CNN model: {e}")

    def predict_chart(self, ohlcv_data: Union[pd.DataFrame, List[dict]]) -> Dict:
        """
        Executes fast visual inference on recent candlestick chart history.
        
        Returns:
            dict containing:
              - 'action': 'BUY_CALL' (+1), 'BUY_PUT' (-1), or 'HOLD' (0)
              - 'confidence': float in [0.0, 1.0]
              - 'probabilities': {'HOLD': p0, 'CALL': p1, 'PUT': p2}
              - 'numeric_signal': +1, -1, or 0
        """
        # Render in-memory 128x128 image
        img_arr = self.renderer.render_ohlc(ohlcv_data)
        
        if self.model is None:
            # Fallback heuristic if model not trained yet
            return {
                "action": "HOLD",
                "confidence": 0.50,
                "probabilities": {"HOLD": 0.50, "CALL": 0.25, "PUT": 0.25},
                "numeric_signal": 0
            }

        import torch
        import torch.nn.functional as F

        # Shape [1, 3, 128, 128]
        tensor = torch.from_numpy(img_arr.transpose(2, 0, 1).astype(np.float32) / 255.0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=-1).cpu().numpy()[0]

        p_hold = float(probs[0])
        p_call = float(probs[1])
        p_put = float(probs[2])

        top_class = int(np.argmax(probs))
        confidence = float(probs[top_class])

        if top_class == 1:
            action = "BUY_CALL"
            num_sig = 1
        elif top_class == 2:
            action = "BUY_PUT"
            num_sig = -1
        else:
            action = "HOLD"
            num_sig = 0

        return {
            "action": action,
            "confidence": round(confidence, 4),
            "probabilities": {
                "HOLD": round(p_hold, 4),
                "CALL": round(p_call, 4),
                "PUT": round(p_put, 4)
            },
            "numeric_signal": num_sig
        }
