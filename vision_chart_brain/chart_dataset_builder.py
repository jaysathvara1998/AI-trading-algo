"""
Dataset Builder for 2D Candlestick Chart Computer Vision AI
Extracts historical bars, generates rolling candlestick snapshots, and auto-labels them:
  - Class 0: HOLD / CHOP / NOISE (Consolidation traps, tight wicks)
  - Class 1: BUY CALL (Strong forward upward impulse > +12 pts within 5 bars, R:R >= 2.0)
  - Class 2: BUY PUT (Strong forward downward impulse < -12 pts within 5 bars, R:R >= 2.0)
"""

import os
from pathlib import Path
from typing import Tuple, List, Optional
import numpy as np
import pandas as pd
import logging

from .chart_renderer import FastChartRenderer

logger = logging.getLogger("vision_chart_brain.dataset")


class ChartDatasetBuilder:
    def __init__(self, window_bars: int = 20, forward_horizon: int = 5, impulse_threshold_pts: float = 12.0):
        self.window_bars = window_bars
        self.forward_horizon = forward_horizon
        self.impulse_threshold_pts = impulse_threshold_pts
        self.renderer = FastChartRenderer(image_size=128, window_bars=window_bars)

    def label_window(self, df: pd.DataFrame, current_idx: int) -> int:
        """
        Labels the rolling window based on forward 5-bar price action:
          0 = CHOP / HOLD
          1 = BUY CALL (Bullish breakout)
          2 = BUY PUT (Bearish breakdown)
        """
        curr_close = df.iloc[current_idx]["close"]
        forward_slice = df.iloc[current_idx + 1 : current_idx + 1 + self.forward_horizon]
        
        if len(forward_slice) < self.forward_horizon:
            return 0

        max_fwd_high = forward_slice["high"].max() if "high" in forward_slice.columns else forward_slice["close"].max()
        min_fwd_low = forward_slice["low"].min() if "low" in forward_slice.columns else forward_slice["close"].min()
        end_fwd_close = forward_slice.iloc[-1]["close"]

        up_move = max_fwd_high - curr_close
        down_move = curr_close - min_fwd_low
        net_move = end_fwd_close - curr_close

        # Bullish Breakout: Reached +12 pts and ended positive with minimal adverse pullback
        if up_move >= self.impulse_threshold_pts and net_move >= (self.impulse_threshold_pts * 0.70) and down_move <= 8.0:
            return 1 # BUY CALL

        # Bearish Breakdown: Dropped -12 pts and ended negative with minimal adverse pullback
        if down_move >= self.impulse_threshold_pts and net_move <= -(self.impulse_threshold_pts * 0.70) and up_move <= 8.0:
            return 2 # BUY PUT

        return 0 # CHOP / HOLD

    def build_dataset_from_df(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Converts full historical bars dataframe into (X_images, y_labels).
        
        Returns:
            X: np.ndarray [N, 3, 128, 128], float32 in [0, 1]
            y: np.ndarray [N], int64 with classes {0, 1, 2}
        """
        n = len(df)
        required_len = self.window_bars + self.forward_horizon
        if n < required_len:
            raise ValueError(f"DataFrame must have at least {required_len} rows, got {n}")

        logger.info(f"Generating visual candlestick dataset across {n} bars (window={self.window_bars})...")
        images = []
        labels = []

        for i in range(self.window_bars - 1, n - self.forward_horizon):
            window_df = df.iloc[i - self.window_bars + 1 : i + 1]
            lbl = self.label_window(df, i)
            
            # Render 128x128 image
            img_arr = self.renderer.render_ohlc(window_df)
            img_tensor = img_arr.transpose(2, 0, 1).astype(np.float32) / 255.0

            images.append(img_tensor)
            labels.append(lbl)

        X = np.array(images, dtype=np.float32)
        y = np.array(labels, dtype=np.int64)

        counts = np.bincount(y, minlength=3)
        logger.info(f"Dataset generated: Total={len(y)} | CHOP(0)={counts[0]}, CALL(1)={counts[1]}, PUT(2)={counts[2]}")
        return X, y

    def save_dataset(self, X: np.ndarray, y: np.ndarray, save_path: str):
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(save_path, X=X, y=y)
        logger.info(f"Dataset saved successfully to {save_path}")

    def load_dataset(self, file_path: str) -> Tuple[np.ndarray, np.ndarray]:
        data = np.load(file_path)
        return data["X"], data["y"]
