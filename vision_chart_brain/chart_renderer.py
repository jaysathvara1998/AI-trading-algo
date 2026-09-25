"""
High-Speed In-Memory Candlestick Chart Renderer (<10ms per frame)
Converts 20-candle OHLCV sequences into normalized 128x128 RGB numpy arrays
for 2D Convolutional Neural Network inference and training.
"""

from typing import List, Union
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


class FastChartRenderer:
    def __init__(self, image_size: int = 128, window_bars: int = 35):
        self.image_size = image_size
        self.window_bars = window_bars
        
        # Premium Dark Mode Theme Palette
        self.bg_color = (15, 23, 42)        # Deep Navy Slate (#0f172a)
        self.bull_color = (34, 197, 94)     # Emerald Green (#22c55e)
        self.bear_color = (239, 68, 68)     # Crimson Red (#ef4444)
        self.ema_color = (234, 179, 8)      # Golden Yellow (#eab308)
        self.grid_color = (30, 41, 59)      # Subtle Slate Grid (#1e293b)

    def render_ohlc(self, ohlcv_data: Union[pd.DataFrame, List[dict]]) -> np.ndarray:
        """
        Renders an in-memory 128x128 RGB image representing the candlestick chart.
        
        Returns:
            np.ndarray of shape (128, 128, 3), dtype=np.uint8, values in [0, 255]
        """
        if isinstance(ohlcv_data, pd.DataFrame):
            df = ohlcv_data.tail(self.window_bars).copy()
            opens = df["open"].values
            highs = df["high"].values
            lows = df["low"].values
            closes = df["close"].values
        else:
            recent = ohlcv_data[-self.window_bars:]
            opens = np.array([x["open"] for x in recent], dtype=float)
            highs = np.array([x.get("high", max(x["open"], x["close"])) for x in recent], dtype=float)
            lows = np.array([x.get("low", min(x["open"], x["close"])) for x in recent], dtype=float)
            closes = np.array([x["close"] for x in recent], dtype=float)

        n_bars = len(closes)
        if n_bars < 5:
            # Return blank dark image if insufficient bars
            return np.full((self.image_size, self.image_size, 3), 15, dtype=np.uint8)

        # Calculate dynamic price scaling
        min_p = float(np.min(lows))
        max_p = float(np.max(highs))
        p_range = max_p - min_p if (max_p - min_p) > 0.05 else 1.0

        # Margin padding (8% top and bottom)
        pad_y = self.image_size * 0.08
        draw_h = self.image_size - (2 * pad_y)

        def price_to_y(p: float) -> int:
            norm = (p - min_p) / p_range
            # Invert because y=0 is top of image
            y = int((self.image_size - pad_y) - (norm * draw_h))
            return max(0, min(self.image_size - 1, y))

        img = Image.new("RGB", (self.image_size, self.image_size), color=self.bg_color)
        draw = ImageDraw.Draw(img)

        # Draw subtle horizontal reference grids
        for level in [0.25, 0.50, 0.75]:
            grid_y = int(pad_y + (level * draw_h))
            draw.line([(0, grid_y), (self.image_size, grid_y)], fill=self.grid_color, width=1)

        # Calculate bar geometry
        bar_slot = self.image_size / float(self.window_bars)
        bar_width = max(2, int(bar_slot * 0.65))

        # Render Candlesticks (Wicks + Bodies)
        ema_points = []
        running_close = []

        for i in range(n_bars):
            center_x = int((i + 0.5) * bar_slot)
            op = opens[i]
            hi = highs[i]
            lo = lows[i]
            cl = closes[i]

            running_close.append(cl)
            # 5-period rolling fast EMA overlay
            if len(running_close) >= 3:
                ema_val = np.mean(running_close[-5:])
                ema_points.append((center_x, price_to_y(ema_val)))

            is_bull = cl >= op
            candle_color = self.bull_color if is_bull else self.bear_color

            y_hi = price_to_y(hi)
            y_lo = price_to_y(lo)
            y_op = price_to_y(op)
            y_cl = price_to_y(cl)

            # Draw Wick shadow line
            draw.line([(center_x, y_hi), (center_x, y_lo)], fill=candle_color, width=1)

            # Draw Real Body rectangle
            top_y = min(y_op, y_cl)
            bot_y = max(y_op, y_cl)
            if top_y == bot_y:
                bot_y += 1  # Ensure at least 1 pixel thickness for doji candles

            x0 = center_x - (bar_width // 2)
            x1 = center_x + (bar_width // 2)
            draw.rectangle([x0, top_y, x1, bot_y], fill=candle_color)

        # Draw EMA overlay line if available
        if len(ema_points) >= 2:
            draw.line(ema_points, fill=self.ema_color, width=1)

        return np.array(img, dtype=np.uint8)

    def render_tensor_batch(self, ohlcv_windows: List[pd.DataFrame]) -> np.ndarray:
        """
        Renders a batch of OHLCV windows to normalized float32 tensor array [B, 3, 128, 128].
        """
        images = []
        for df in ohlcv_windows:
            arr = self.render_ohlc(df)
            # Transpose (H, W, C) -> (C, H, W) and normalize to [0.0, 1.0]
            arr_norm = arr.transpose(2, 0, 1).astype(np.float32) / 255.0
            images.append(arr_norm)
        return np.array(images, dtype=np.float32)
