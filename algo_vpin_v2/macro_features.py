"""
Multi-Timeframe Macro Feature Engine for Algo VPIN v2.0
Computes key institutional price levels:
- Previous Day High (PDH) & Previous Day Low (PDL) Proximity
- 1-Week High & Low Range Position (0.0 to 1.0)
- 15-Minute Higher Timeframe Trend (EMA 20 vs EMA 50)
- Intraday Open Delta & Session Extreme Tracking
"""

from dataclasses import dataclass
from typing import List, Optional
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger("algo_vpin_v2.macro_features")


@dataclass
class MacroFeatureState:
    """Multi-Timeframe feature state representation"""
    pdh: float                      # Previous Day High
    pdl: float                      # Previous Day Low
    week_high: float                # 1-Week High
    week_low: float                 # 1-Week Low
    day_open: float                 # Today's Open Price (09:15)
    ema_15m_fast: float             # 15-min Fast EMA (20)
    ema_15m_slow: float             # 15-min Slow EMA (50)
    dist_pdh_pct: float             # (Price - PDH) / PDH
    dist_pdl_pct: float             # (Price - PDL) / PDL
    week_range_position: float      # Normalized (0.0 = Weekly Low, 1.0 = Weekly High)
    trend_15m_bias: int             # +1 Bullish, -1 Bearish, 0 Neutral


class MacroFeatureEngine:
    """
    Maintains higher-timeframe state across daily, weekly, and 15-min bars.
    """
    def __init__(self):
        self.pdh: Optional[float] = None
        self.pdl: Optional[float] = None
        self.pdc: Optional[float] = None
        self.week_high: Optional[float] = None
        self.week_low: Optional[float] = None
        self.day_open: Optional[float] = None
        
        # 15-min candle aggregation
        self.bars_15m: List[float] = []
        self._current_15m_open: Optional[float] = None
        self._current_15m_high: float = -np.inf
        self._current_15m_low: float = np.inf
        self._current_15m_close: Optional[float] = None
        self._bar_count_in_15m: int = 0
        
        # EMAs
        self.ema_20_val: Optional[float] = None
        self.ema_50_val: Optional[float] = None

    def initialize_from_history(self, hist_df: pd.DataFrame):
        """
        Initializes PDH, PDL, PDC, Weekly High/Low from past multi-day historical data.
        """
        if hist_df is None or hist_df.empty:
            return

        try:
            highs = hist_df["high"].astype(float).values
            lows = hist_df["low"].astype(float).values
            closes = hist_df["close"].astype(float).values

            # Assume past 375 bars is previous day
            if len(hist_df) >= 375:
                prev_day = hist_df.iloc[-750:-375] if len(hist_df) >= 750 else hist_df.iloc[:375]
                self.pdh = float(prev_day["high"].max())
                self.pdl = float(prev_day["low"].min())
                self.pdc = float(prev_day["close"].iloc[-1])
            else:
                self.pdh = float(np.max(highs))
                self.pdl = float(np.min(lows))
                self.pdc = float(closes[-1])

            self.week_high = float(np.max(highs))
            self.week_low = float(np.min(lows))
            self.day_open = float(closes[0])

            # Seed 15-min EMAs
            self.ema_20_val = float(np.mean(closes[-20:])) if len(closes) >= 20 else closes[-1]
            self.ema_50_val = float(np.mean(closes[-50:])) if len(closes) >= 50 else closes[-1]

            logger.info(
                f"[Macro Feature Engine] Initialized: PDH={self.pdh:.2f}, PDL={self.pdl:.2f}, "
                f"WeekHigh={self.week_high:.2f}, WeekLow={self.week_low:.2f}"
            )
        except Exception as e:
            logger.warning(f"Could not initialize macro features from history: {e}")

    def update_1min_bar(self, close_price: float, high_price: float, low_price: float, is_session_open: bool = False) -> MacroFeatureState:
        """
        Updates 15-min aggregation and calculates real-time MacroFeatureState.
        """
        if self.day_open is None or is_session_open:
            self.day_open = close_price

        # Update dynamic bounds if not set
        if self.pdh is None:
            self.pdh = high_price * 1.005
        if self.pdl is None:
            self.pdl = low_price * 0.995
        if self.week_high is None:
            self.week_high = max(high_price, self.pdh * 1.01)
        if self.week_low is None:
            self.week_low = min(low_price, self.pdl * 0.99)

        # 15-minute bar builder
        self._bar_count_in_15m += 1
        self._current_15m_high = max(self._current_15m_high, high_price)
        self._current_15m_low = min(self._current_15m_low, low_price)
        self._current_15m_close = close_price

        if self._bar_count_in_15m >= 15:
            # 15-min bar completed: Update EMAs
            self._update_15m_ema(close_price)
            self._bar_count_in_15m = 0
            self._current_15m_high = -np.inf
            self._current_15m_low = np.inf

        # Distance to PDH & PDL
        dist_pdh = (close_price - self.pdh) / self.pdh
        dist_pdl = (close_price - self.pdl) / self.pdl

        # 1-Week Range Position (0.0 to 1.0)
        week_span = self.week_high - self.week_low
        week_pos = (close_price - self.week_low) / week_span if week_span > 0 else 0.5
        week_pos = max(0.0, min(1.0, week_pos))

        # 15-min Trend Direction
        fast = self.ema_20_val or close_price
        slow = self.ema_50_val or close_price
        if fast > slow * 1.0005:
            trend_bias = 1
        elif fast < slow * 0.9995:
            trend_bias = -1
        else:
            trend_bias = 0

        return MacroFeatureState(
            pdh=self.pdh,
            pdl=self.pdl,
            week_high=self.week_high,
            week_low=self.week_low,
            day_open=self.day_open,
            ema_15m_fast=fast,
            ema_15m_slow=slow,
            dist_pdh_pct=dist_pdh,
            dist_pdl_pct=dist_pdl,
            week_range_position=week_pos,
            trend_15m_bias=trend_bias
        )

    def _update_15m_ema(self, close_price: float):
        k_20 = 2.0 / (20.0 + 1.0)
        k_50 = 2.0 / (50.0 + 1.0)
        if self.ema_20_val is None:
            self.ema_20_val = close_price
        else:
            self.ema_20_val = (close_price * k_20) + (self.ema_20_val * (1.0 - k_20))

        if self.ema_50_val is None:
            self.ema_50_val = close_price
        else:
            self.ema_50_val = (close_price * k_50) + (self.ema_50_val * (1.0 - k_50))
