"""
Multi-Timeframe (MTF) Real-Time Candle Aggregator & Pyramid Engine for Algo VPIN v2.0
Builds and updates 5M, 15M, 1H, and 4H higher-timeframe structures in real-time from 1M streaming bars.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger("algo_vpin_v2.mtf_aggregator")


@dataclass
class MTFCandle:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    timeframe: str


@dataclass
class MTFState:
    trend_4h: str            # 'BULLISH', 'BEARISH', 'SIDEWAYS'
    trend_1h: str            # 'BULLISH', 'BEARISH', 'SIDEWAYS'
    trend_15m: str           # 'BULLISH', 'BEARISH', 'SIDEWAYS'
    trend_5m: str            # 'BULLISH', 'BEARISH', 'SIDEWAYS'
    ema9_5m: float
    ema20_5m: float
    ema20_1h: float
    major_resistance_1h: float
    major_support_1h: float
    distance_to_res_pts: float
    distance_to_sup_pts: float
    is_aligned_bullish: bool
    is_aligned_bearish: bool


class MultiTimeframeAggregator:
    """
    Maintains a high-performance rolling memory buffer of 1-minute bars
    and continuously compiles 5M, 15M, 1H, and 4H candles with zero network latency.
    """
    def __init__(self, max_1m_bars: int = 2000):
        self.max_1m_bars = max_1m_bars
        self.bars_1m: List[Dict] = []
        self.df_1m: pd.DataFrame = pd.DataFrame()

    def ingest_1min_bar(self, timestamp: datetime, open_p: float, high_p: float, low_p: float, close_p: float, volume: float):
        """Adds a newly closed 1-minute candle to the rolling buffer"""
        self.bars_1m.append({
            "timestamp": timestamp,
            "open": float(open_p),
            "high": float(high_p),
            "low": float(low_p),
            "close": float(close_p),
            "volume": float(volume)
        })
        if len(self.bars_1m) > self.max_1m_bars:
            self.bars_1m.pop(0)

    def ingest_dataframe(self, df: pd.DataFrame):
        """Bulk loads historical 1-minute bars into aggregator"""
        if df is None or df.empty:
            return
        clean_records = []
        for _, row in df.iterrows():
            clean_records.append({
                "timestamp": row["timestamp"],
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row.get("volume", 0.0))
            })
        self.bars_1m = clean_records[-self.max_1m_bars:]

    def get_timeframe_df(self, rule: str) -> pd.DataFrame:
        """
        Resamples rolling 1M bars into 5M ('5min'), 15M ('15min'), 1H ('1h'), or 4H ('4h').
        """
        if not self.bars_1m:
            return pd.DataFrame()

        df = pd.DataFrame(self.bars_1m)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp').sort_index()

        offset_val = '15min' if rule in ['1h', '4h'] else '0min'
        resampled = df.resample(rule, origin='start_day', offset=offset_val).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()

        return resampled

    def compute_mtf_state(self, current_spot: float) -> MTFState:
        """
        Calculates higher timeframe trends, 5M/1H EMAs, and key Major S/R levels.
        """
        df_5m = self.get_timeframe_df('5min')
        df_15m = self.get_timeframe_df('15min')
        df_1h = self.get_timeframe_df('1h')
        df_4h = self.get_timeframe_df('4h')

        def get_trend(df: pd.DataFrame) -> str:
            if df.empty or len(df) < 4:
                if len(df) >= 3:
                    closes = df['close'].values
                    if closes[-2] > closes[-3]:
                        return "BULLISH"
                    elif closes[-2] < closes[-3]:
                        return "BEARISH"
                return "NEUTRAL"
            closes = df['close'].values
            # Evaluates confirmed completed candles to prevent in-progress repainting
            if closes[-2] > closes[-3] and closes[-3] >= closes[-4]:
                return "BULLISH"
            elif closes[-2] < closes[-3] and closes[-3] <= closes[-4]:
                return "BEARISH"
            return "SIDEWAYS"

        trend_4h = get_trend(df_4h)
        trend_1h = get_trend(df_1h)
        trend_15m = get_trend(df_15m)
        trend_5m = get_trend(df_5m)

        # 5M EMAs
        ema9_5m = float(df_5m['close'].ewm(span=9, adjust=False).mean().iloc[-1]) if len(df_5m) >= 9 else current_spot
        ema20_5m = float(df_5m['close'].ewm(span=20, adjust=False).mean().iloc[-1]) if len(df_5m) >= 20 else current_spot

        # 1H EMAs and Swing S/R
        ema20_1h = float(df_1h['close'].ewm(span=20, adjust=False).mean().iloc[-1]) if len(df_1h) >= 20 else current_spot
        major_res_1h = float(df_1h['high'].tail(10).max()) if not df_1h.empty else current_spot + 100.0
        major_sup_1h = float(df_1h['low'].tail(10).min()) if not df_1h.empty else current_spot - 100.0

        dist_res = max(0.0, major_res_1h - current_spot)
        dist_sup = max(0.0, current_spot - major_sup_1h)

        is_aligned_bull = (trend_5m == "BULLISH" and trend_15m in ("BULLISH", "SIDEWAYS") and trend_1h in ("BULLISH", "SIDEWAYS") and current_spot > ema20_5m)
        is_aligned_bear = (trend_5m == "BEARISH" and trend_15m in ("BEARISH", "SIDEWAYS") and trend_1h in ("BEARISH", "SIDEWAYS") and current_spot < ema20_5m)

        return MTFState(
            trend_4h=trend_4h,
            trend_1h=trend_1h,
            trend_15m=trend_15m,
            trend_5m=trend_5m,
            ema9_5m=round(ema9_5m, 2),
            ema20_5m=round(ema20_5m, 2),
            ema20_1h=round(ema20_1h, 2),
            major_resistance_1h=round(major_res_1h, 2),
            major_support_1h=round(major_sup_1h, 2),
            distance_to_res_pts=round(dist_res, 2),
            distance_to_sup_pts=round(dist_sup, 2),
            is_aligned_bullish=is_aligned_bull,
            is_aligned_bearish=is_aligned_bear
        )
