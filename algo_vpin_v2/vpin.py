"""
Layer 1: Bulk Volume Classification (BVC) & VPIN Toxicity Calculator (v2.0)
Volume-synchronized probability of toxicity estimation.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import List, Optional
import numpy as np
import pandas as pd
from scipy.stats import norm
import logging

from .config import VPINConfig

logger = logging.getLogger("algo_vpin_v2.vpin")


class ToxicityRegime(Enum):
    LOW_TOXICITY = "LOW_TOXICITY"
    NORMAL_TOXICITY = "NORMAL_TOXICITY"
    HIGH_TOXICITY = "HIGH_TOXICITY"
    EXTREME_TOXICITY = "EXTREME_TOXICITY"


@dataclass
class VolumeBucket:
    bucket_id: int
    volume: float
    buy_volume: float
    sell_volume: float
    start_timestamp: pd.Timestamp
    end_timestamp: pd.Timestamp
    vwap: float
    imbalance: float


@dataclass
class VPINResult:
    vpin: float
    regime: ToxicityRegime
    cdf_z: float
    buy_volume: float
    sell_volume: float
    completed_bucket_count: int
    current_bucket_fill_pct: float
    timestamp: pd.Timestamp


class VPINCalculator:
    def __init__(self, config: Optional[VPINConfig] = None):
        self.config = config or VPINConfig()
        self.bucket_volume = self.config.default_daily_volume / float(self.config.n_buckets)
        self.price_deltas: List[float] = []
        self.completed_buckets: List[VolumeBucket] = []
        
        self.current_bucket_vol: float = 0.0
        self.current_bucket_buy_vol: float = 0.0
        self.current_bucket_sell_vol: float = 0.0
        self.current_bucket_pv: float = 0.0
        self.current_bucket_start: Optional[pd.Timestamp] = None
        self.bucket_counter: int = 0
        self.prev_close: Optional[float] = None
        self.current_vpin: float = 0.0
        self.current_regime: ToxicityRegime = ToxicityRegime.NORMAL_TOXICITY

    def update_bucket_volume(self, avg_daily_volume: float):
        if avg_daily_volume > 0:
            self.bucket_volume = max(1000.0, avg_daily_volume / float(self.config.n_buckets))
            logger.info(f"Updated VPIN Bucket Volume V = {self.bucket_volume:.2f} (from ADV: {avg_daily_volume:.0f})")

    def _compute_sigma_delta_p(self) -> float:
        if len(self.price_deltas) < 5:
            return 1.0
        window = self.price_deltas[-self.config.sigma_delta_window:]
        std = float(np.std(window, ddof=1))
        return std if std > 1e-6 else 1.0

    def classify_bvc(self, delta_p: float, volume: float) -> tuple:
        sigma_dp = self._compute_sigma_delta_p()
        z = delta_p / sigma_dp
        cdf_z = float(norm.cdf(z))
        cdf_z = max(0.001, min(0.999, cdf_z))
        v_buy = volume * cdf_z
        v_sell = volume * (1.0 - cdf_z)
        return v_buy, v_sell, cdf_z

    def process_bar(self, close_price: float, volume: float, timestamp: pd.Timestamp) -> VPINResult:
        if volume <= 0:
            volume = 1.0
        if self.prev_close is None:
            self.prev_close = close_price
            delta_p = 0.0
        else:
            delta_p = close_price - self.prev_close
            self.prev_close = close_price

        self.price_deltas.append(delta_p)
        if len(self.price_deltas) > self.config.sigma_delta_window * 3:
            self.price_deltas = self.price_deltas[-self.config.sigma_delta_window * 2:]

        v_buy, v_sell, cdf_z = self.classify_bvc(delta_p, volume)
        remaining_vol = volume
        buy_ratio = cdf_z
        sell_ratio = 1.0 - cdf_z

        while remaining_vol > 0:
            space_in_bucket = self.bucket_volume - self.current_bucket_vol
            if self.current_bucket_start is None:
                self.current_bucket_start = timestamp

            if remaining_vol < space_in_bucket:
                chunk_buy = remaining_vol * buy_ratio
                chunk_sell = remaining_vol * sell_ratio
                self.current_bucket_vol += remaining_vol
                self.current_bucket_buy_vol += chunk_buy
                self.current_bucket_sell_vol += chunk_sell
                self.current_bucket_pv += (close_price * remaining_vol)
                remaining_vol = 0.0
            else:
                chunk_fill = space_in_bucket
                chunk_buy = chunk_fill * buy_ratio
                chunk_sell = chunk_fill * sell_ratio
                self.current_bucket_vol += chunk_fill
                self.current_bucket_buy_vol += chunk_buy
                self.current_bucket_sell_vol += chunk_sell
                self.current_bucket_pv += (close_price * chunk_fill)
                remaining_vol -= chunk_fill

                vwap = self.current_bucket_pv / self.current_bucket_vol if self.current_bucket_vol > 0 else close_price
                imbalance = abs(self.current_bucket_buy_vol - self.current_bucket_sell_vol)
                
                bucket = VolumeBucket(
                    bucket_id=self.bucket_counter,
                    volume=self.current_bucket_vol,
                    buy_volume=self.current_bucket_buy_vol,
                    sell_volume=self.current_bucket_sell_vol,
                    start_timestamp=self.current_bucket_start,
                    end_timestamp=timestamp,
                    vwap=vwap,
                    imbalance=imbalance
                )
                self.completed_buckets.append(bucket)
                self.bucket_counter += 1

                if len(self.completed_buckets) > self.config.n_buckets:
                    self.completed_buckets.pop(0)

                self._recalculate_vpin()
                self.current_bucket_vol = 0.0
                self.current_bucket_buy_vol = 0.0
                self.current_bucket_sell_vol = 0.0
                self.current_bucket_pv = 0.0
                self.current_bucket_start = timestamp

        fill_pct = (self.current_bucket_vol / self.bucket_volume) if self.bucket_volume > 0 else 0.0
        return VPINResult(
            vpin=self.current_vpin,
            regime=self.current_regime,
            cdf_z=cdf_z,
            buy_volume=v_buy,
            sell_volume=v_sell,
            completed_bucket_count=len(self.completed_buckets),
            current_bucket_fill_pct=fill_pct,
            timestamp=timestamp
        )

    def _recalculate_vpin(self):
        n = len(self.completed_buckets)
        if n == 0:
            self.current_vpin = 0.0
            return
        total_imbalance = sum(b.imbalance for b in self.completed_buckets)
        total_vol = n * self.bucket_volume
        self.current_vpin = total_imbalance / total_vol if total_vol > 0 else 0.0
        self.current_vpin = max(0.0, min(1.0, self.current_vpin))

        if self.current_vpin >= self.config.regime_extreme_threshold:
            self.current_regime = ToxicityRegime.EXTREME_TOXICITY
        elif self.current_vpin >= self.config.regime_high_threshold:
            self.current_regime = ToxicityRegime.HIGH_TOXICITY
        elif self.current_vpin <= self.config.regime_low_threshold:
            self.current_regime = ToxicityRegime.LOW_TOXICITY
        else:
            self.current_regime = ToxicityRegime.NORMAL_TOXICITY
