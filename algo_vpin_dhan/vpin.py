"""
Layer 1: Bulk Volume Classification (BVC) & VPIN Engine
Implements Volume-Synchronized Probability of Informed Trading (Easley et al., 2012; Fang & Feng, 2019)
"""

from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.stats import norm

from .config import VPINConfig


class ToxicityRegime(Enum):
    LOW_TOXICITY = "LOW_TOXICITY"       # VPIN < delta_3 (0.25) -> Tighten bands, full size
    NORMAL_TOXICITY = "NORMAL_TOXICITY" # delta_3 <= VPIN <= delta_2
    HIGH_TOXICITY = "HIGH_TOXICITY"     # VPIN > delta_2 (0.55) -> Widen bands, halve size


@dataclass
class VolumeBucket:
    """Represents a completed volume-synchronized bucket"""
    bucket_id: int
    buy_volume: float
    sell_volume: float
    total_volume: float
    imbalance: float  # |V_tau^B - V_tau^S|
    end_timestamp: Optional[pd.Timestamp] = None


@dataclass
class VPINResult:
    """Output metrics from VPIN computation"""
    vpin: float
    regime: ToxicityRegime
    buy_volume_bar: float
    sell_volume_bar: float
    bucket_volume: float
    completed_buckets: int
    entry_band_multiplier: float  # Multiplier to adjust limit order spreads
    size_multiplier: float        # Multiplier to adjust position size


class VPINCalculator:
    """
    Computes Bulk Volume Classification (BVC) and tracks rolling VPIN across N volume buckets.
    """

    def __init__(self, config: Optional[VPINConfig] = None):
        self.config = config or VPINConfig()
        self.n_buckets = self.config.n_buckets
        self.sigma_window = self.config.sigma_window
        self.bucket_volume = self.config.default_bucket_volume
        self.delta_2 = self.config.delta_2
        self.delta_3 = self.config.delta_3

        # State buffers
        self.price_history: deque = deque(maxlen=self.sigma_window + 10)
        self.delta_p_history: deque = deque(maxlen=self.sigma_window)
        self.completed_buckets: deque = deque(maxlen=self.n_buckets)

        # In-progress bucket accumulation
        self.current_bucket_id = 0
        self.curr_bucket_buy_vol = 0.0
        self.curr_bucket_sell_vol = 0.0
        self.curr_bucket_filled = 0.0

        # Latest calculated values
        self.latest_vpin: Optional[float] = None
        self.latest_regime: ToxicityRegime = ToxicityRegime.NORMAL_TOXICITY

    def update_bucket_volume(self, daily_average_volume: float):
        """
        Dynamically adjusts bucket volume V = Total Daily Volume / N
        """
        if daily_average_volume > 0:
            self.bucket_volume = max(100.0, daily_average_volume / self.n_buckets)

    def classify_bar_bvc(self, price: float, prev_price: float, volume: float) -> Tuple[float, float]:
        """
        Bulk Volume Classification (BVC):
        Delta P = P_t - P_{t-1}
        Z = Delta P / sigma_{Delta P}
        V^B = V * Phi(Z)
        V^S = V * (1 - Phi(Z))
        """
        if volume <= 0:
            return 0.0, 0.0

        delta_p = price - prev_price
        self.delta_p_history.append(delta_p)

        # Compute rolling sigma(Delta P)
        if len(self.delta_p_history) >= 3:
            sigma_dp = float(np.std(self.delta_p_history, ddof=1))
            if sigma_dp < 1e-8:
                sigma_dp = 1e-4
        else:
            sigma_dp = max(abs(delta_p), 1.0)

        # Standardized price delta Z and Normal CDF Phi(Z)
        z = delta_p / sigma_dp
        # Clip Z to avoid extreme floating point saturation
        z_clipped = np.clip(z, -8.0, 8.0)
        phi_z = float(norm.cdf(z_clipped))

        buy_vol = volume * phi_z
        sell_vol = volume * (1.0 - phi_z)

        # Ensure volume conservation
        diff = volume - (buy_vol + sell_vol)
        buy_vol += diff / 2.0
        sell_vol += diff / 2.0

        return buy_vol, sell_vol

    def process_bar(
        self,
        close_price: float,
        volume: float,
        timestamp: Optional[pd.Timestamp] = None
    ) -> VPINResult:
        """
        Processes a single 1-minute OHLCV bar and updates VPIN state.
        Handles boundary splitting across volume buckets.
        """
        self.price_history.append(close_price)

        if len(self.price_history) < 2 or volume <= 0:
            buy_vol, sell_vol = volume / 2.0, volume / 2.0
        else:
            prev_price = self.price_history[-2]
            buy_vol, sell_vol = self.classify_bar_bvc(close_price, prev_price, volume)

        # Feed the classified buy/sell volumes into volume buckets with exact boundary splitting
        remaining_buy = buy_vol
        remaining_sell = sell_vol
        remaining_total = volume

        while remaining_total > 0:
            space_left = self.bucket_volume - self.curr_bucket_filled
            fill_amount = min(remaining_total, space_left)

            # Proportionally assign buy & sell portions
            frac = fill_amount / remaining_total if remaining_total > 0 else 0.0
            sub_buy = remaining_buy * frac
            sub_sell = remaining_sell * frac

            self.curr_bucket_buy_vol += sub_buy
            self.curr_bucket_sell_vol += sub_sell
            self.curr_bucket_filled += fill_amount

            remaining_buy -= sub_buy
            remaining_sell -= sub_sell
            remaining_total -= fill_amount

            # Check if bucket is full
            if self.curr_bucket_filled >= self.bucket_volume - 1e-7:
                self.current_bucket_id += 1
                imbalance = abs(self.curr_bucket_buy_vol - self.curr_bucket_sell_vol)
                bucket = VolumeBucket(
                    bucket_id=self.current_bucket_id,
                    buy_volume=self.curr_bucket_buy_vol,
                    sell_volume=self.curr_bucket_sell_vol,
                    total_volume=self.curr_bucket_filled,
                    imbalance=imbalance,
                    end_timestamp=timestamp
                )
                self.completed_buckets.append(bucket)

                # Reset current bucket
                self.curr_bucket_buy_vol = 0.0
                self.curr_bucket_sell_vol = 0.0
                self.curr_bucket_filled = 0.0

        # Calculate VPIN if we have completed buckets
        vpin_val = self._compute_vpin()
        regime, entry_mult, size_mult = self._classify_regime(vpin_val)

        self.latest_vpin = vpin_val
        self.latest_regime = regime

        return VPINResult(
            vpin=vpin_val,
            regime=regime,
            buy_volume_bar=buy_vol,
            sell_volume_bar=sell_vol,
            bucket_volume=self.bucket_volume,
            completed_buckets=len(self.completed_buckets),
            entry_band_multiplier=entry_mult,
            size_multiplier=size_mult
        )

    def _compute_vpin(self) -> float:
        """
        VPIN = sum_{tau=1}^N |V_tau^B - V_tau^S| / (N * V)
        """
        if not self.completed_buckets:
            return 0.5  # Neutral default during cold start

        total_imbalance = sum(b.imbalance for b in self.completed_buckets)
        k = len(self.completed_buckets)
        vpin = total_imbalance / (k * self.bucket_volume)
        return float(np.clip(vpin, 0.0, 1.0))

    def _classify_regime(self, vpin: float) -> Tuple[ToxicityRegime, float, float]:
        """
        Categorizes market into Toxicity Regimes and returns execution multipliers:
        - HIGH_TOXICITY (VPIN > delta_2): Widen entry bands (1.5x), halve order size (0.5x)
        - LOW_TOXICITY (VPIN < delta_3): Tighten entry bands (0.8x), full size (1.0x)
        - NORMAL_TOXICITY: Standard entry bands (1.0x), standard size (0.75x)
        """
        if vpin > self.delta_2:
            return ToxicityRegime.HIGH_TOXICITY, 1.5, 0.5
        elif vpin < self.delta_3:
            return ToxicityRegime.LOW_TOXICITY, 0.8, 1.0
        else:
            return ToxicityRegime.NORMAL_TOXICITY, 1.0, 0.75

    def reset(self):
        """Resets the state buffers for a new session or symbol"""
        self.price_history.clear()
        self.delta_p_history.clear()
        self.completed_buckets.clear()
        self.current_bucket_id = 0
        self.curr_bucket_buy_vol = 0.0
        self.curr_bucket_sell_vol = 0.0
        self.curr_bucket_filled = 0.0
        self.latest_vpin = None
        self.latest_regime = ToxicityRegime.NORMAL_TOXICITY
