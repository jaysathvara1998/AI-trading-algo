"""
Layer 2B: Multi-Timeframe (MTF) Volatility & Momentum Analyst for Algo VPIN v2.0
Tracks 1-min, 3-min, and 5-min rolling spot momentum windows to eliminate
single-bar blindspots during gradual multi-bar V-reversals and sustained trends.
"""

from collections import deque
from dataclasses import dataclass
from typing import List, Optional
import numpy as np
import logging

logger = logging.getLogger("algo_vpin_v2.mtf_analyst")


@dataclass
class MTFMomentumState:
    current_spot: float
    delta_1m: float           # 1-min immediate price change
    delta_3m: float           # 3-min rolling momentum change
    delta_5m: float           # 5-min cumulative momentum change
    pct_change_5m: float      # 5-min percentage return
    rolling_bias_5m: int      # +1 Bullish, -1 Bearish, 0 Neutral
    velocity_pts_per_min: float


class MultiTimeframeAnalyst:
    """
    Maintains rolling price history across 1m, 3m, and 5m horizons.
    Detects cumulative multi-bar counter-trends, V-shape reversals, and enforces
    strict 5-minute Higher Timeframe (HTF) candle alignment before trade entries.
    """
    def __init__(self, max_window: int = 30):
        self.price_history: deque = deque(maxlen=max_window)
        self.current_state: Optional[MTFMomentumState] = None
        self.current_5m_open: float = 0.0
        self.current_5m_high: float = 0.0
        self.current_5m_low: float = 0.0
        self.last_5m_bucket: int = -1
        self.prev_5m_closed_color: str = "NEUTRAL" # "GREEN", "RED", "NEUTRAL"

    def update_bar(self, spot_price: float, timestamp=None) -> MTFMomentumState:
        """Ingests new 1-minute closed candle close price and updates rolling state & 5m candle"""
        spot = float(spot_price)
        self.price_history.append(spot)
        n = len(self.price_history)

        curr = self.price_history[-1]
        p_1m = self.price_history[-2] if n >= 2 else curr
        p_3m = self.price_history[-4] if n >= 4 else self.price_history[0]
        p_5m = self.price_history[-6] if n >= 6 else self.price_history[0]

        delta_1m = curr - p_1m
        delta_3m = curr - p_3m
        delta_5m = curr - p_5m
        pct_change_5m = (delta_5m / p_5m) if p_5m > 0 else 0.0

        # Rolling 5-min trend bias threshold (+/- 10 points)
        if delta_5m >= 10.0:
            bias = 1
        elif delta_5m <= -10.0:
            bias = -1
        else:
            bias = 0

        # Velocity in points per minute over last 5 bars
        bars_count = min(n - 1, 5)
        velocity = (delta_5m / bars_count) if bars_count > 0 else delta_1m

        # --- 5-Minute Candle Tracking ---
        if timestamp is not None and hasattr(timestamp, "minute"):
            minute = timestamp.minute
            bucket_5m = minute // 5
            if bucket_5m != self.last_5m_bucket:
                # Store previous 5m candle closed color
                if self.current_5m_open > 0:
                    if curr > self.current_5m_open:
                        self.prev_5m_closed_color = "GREEN"
                    elif curr < self.current_5m_open:
                        self.prev_5m_closed_color = "RED"
                    else:
                        self.prev_5m_closed_color = "NEUTRAL"

                self.last_5m_bucket = bucket_5m
                self.current_5m_open = spot
                self.current_5m_high = spot
                self.current_5m_low = spot
            else:
                self.current_5m_high = max(self.current_5m_high, spot)
                self.current_5m_low = min(self.current_5m_low, spot)
        else:
            if self.current_5m_open == 0:
                self.current_5m_open = spot

        self.current_state = MTFMomentumState(
            current_spot=curr,
            delta_1m=round(delta_1m, 2),
            delta_3m=round(delta_3m, 2),
            delta_5m=round(delta_5m, 2),
            pct_change_5m=round(pct_change_5m, 6),
            rolling_bias_5m=bias,
            velocity_pts_per_min=round(velocity, 2)
        )
        return self.current_state

    def validate_htf_entry_alignment(self, action: str, current_spot: float) -> (bool, str):
        """
        Enforces strict 5-Minute Higher Timeframe (HTF) Alignment before entering 1-Minute trades.
        Blocks CALL buying when 5-min candle or 5-min rolling momentum is RED / Bearish.
        Blocks PUT buying when 5-min candle or 5-min rolling momentum is GREEN / Bullish.
        """
        action_up = action.strip().upper()
        if self.current_state is None or self.current_5m_open <= 0:
            return True, "HTF_WARMING_UP"

        st = self.current_state
        is_5m_green = current_spot >= self.current_5m_open
        is_5m_red = current_spot < self.current_5m_open

        # --- CALL / BUY VALIDATION ---
        if action_up in ("BUY", "CALL", "CE"):
            # Check 1: 5-minute current candle is RED (allow 6.0 pt normal noise wick)
            if is_5m_red and (self.current_5m_open - current_spot) >= 6.0:
                return False, f"5M_HTF_MISALIGNMENT: 5-Min candle is RED (Spot {current_spot:.2f} < 5M Open {self.current_5m_open:.2f})"

            # Check 2: 5-minute rolling momentum is actively falling
            if st.delta_5m <= -8.0:
                return False, f"5M_HTF_MISALIGNMENT: 5-Min rolling delta is falling ({st.delta_5m:+.1f} pts)"

            # Check 3: 5-minute rolling bias is bearish
            if st.rolling_bias_5m == -1 and is_5m_red:
                return False, "5M_HTF_MISALIGNMENT: 5-Min rolling bias is BEARISH"

            # Check 4: Anti-Chasing / Vertical Climax Exhaustion Filter
            if st.delta_5m >= 30.0 and st.delta_1m > 3.0:
                return False, f"5M_HTF_EXHAUSTION: Spot already surged +{st.delta_5m:.1f} pts in 5m — wait for pullback to avoid climax buying"

        # --- PUT / SELL VALIDATION ---
        elif action_up in ("SELL", "PUT", "PE"):
            # Check 1: 5-minute current candle is GREEN (allow 6.0 pt normal noise wick)
            if is_5m_green and (current_spot - self.current_5m_open) >= 6.0:
                return False, f"5M_HTF_MISALIGNMENT: 5-Min candle is GREEN (Spot {current_spot:.2f} > 5M Open {self.current_5m_open:.2f})"

            # Check 2: 5-minute rolling momentum is actively surging
            if st.delta_5m >= 8.0:
                return False, f"5M_HTF_MISALIGNMENT: 5-Min rolling delta is surging ({st.delta_5m:+.1f} pts)"

            # Check 3: 5-minute rolling bias is bullish
            if st.rolling_bias_5m == 1 and is_5m_green:
                return False, "5M_HTF_MISALIGNMENT: 5-Min rolling bias is BULLISH"

            # Check 4: Anti-Chasing / Vertical Climax Exhaustion Filter
            if st.delta_5m <= -30.0 and st.delta_1m < -3.0:
                return False, f"5M_HTF_EXHAUSTION: Spot already dumped {st.delta_5m:.1f} pts in 5m — wait for pullback to avoid climax buying"

        return True, "5M_HTF_ALIGNED"

    def check_cumulative_counter_trend(
        self,
        holding_call: bool,
        holding_put: bool,
        threshold_pts: float = 20.0
    ) -> (bool, str):
        """
        Checks if the rolling 3m or 5m price movement is strongly opposing the active position.
        """
        if self.current_state is None:
            return False, ""

        state = self.current_state

        # Holding a PUT, but Spot has surged up by >= threshold_pts in 3m or 5m (V-Recovery)
        if holding_put:
            if state.delta_5m >= threshold_pts:
                reason = f"CUMULATIVE_COUNTER_TREND: Spot surged +{state.delta_5m:.1f} pts in 5 mins against PUT"
                return True, reason
            if state.delta_3m >= (threshold_pts * 0.85):
                reason = f"FAST_COUNTER_SURGE: Spot surged +{state.delta_3m:.1f} pts in 3 mins against PUT"
                return True, reason

        # Holding a CALL, but Spot has dropped by >= threshold_pts in 3m or 5m (V-Dump)
        if holding_call:
            if state.delta_5m <= -threshold_pts:
                reason = f"CUMULATIVE_COUNTER_TREND: Spot dropped {state.delta_5m:.1f} pts in 5 mins against CALL"
                return True, reason
            if state.delta_3m <= -(threshold_pts * 0.85):
                reason = f"FAST_COUNTER_SURGE: Spot dropped {state.delta_3m:.1f} pts in 3 mins against CALL"
                return True, reason

        return False, ""
