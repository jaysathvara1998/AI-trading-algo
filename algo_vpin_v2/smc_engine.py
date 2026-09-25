"""
Smart Money Concepts (SMC) & Institutional Price Action Engine for Algo VPIN v2.0
Implements:
1. Swing Pivots: Higher Highs (HH), Higher Lows (HL), Lower Lows (LL), Lower Highs (LH)
2. Market Structure: Break of Structure (BOS) & Change of Character (CHoCH)
3. Fair Value Gaps (FVG) / Inefficiencies (BISI / SIBI)
4. Institutional Order Blocks (OB) & Mitigations
5. Liquidity Sweeps (PDH, PDL, Equal Highs/Lows Sweeps with Wick Rejections)
6. Premium vs Discount Dealing Range Equilibrium (0% - 50% - 100%)
"""

from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Optional, Tuple, Deque
import numpy as np
import pandas as pd


class StructureTrend(Enum):
    BULLISH = "BULLISH_STRUCTURE"    # HH + HL series
    BEARISH = "BEARISH_STRUCTURE"    # LL + LH series
    RANGING = "RANGING_STRUCTURE"


class FVGType(Enum):
    BULLISH = "BULLISH_FVG"  # Low[t] > High[t-2]
    BEARISH = "BEARISH_FVG"  # High[t] < Low[t-2]
    NONE = "NONE"


@dataclass
class FairValueGap:
    gap_type: FVGType
    top_price: float
    bottom_price: float
    midpoint: float
    bar_index: int
    is_mitigated: bool = False


@dataclass
class OrderBlock:
    ob_type: str  # "BULLISH_OB" or "BEARISH_OB"
    high: float
    low: float
    bar_index: int
    is_mitigated: bool = False


@dataclass
class SMCState:
    trend: StructureTrend
    current_hh: float
    current_hl: float
    current_lh: float
    current_ll: float
    has_bos: bool
    has_choch: bool
    bos_direction: int  # +1 Bullish BOS, -1 Bearish BOS, 0 None
    choch_direction: int  # +1 Bullish CHoCH, -1 Bearish CHoCH, 0 None
    active_fvgs: List[FairValueGap]
    active_obs: List[OrderBlock]
    in_discount_zone: bool  # Price < 50% dealing range (favorable for Calls)
    in_premium_zone: bool   # Price > 50% dealing range (favorable for Puts)
    equilibrium_price: float
    dealing_range_pos: float # 0.0 (Extreme Discount) to 1.0 (Extreme Premium)
    liquidity_sweep: Optional[str] # "PDH_SWEEP", "PDL_SWEEP", "EQUAL_HIGHS_SWEEP", etc.

    @property
    def zone(self) -> str:
        if self.in_discount_zone:
            return "DISCOUNT"
        elif self.in_premium_zone:
            return "PREMIUM"
        return "EQUILIBRIUM"

    @property
    def dealing_range_pct(self) -> float:
        return self.dealing_range_pos


class SMCEngine:
    """
    Smart Money Concepts Real-Time Analyst & Feature Extractor (Optimized for High-Throughput Stream Processing)
    """

    def __init__(self, pivot_lookback: int = 5, dealing_range_bars: int = 30):
        self.pivot_lookback = pivot_lookback
        self.dealing_range_bars = dealing_range_bars
        self.bars: Deque[Dict[str, float]] = deque(maxlen=60)
        self.total_bars_seen: int = 0
        
        # Structure Tracking
        self.swing_highs: Deque[Tuple[int, float]] = deque(maxlen=10)
        self.swing_lows: Deque[Tuple[int, float]] = deque(maxlen=10)
        self.current_trend = StructureTrend.RANGING
        
        # SMC Storage
        self.active_fvgs: List[FairValueGap] = []
        self.active_obs: List[OrderBlock] = []
        self.pdh: float = 0.0
        self.pdl: float = 0.0

    def set_prior_day_levels(self, pdh: float, pdl: float):
        self.pdh = pdh
        self.pdl = pdl

    def update_bar(self, open_p: float, high_p: float, low_p: float, close_p: float, volume: float = 0.0) -> SMCState:
        bar_idx = self.total_bars_seen
        self.total_bars_seen += 1
        bar = {
            "open": open_p,
            "high": high_p,
            "low": low_p,
            "close": close_p,
            "volume": volume,
            "bar_idx": bar_idx
        }
        self.bars.append(bar)

        # 1. Detect Pivot Highs & Lows
        self._detect_pivots()

        # 2. Market Structure (BOS & CHoCH)
        has_bos, bos_dir, has_choch, choch_dir = self._evaluate_market_structure(close_p, high_p, low_p)

        # 3. Detect Fair Value Gaps (FVG)
        self._detect_fvgs(bar_idx)

        # 4. Check FVG & OB Mitigations
        self._check_mitigations(high_p, low_p)

        # 5. Order Blocks (OB) Detection on strong BOS/FVG bars
        if has_bos or (self.active_fvgs and self.active_fvgs[-1].bar_index == bar_idx):
            self._detect_order_blocks(bar_idx, bos_dir)

        # 6. Dealing Range: Premium vs Discount Equilibrium
        range_pos, eq_price, in_disc, in_prem = self._calculate_dealing_range(close_p)

        # 7. Liquidity Sweeps
        sweep_tag = self._check_liquidity_sweeps(high_p, low_p, close_p)

        # Last confirmed swing points
        hh = self.swing_highs[-1][1] if self.swing_highs else high_p
        hl = self.swing_lows[-1][1] if self.swing_lows else low_p
        lh = self.swing_highs[-2][1] if len(self.swing_highs) >= 2 else hh
        ll = self.swing_lows[-2][1] if len(self.swing_lows) >= 2 else hl

        return SMCState(
            trend=self.current_trend,
            current_hh=hh,
            current_hl=hl,
            current_lh=lh,
            current_ll=ll,
            has_bos=has_bos,
            has_choch=has_choch,
            bos_direction=bos_dir,
            choch_direction=choch_dir,
            active_fvgs=self.active_fvgs[-5:],  # Keep recent 5 active
            active_obs=self.active_obs[-3:],    # Keep recent 3 active
            in_discount_zone=in_disc,
            in_premium_zone=in_prem,
            equilibrium_price=eq_price,
            dealing_range_pos=range_pos,
            liquidity_sweep=sweep_tag
        )

    def _detect_pivots(self):
        k = self.pivot_lookback
        n = len(self.bars)
        if n < (2 * k + 1):
            return

        idx = n - k - 1
        target_high = self.bars[idx]["high"]
        target_low = self.bars[idx]["low"]

        # Check Swing High
        is_pivot_high = True
        for i in range(idx - k, idx + k + 1):
            if i != idx and self.bars[i]["high"] >= target_high:
                is_pivot_high = False
                break
        if is_pivot_high:
            if not self.swing_highs or self.swing_highs[-1][0] != idx:
                self.swing_highs.append((idx, target_high))

        # Check Swing Low
        is_pivot_low = True
        for i in range(idx - k, idx + k + 1):
            if i != idx and self.bars[i]["low"] <= target_low:
                is_pivot_low = False
                break
        if is_pivot_low:
            if not self.swing_lows or self.swing_lows[-1][0] != idx:
                self.swing_lows.append((idx, target_low))

    def _evaluate_market_structure(self, close_p: float, high_p: float, low_p: float) -> Tuple[bool, int, bool, int]:
        has_bos = False
        bos_dir = 0
        has_choch = False
        choch_dir = 0

        if not self.swing_highs or not self.swing_lows:
            return has_bos, bos_dir, has_choch, choch_dir

        last_sh = self.swing_highs[-1][1]
        last_sl = self.swing_lows[-1][1]

        # Break of Structure (BOS) -> Continuing trend
        if self.current_trend == StructureTrend.BULLISH:
            if close_p > last_sh:
                has_bos = True
                bos_dir = 1
            elif close_p < last_sl:
                # Bearish Change of Character (CHoCH) - Uptrend Broken!
                has_choch = True
                choch_dir = -1
                self.current_trend = StructureTrend.BEARISH
        elif self.current_trend == StructureTrend.BEARISH:
            if close_p < last_sl:
                has_bos = True
                bos_dir = -1
            elif close_p > last_sh:
                # Bullish Change of Character (CHoCH) - Downtrend Broken!
                has_choch = True
                choch_dir = 1
                self.current_trend = StructureTrend.BULLISH
        else:
            # Establishing initial trend
            if close_p > last_sh:
                self.current_trend = StructureTrend.BULLISH
                has_choch = True
                choch_dir = 1
            elif close_p < last_sl:
                self.current_trend = StructureTrend.BEARISH
                has_choch = True
                choch_dir = -1

        return has_bos, bos_dir, has_choch, choch_dir

    def _detect_fvgs(self, bar_idx: int):
        if len(self.bars) < 3:
            return

        c1 = self.bars[-3]  # Candle 1
        c2 = self.bars[-2]  # Candle 2 (Displacement Candle)
        c3 = self.bars[-1]  # Candle 3

        # Bullish FVG (BISI): Candle 3 Low > Candle 1 High (Unfilled buy gap)
        if c3["low"] > c1["high"]:
            gap_size = c3["low"] - c1["high"]
            if gap_size >= (c2["high"] * 0.0003): # Significant gap threshold
                top = c3["low"]
                bottom = c1["high"]
                fvg = FairValueGap(
                    gap_type=FVGType.BULLISH,
                    top_price=top,
                    bottom_price=bottom,
                    midpoint=(top + bottom) / 2.0,
                    bar_index=bar_idx - 1
                )
                self.active_fvgs.append(fvg)

        # Bearish FVG (SIBI): Candle 3 High < Candle 1 Low (Unfilled sell gap)
        elif c3["high"] < c1["low"]:
            gap_size = c1["low"] - c3["high"]
            if gap_size >= (c2["high"] * 0.0003):
                top = c1["low"]
                bottom = c3["high"]
                fvg = FairValueGap(
                    gap_type=FVGType.BEARISH,
                    top_price=top,
                    bottom_price=bottom,
                    midpoint=(top + bottom) / 2.0,
                    bar_index=bar_idx - 1
                )
                self.active_fvgs.append(fvg)

    def _check_mitigations(self, high_p: float, low_p: float):
        # Check if active FVGs have been filled/mitigated
        for fvg in self.active_fvgs:
            if not fvg.is_mitigated:
                if fvg.gap_type == FVGType.BULLISH and low_p <= fvg.midpoint:
                    fvg.is_mitigated = True
                elif fvg.gap_type == FVGType.BEARISH and high_p >= fvg.midpoint:
                    fvg.is_mitigated = True

        # Keep only unmitigated FVGs (max 10)
        self.active_fvgs = [f for f in self.active_fvgs if not f.is_mitigated][-10:]

        for ob in self.active_obs:
            if not ob.is_mitigated:
                if ob.ob_type == "BULLISH_OB" and low_p <= ob.high:
                    ob.is_mitigated = True
                elif ob.ob_type == "BEARISH_OB" and high_p >= ob.low:
                    ob.is_mitigated = True

        self.active_obs = [o for o in self.active_obs if not o.is_mitigated][-5:]

    def _detect_order_blocks(self, bar_idx: int, direction: int):
        if len(self.bars) < 3:
            return

        # Bullish OB: Last down-close candle before aggressive green expansion
        if direction == 1:
            for i in range(len(self.bars) - 2, max(-1, len(self.bars) - 5), -1):
                b = self.bars[i]
                if b["close"] < b["open"]: # Red candle
                    ob = OrderBlock(
                        ob_type="BULLISH_OB",
                        high=b["high"],
                        low=b["low"],
                        bar_index=b["bar_idx"]
                    )
                    self.active_obs.append(ob)
                    break

        # Bearish OB: Last up-close candle before aggressive red expansion
        elif direction == -1:
            for i in range(len(self.bars) - 2, max(-1, len(self.bars) - 5), -1):
                b = self.bars[i]
                if b["close"] > b["open"]: # Green candle
                    ob = OrderBlock(
                        ob_type="BEARISH_OB",
                        high=b["high"],
                        low=b["low"],
                        bar_index=b["bar_idx"]
                    )
                    self.active_obs.append(ob)
                    break

    def _calculate_dealing_range(self, close_p: float) -> Tuple[float, float, bool, bool]:
        lookback = min(len(self.bars), self.dealing_range_bars)
        if lookback < 5:
            return 0.5, close_p, False, False

        recent_bars = list(self.bars)[-lookback:]
        highs = [b["high"] for b in recent_bars]
        lows = [b["low"] for b in recent_bars]

        range_high = max(highs)
        range_low = min(lows)
        total_range = range_high - range_low

        if total_range <= 0:
            return 0.5, close_p, False, False

        eq_price = (range_high + range_low) / 2.0
        pos = (close_p - range_low) / total_range
        pos = max(0.0, min(1.0, pos))

        in_discount = pos < 0.45  # In Discount zone (Ideal for Calls)
        in_premium = pos > 0.55   # In Premium zone (Ideal for Puts)

        return round(pos, 3), round(eq_price, 2), in_discount, in_premium

    def _check_liquidity_sweeps(self, high_p: float, low_p: float, close_p: float) -> Optional[str]:
        # 1. Sweep of Prior Day High (PDH) with Rejection (Bearish Trap)
        if self.pdh > 0:
            if high_p > self.pdh and close_p < self.pdh:
                return "PDH_LIQUIDITY_SWEEP_BEARISH_TRAP"

        # 2. Sweep of Prior Day Low (PDL) with Rejection (Bullish Trap)
        if self.pdl > 0:
            if low_p < self.pdl and close_p > self.pdl:
                return "PDL_LIQUIDITY_SWEEP_BULLISH_TRAP"

        # 3. Equal Lows / Highs Sweeps
        if len(self.swing_lows) >= 2:
            sl1, sl2 = self.swing_lows[-1][1], self.swing_lows[-2][1]
            if abs(sl1 - sl2) < (close_p * 0.0005): # Equal lows
                if low_p < min(sl1, sl2) and close_p > min(sl1, sl2):
                    return "EQUAL_LOWS_LIQUIDITY_SWEEP"

        if len(self.swing_highs) >= 2:
            sh1, sh2 = self.swing_highs[-1][1], self.swing_highs[-2][1]
            if abs(sh1 - sh2) < (close_p * 0.0005): # Equal highs
                if high_p > max(sh1, sh2) and close_p < max(sh1, sh2):
                    return "EQUAL_HIGHS_LIQUIDITY_SWEEP"

        return None
