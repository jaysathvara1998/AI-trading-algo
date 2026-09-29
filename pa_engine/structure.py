"""
Market structure from swings: HH/HL/LH/LL labelling, trend state, break of structure (BOS: continuation)
and change of character (CHOCH: first break against the prevailing structure), plus context regime.
Breaks require a candle CLOSE beyond the swing by at least `bos_min_atr` x ATR (noise band); each swing
can be broken once.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from .candles import Candle, atr as atr_of
from .swings import Swing, detect_swings, last_of


@dataclass(frozen=True)
class StructureEvent:
    idx: int                # candle index of the breaking close
    kind: str               # "BOS" or "CHOCH"
    direction: int          # +1 bullish break, -1 bearish break
    level: float            # the swing price that was broken
    swing_idx: int
    distance_atr: float


@dataclass
class StructureState:
    trend: str = "RANGE"                    # BULL | BEAR | RANGE | TRANSITION
    labels: List[str] = field(default_factory=list)   # e.g. ["HH","HL","HH",...] in swing order
    events: List[StructureEvent] = field(default_factory=list)
    last_high: Optional[Swing] = None
    last_low: Optional[Swing] = None


def label_swings(swings: Sequence[Swing]) -> List[str]:
    labels: List[str] = []
    prev_h: Optional[float] = None; prev_l: Optional[float] = None
    for s in swings:
        if s.kind == "H":
            labels.append("HH" if prev_h is not None and s.price > prev_h else ("LH" if prev_h is not None else "H"))
            prev_h = s.price
        else:
            labels.append("HL" if prev_l is not None and s.price > prev_l else ("LL" if prev_l is not None else "L"))
            prev_l = s.price
    return labels


def trend_from_labels(labels: Sequence[str]) -> str:
    recent = [l for l in labels if l in ("HH", "HL", "LH", "LL")][-4:]
    if len(recent) < 2:
        return "RANGE"
    ups = sum(1 for l in recent if l in ("HH", "HL")); dns = sum(1 for l in recent if l in ("LH", "LL"))
    last2 = recent[-2:]
    if all(l in ("HH", "HL") for l in last2) and ups >= 3:
        return "BULL"
    if all(l in ("LH", "LL") for l in last2) and dns >= 3:
        return "BEAR"
    if ups and dns and (recent[-1] in ("HH", "HL")) != (recent[-2] in ("HH", "HL")):
        return "TRANSITION"
    return "RANGE"


def compute_structure(c: Sequence[Candle], swings: Sequence[Swing], atr_value: float, bos_min_atr: float = 0.15,
                      upto: int | None = None) -> StructureState:
    """Walk candles in order; at each close, check for a break of the latest confirmed swing of each kind."""
    n = len(c) if upto is None else min(len(c), upto + 1)
    st = StructureState()
    broken = set()
    band = bos_min_atr * atr_value
    order = sorted(swings, key=lambda s: s.confirmed_idx)
    known: List[Swing] = []; hs: List[Swing] = []; ls: List[Swing] = []
    ptr = 0; trend = "RANGE"
    for i in range(n):
        # candidate swings become "known" only at their confirmation index
        changed = False
        while ptr < len(order) and order[ptr].confirmed_idx <= i:
            s = order[ptr]; known.append(s); (hs if s.kind == "H" else ls).append(s); ptr += 1; changed = True
        if changed:
            known.sort(key=lambda s: s.idx); hs.sort(key=lambda s: s.idx); ls.sort(key=lambda s: s.idx)
            st.labels = label_swings(known)
            trend = trend_from_labels(st.labels)
            st.last_high = hs[-1] if hs else None; st.last_low = ls[-1] if ls else None
        close = c[i].close
        # bullish break of the latest unbroken swing high
        for s in reversed(hs):
            if (s.idx, "H") in broken:
                break
            if close > s.price + band:
                kind = "BOS" if trend == "BULL" else "CHOCH"
                st.events.append(StructureEvent(i, kind, +1, s.price, s.idx, (close - s.price) / atr_value if atr_value else 0.0))
                broken.add((s.idx, "H"))
            break
        for s in reversed(ls):
            if (s.idx, "L") in broken:
                break
            if close < s.price - band:
                kind = "BOS" if trend == "BEAR" else "CHOCH"
                st.events.append(StructureEvent(i, kind, -1, s.price, s.idx, (s.price - close) / atr_value if atr_value else 0.0))
                broken.add((s.idx, "L"))
            break
        st.trend = trend
    return st


def regime(c_ctx: Sequence[Candle], sw_ctx: Sequence[Swing], st_ctx: StructureState, fast_atr: int, slow_atr: int,
           highvol_ratio: float, overlap_atr: float) -> str:
    """Context regime: BULL_TREND | BEAR_TREND | RANGE | TRANSITION | HIGH_VOL."""
    if len(c_ctx) < slow_atr:
        return "TRANSITION"
    a_fast = atr_of(c_ctx, fast_atr); a_slow = atr_of(c_ctx, slow_atr)
    if a_slow > 0 and a_fast / a_slow >= highvol_ratio:
        return "HIGH_VOL"
    hs = last_of(sw_ctx, "H", 2); ls = last_of(sw_ctx, "L", 2)
    if len(hs) == 2 and len(ls) == 2:
        tol = overlap_atr * a_slow
        if abs(hs[0].price - hs[1].price) <= tol and abs(ls[0].price - ls[1].price) <= tol:
            return "RANGE"
    return {"BULL": "BULL_TREND", "BEAR": "BEAR_TREND", "RANGE": "RANGE", "TRANSITION": "TRANSITION"}[st_ctx.trend]
