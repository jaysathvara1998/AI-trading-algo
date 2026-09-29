"""
Swing detection: a swing high is a candle whose high exceeds `left` candles before and `right` candles after
(so it is confirmed only `right` candles later; that latency is recorded and respected everywhere).
Consecutive swings alternate H/L; a same-kind swing replaces the previous one if more extreme, and swings
closer than `min_dist_atr` x ATR to the previous opposite swing are ignored (noise filter).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from .candles import Candle


@dataclass(frozen=True)
class Swing:
    idx: int            # candle index of the extremum
    price: float
    kind: str           # "H" or "L"
    confirmed_idx: int  # candle index at which the swing became known


def detect_swings(c: Sequence[Candle], atr_value: float, left: int = 2, right: int = 2, min_dist_atr: float = 0.5,
                  upto: int | None = None) -> List[Swing]:
    """Swings confirmed by candle index `upto` (default: last). Deterministic, no look-ahead."""
    n = len(c) if upto is None else min(len(c), upto + 1)
    out: List[Swing] = []
    min_dist = min_dist_atr * atr_value
    for i in range(left, n - right):
        hi = c[i].high; lo = c[i].low
        is_h = all(hi > c[j].high for j in range(i - left, i)) and all(hi >= c[j].high for j in range(i + 1, i + right + 1))
        is_l = all(lo < c[j].low for j in range(i - left, i)) and all(lo <= c[j].low for j in range(i + 1, i + right + 1))
        for kind, price, ok in (("H", hi, is_h), ("L", lo, is_l)):
            if not ok:
                continue
            sw = Swing(idx=i, price=price, kind=kind, confirmed_idx=i + right)
            if out and out[-1].kind == kind:
                # same kind twice: keep the more extreme one
                if (kind == "H" and price > out[-1].price) or (kind == "L" and price < out[-1].price):
                    out[-1] = sw
                continue
            if out and abs(price - out[-1].price) < min_dist:
                # too close to the previous opposite swing: noise; drop this one, and if it makes the previous
                # swing less extreme in hindsight nothing changes (we never rewrite history)
                continue
            out.append(sw)
    return out


def last_of(swings: Sequence[Swing], kind: str, n: int = 1, before_idx: int | None = None) -> List[Swing]:
    """Last n swings of a kind confirmed by candle `before_idx` (inclusive)."""
    sel = [s for s in swings if s.kind == kind and (before_idx is None or s.confirmed_idx <= before_idx)]
    return sel[-n:]
