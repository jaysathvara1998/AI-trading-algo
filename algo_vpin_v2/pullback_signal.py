"""
Context-first entry: 15-minute trend + pullback to the last higher low / lower high ("make or break" level)
+ 5-minute confirmation close in the trend direction. Pure functions over the session's 1-min arrays so far.

signal(o, h, l, c, m) -> (direction, level, note) evaluated at 1-min index m (must be a 5-min close: (m+1) % 5 == 0).
direction +1 = CALL, -1 = PUT, 0 = nothing. `level` is the make-or-break level (use it for the structure stop).
"""
from __future__ import annotations
import numpy as np
from typing import Tuple


def _agg(o, h, l, c, k, upto):
    n = (upto + 1) // k
    if n <= 0:
        return np.array([]), np.array([]), np.array([]), np.array([])
    O = o[:n * k:k]; C = c[k - 1:n * k:k]
    H = np.array([h[i * k:(i + 1) * k].max() for i in range(n)]); L = np.array([l[i * k:(i + 1) * k].min() for i in range(n)])
    return O, H, L, C


def signal(o, h, l, c, m: int, touch_atr: float = 0.25, min_bars15: int = 6) -> Tuple[int, float, str]:
    if (m + 1) % 5 != 0:
        return 0, 0.0, "not a 5-min close"
    O15, H15, L15, C15 = _agg(o, h, l, c, 15, m)          # completed 15-min bars only
    n15 = len(C15)
    if n15 < min_bars15:
        return 0, 0.0, "structure not formed"
    rng = H15 - L15
    atr = max(1e-6, float(np.median(rng[-8:])))
    sh = [i for i in range(1, n15 - 1) if H15[i] > H15[i - 1] and H15[i] >= H15[i + 1]]
    sl = [i for i in range(1, n15 - 1) if L15[i] < L15[i - 1] and L15[i] <= L15[i + 1]]
    hs, ls = sh[-2:], sl[-2:]
    if len(hs) < 2 or len(ls) < 2:
        return 0, 0.0, "fewer than two swings"
    up = H15[hs[1]] > H15[hs[0]] and L15[ls[1]] > L15[ls[0]]
    dn = H15[hs[1]] < H15[hs[0]] and L15[ls[1]] < L15[ls[0]]
    if not (up or dn):
        return 0, 0.0, "no trend"
    d = 1 if up else -1
    level = float(L15[ls[1]]) if up else float(H15[hs[1]])
    O5, H5, L5, C5 = _agg(o, h, l, c, 5, m)
    if len(C5) < 2:
        return 0, 0.0, "no 5-min bars"
    prev, cur = len(C5) - 2, len(C5) - 1
    # pullback: previous 5-min candle touched the level (within touch_atr) and did not close through it
    near = (L5[prev] <= level + touch_atr * atr and C5[prev] > level) if up else (H5[prev] >= level - touch_atr * atr and C5[prev] < level)
    if not near:
        return 0, level, "no pullback to the level"
    # confirmation: current 5-min candle closes in trend direction beyond the pullback candle's extreme
    conf = (C5[cur] > H5[prev]) if up else (C5[cur] < L5[prev])
    if not conf:
        return 0, level, "no confirmation close"
    return d, level, f"15-min {'up' if up else 'down'}trend, pullback to {level:.2f}, 5-min confirmation close {C5[cur]:.2f}"
