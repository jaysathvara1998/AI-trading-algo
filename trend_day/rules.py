"""
The one-page method, written as rules.

Levels     : previous day high/low/close, first 30-minute range (OR).
Day type   : ACCEPTANCE = `confirm_closes` consecutive 15-minute closes beyond the OR (and, optionally, beyond
             yesterday's range). First side to do so by `deadline` sets the direction. Otherwise it is a
             balanced day and nothing is traded.
Trade      : after acceptance, the first 15-minute pullback (a counter-direction 15-minute close) that holds
             above/below the OR. Entry on the first 5-minute candle that closes back in the direction beyond the
             previous 5-minute candle's extreme. Fill at the next 1-minute open.
Stop       : pullback extreme +/- buffer (fraction of 15-minute ATR). Hard, intrabar.
Exit       : 15-minute close beyond the last 15-minute swing (structure trail), or the flatten time.
Limits     : one trade a day (a second is allowed only after a winner, if enabled), no entry after `last_entry`.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List, Optional

from pa_engine.candles import Candle, aggregate, atr, parse_hhmm


@dataclass
class Rules:
    or_minutes: int = 30
    confirm_closes: int = 2           # consecutive 15m closes beyond the OR
    require_beyond_pdr: bool = False  # acceptance must also be beyond the previous day's range
    deadline: str = "11:30"           # day type must be decided by this 15m close
    last_entry: str = "13:30"
    flatten: str = "15:10"
    stop_buffer_atr: float = 0.10     # of 15m ATR
    max_stop_pct: float = 0.006       # skip if stop > this fraction of price (0.6%)
    trail: str = "swing"              # "swing": exit on 15m close beyond last 15m swing; "none": hold to flatten
    max_trades: int = 1
    second_only_after_win: bool = True
    entry_tf: int = 5
    min_or_ratio: float = 0.0         # OR range / average daily range of the last 10 sessions must be >= this
    max_or_ratio: float = 9.9
    min_gap_pct: float = 0.0          # |open - previous close| / previous close must be >= this


@dataclass
class DayResult:
    day: str
    day_type: str                      # "UP", "DOWN", "BALANCED"
    accept_minute: Optional[int] = None
    accept_price: Optional[float] = None
    drift_to_close: Optional[float] = None    # signed, in the accepted direction
    trades: List[dict] = field(default_factory=list)
    or_ratio: float = float("nan")
    gap_pct: float = float("nan")


def classify(c1: List[Candle], c15: List[Candle], prev_high, prev_low, r: Rules):
    """Returns (direction, accept_15m_index) or (0, None). Only completed 15m candles are used."""
    orh = max(x.high for x in c1[:r.or_minutes]); orl = min(x.low for x in c1[:r.or_minutes])
    dl = parse_hhmm(r.deadline)
    first_idx = r.or_minutes // 15
    up = dn = 0
    for i in range(first_idx, len(c15)):
        end_minute = (i + 1) * 15 - 1
        if end_minute > dl:
            break
        c = c15[i]
        a_up = c.close > orh and (not r.require_beyond_pdr or prev_high is None or c.close > prev_high)
        a_dn = c.close < orl and (not r.require_beyond_pdr or prev_low is None or c.close < prev_low)
        up = up + 1 if a_up else 0; dn = dn + 1 if a_dn else 0
        if up >= r.confirm_closes:
            return +1, i, orh, orl
        if dn >= r.confirm_closes:
            return -1, i, orh, orl
    return 0, None, orh, orl


def run_day(c1: List[Candle], prev_high, prev_low, r: Rules, day: str, avg_range: float = 0.0, prev_close: float = 0.0) -> DayResult:
    n = len(c1)
    c15 = aggregate(c1, 15)
    d, ai, orh, orl = classify(c1, c15, prev_high, prev_low, r)
    res = DayResult(day=day, day_type="UP" if d > 0 else ("DOWN" if d < 0 else "BALANCED"))
    res.or_ratio = (orh - orl) / avg_range if avg_range else float("nan")
    res.gap_pct = abs(c1[0].open - prev_close) / prev_close if prev_close else float("nan")
    if d == 0:
        return res
    if not (r.min_or_ratio <= res.or_ratio <= r.max_or_ratio) or res.gap_pct < r.min_gap_pct:
        res.day_type = "FILTERED"; return res
    accept_minute = (ai + 1) * 15 - 1
    res.accept_minute = accept_minute; res.accept_price = c1[accept_minute].close
    res.drift_to_close = (c1[min(n - 1, parse_hhmm(r.flatten))].close - res.accept_price) * d
    last_entry = parse_hhmm(r.last_entry); flatten = parse_hhmm(r.flatten)
    level = orh if d > 0 else orl
    trades = []
    state = "WAIT_PULLBACK"; pb_extreme = None; pos = None; wins = 0
    tf = r.entry_tf
    for m in range(accept_minute + 1, n):
        c = c1[m]
        # ---------- manage open position (adverse-first, hard stop)
        if pos is not None:
            hit = (c.low <= pos["stop"]) if d > 0 else (c.high >= pos["stop"])
            if hit:
                trades.append(_close(pos, m, pos["stop"], "STOP", d)); wins += trades[-1]["pts"] > 0; pos = None
            elif m >= flatten:
                trades.append(_close(pos, m, c.close, "FLATTEN", d)); wins += trades[-1]["pts"] > 0; pos = None
        if m >= flatten:
            break
        m15_close = (m + 1) % 15 == 0
        m5_close = (m + 1) % tf == 0
        if m15_close:
            i15 = (m + 1) // 15 - 1; k = c15[i15]
            # structure trail / exit on 15m close beyond the last completed 15m swing
            if pos is not None and r.trail == "swing":
                sw = _last_swing(c15, i15, d, pos["entry_i15"])
                if sw is not None:
                    lvl = sw - d * r.stop_buffer_atr * pos["atr"]
                    if (d > 0 and lvl > pos["stop"]) or (d < 0 and lvl < pos["stop"]):
                        pos["stop"] = lvl
                    if (k.close < sw) if d > 0 else (k.close > sw):
                        trades.append(_close(pos, m, k.close, "STRUCTURE", d)); wins += trades[-1]["pts"] > 0; pos = None
            # day invalidated: 15m close back inside the opening range
            if (k.close < level) if d > 0 else (k.close > level):
                if pos is not None:
                    trades.append(_close(pos, m, k.close, "INVALIDATED", d)); wins += trades[-1]["pts"] > 0; pos = None
                state = "DONE"
            elif state == "WAIT_PULLBACK" and pos is None:
                counter = (k.close < k.open) if d > 0 else (k.close > k.open)
                if counter:
                    state = "IN_PULLBACK"; pb_extreme = k.low if d > 0 else k.high
            elif state == "IN_PULLBACK" and pos is None:
                pb_extreme = min(pb_extreme, k.low) if d > 0 else max(pb_extreme, k.high)
        elif state == "IN_PULLBACK" and pos is None:
            pb_extreme = min(pb_extreme, c.low) if d > 0 else max(pb_extreme, c.high)
        # ---------- entry trigger on a 5m close
        if state == "IN_PULLBACK" and pos is None and m5_close and m <= last_entry and m + 1 < n:
            if len(trades) >= r.max_trades or (len(trades) >= 1 and r.second_only_after_win and wins < len(trades)):
                state = "DONE"; continue
            c5 = aggregate(c1, tf, m); cur, prev = c5[-1], c5[-2]
            go = (cur.close > cur.open and cur.close > prev.high) if d > 0 else (cur.close < cur.open and cur.close < prev.low)
            if go:
                a15 = atr(c15[: (m + 1) // 15], 14)
                stop = pb_extreme - d * r.stop_buffer_atr * a15
                fill = c1[m + 1].open
                if abs(fill - stop) > r.max_stop_pct * fill or (fill - stop) * d <= 0:
                    state = "DONE"; continue
                pos = {"entry_minute": m + 1, "entry": fill, "stop": stop, "stop0": stop, "atr": a15, "entry_i15": (m + 1) // 15, "pb_extreme": pb_extreme}
                state = "WAIT_PULLBACK"
    if pos is not None:
        trades.append(_close(pos, n - 1, c1[n - 1].close, "EOD", d))
    for t in trades:
        t["day"] = day; t["direction"] = d; t["or_ratio"] = res.or_ratio; t["gap_pct"] = res.gap_pct; t["accept_minute"] = accept_minute
    res.trades = trades
    return res


def _last_swing(c15, i15, d, entry_i15):
    """Most recent 15m swing low (long) / high (short) completed since the entry candle, 1 bar each side."""
    for i in range(i15 - 1, max(entry_i15 - 1, 0), -1):
        if d > 0 and c15[i].low < c15[i - 1].low and c15[i].low <= c15[i + 1].low:
            return c15[i].low
        if d < 0 and c15[i].high > c15[i - 1].high and c15[i].high >= c15[i + 1].high:
            return c15[i].high
    return None


def _close(pos, m, price, reason, d):
    return {"entry_minute": pos["entry_minute"], "entry": pos["entry"], "stop0": pos["stop0"], "risk_pts": abs(pos["entry"] - pos["stop0"]),
            "exit_minute": m, "exit": price, "reason": reason, "pts": (price - pos["entry"]) * d, "hold_min": m - pos["entry_minute"]}
