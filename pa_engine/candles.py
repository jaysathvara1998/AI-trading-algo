"""
Candle data, sessions, timeframe aggregation and ATR. Index convention: minute 0 = 09:15, minute 374 = 15:29.
Aggregation never uses an incomplete candle: at minute m, the k-minute candle j is available only if
(j + 1) * k - 1 <= m.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
import pytz

IST = pytz.timezone("Asia/Kolkata")
SESSION_MINUTES = 375


@dataclass(frozen=True)
class Candle:
    ts: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0

    @property
    def range(self) -> float:
        return self.high - self.low

    @property
    def body(self) -> float:
        return abs(self.close - self.open)

    @property
    def bullish(self) -> bool:
        return self.close > self.open

    @property
    def bearish(self) -> bool:
        return self.close < self.open


def minute_index(ts: pd.Timestamp) -> int:
    return (ts.hour - 9) * 60 + ts.minute - 15


def parse_hhmm(s: str) -> int:
    h, m = s.split(":")
    return (int(h) - 9) * 60 + int(m) - 15


def aggregate(c1: Sequence[Candle], k: int, upto: Optional[int] = None) -> List[Candle]:
    """Completed k-minute candles from 1-minute candles c1[0..upto]. Index j spans minutes j*k .. (j+1)*k-1."""
    n = len(c1) if upto is None else min(len(c1), upto + 1)
    if k <= 1:
        return list(c1[:n])
    out: List[Candle] = []
    full = n // k
    for j in range(full):
        g = c1[j * k:(j + 1) * k]
        out.append(Candle(ts=g[0].ts, open=g[0].open, high=max(x.high for x in g), low=min(x.low for x in g),
                          close=g[-1].close, volume=float(sum(x.volume for x in g))))
    return out


def true_ranges(c: Sequence[Candle]) -> np.ndarray:
    if not c:
        return np.array([])
    tr = np.empty(len(c))
    tr[0] = c[0].range
    for i in range(1, len(c)):
        tr[i] = max(c[i].high - c[i].low, abs(c[i].high - c[i - 1].close), abs(c[i].low - c[i - 1].close))
    return tr


def atr(c: Sequence[Candle], period: int = 14, floor: float = 1e-6) -> float:
    """Wilder-style ATR over the last `period` candles (simple mean of true ranges; robust median floor)."""
    tr = true_ranges(c)
    if len(tr) == 0:
        return floor
    w = tr[-period:]
    return float(max(floor, w.mean()))


@dataclass
class Session:
    day: date
    candles: List[Candle]                 # 1-minute, index == minute since 09:15 (gaps carried forward)
    prev_high: Optional[float] = None
    prev_low: Optional[float] = None
    prev_close: Optional[float] = None
    prev_day: Optional[date] = None

    @property
    def n(self) -> int:
        return len(self.candles)


def _to_sessions(df: pd.DataFrame, min_bars: int = 300) -> List[Session]:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize(IST)
    else:
        df["timestamp"] = df["timestamp"].dt.tz_convert(IST)
    t = df["timestamp"].dt.time
    df = df[(t >= time(9, 15)) & (t < time(15, 30))].sort_values("timestamp")
    df["day"] = df["timestamp"].dt.date
    sessions: List[Session] = []
    prev: Optional[Session] = None
    for day, g in df.groupby("day", sort=True):
        if len(g) < min_bars:
            continue
        cands: List[Candle] = []
        for r in g.itertuples(index=False):
            m = minute_index(r.timestamp)
            if m < 0 or m >= SESSION_MINUTES:
                continue
            while len(cands) < m:           # fill a missing minute with a flat carry-forward candle
                last = cands[-1].close if cands else float(r.open)
                cands.append(Candle(ts=cands[-1].ts + pd.Timedelta(minutes=1) if cands else r.timestamp, open=last, high=last, low=last, close=last))
            if len(cands) == m:
                cands.append(Candle(ts=r.timestamp, open=float(r.open), high=float(r.high), low=float(r.low), close=float(r.close), volume=float(getattr(r, "volume", 0.0) or 0.0)))
        s = Session(day=day, candles=cands)
        if prev is not None:
            s.prev_high = max(c.high for c in prev.candles); s.prev_low = min(c.low for c in prev.candles)
            s.prev_close = prev.candles[-1].close; s.prev_day = prev.day
        sessions.append(s)
        prev = s
    return sessions


def load_sessions(path: str | Path, min_bars: int = 300) -> List[Session]:
    return _to_sessions(pd.read_csv(path), min_bars)


def sessions_from_frame(df: pd.DataFrame) -> List[Session]:
    return _to_sessions(df, min_bars=1)
