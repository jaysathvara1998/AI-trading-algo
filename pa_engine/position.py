"""
Position manager: deterministic exit state machine (spec sections 18-21, 25).
States: INITIAL -> PROFIT_1R -> PROFIT_2R -> RUNNER -> EXIT. Checks on every execution candle use the
adverse extreme first (stop), then the favourable extreme (targets), then the close (time / invalidation /
session rules). Trailing is updated on setup-timeframe closes from confirmed structure and/or ATR; the stop
is never loosened.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from .candles import Candle
from .risk import RiskPlan
from .swings import Swing


@dataclass
class Fill:
    minute: int
    price: float           # underlying price at which this portion exited
    quantity: int
    reason: str
    state: str


@dataclass
class Position:
    plan: RiskPlan
    entry_minute: int
    entry_price: float
    quantity: int
    setup_id: str
    stop: float = 0.0
    state: str = "INITIAL"
    remaining: int = 0
    fills: List[Fill] = field(default_factory=list)
    best: float = 0.0
    tp1_done: bool = False
    tp2_done: bool = False
    bars_since_entry_setup: int = 0
    stop_history: List[tuple] = field(default_factory=list)

    def __post_init__(self):
        self.stop = self.plan.stop; self.remaining = self.quantity; self.best = self.entry_price
        self.stop_history.append((self.entry_minute, self.stop, "initial"))

    @property
    def direction(self) -> int:
        return self.plan.direction

    @property
    def open(self) -> bool:
        return self.remaining > 0 and self.state != "EXIT"

    def r_at(self, price: float) -> float:
        return (price - self.entry_price) * self.direction / self.plan.risk_pts if self.plan.risk_pts else 0.0

    def _tighten(self, new_stop: float, minute: int, why: str) -> bool:
        if (self.direction > 0 and new_stop > self.stop) or (self.direction < 0 and new_stop < self.stop):
            self.stop = new_stop; self.stop_history.append((minute, new_stop, why)); return True
        return False

    # ---------------- execution-timeframe update ----------------
    def on_candle(self, c: Candle, minute: int, flatten_minute: int) -> List[Fill]:
        """Adverse-first evaluation of the completed execution candle."""
        out: List[Fill] = []
        if not self.open:
            return out
        d = self.direction
        adverse = c.low if d > 0 else c.high
        favourable = c.high if d > 0 else c.low
        # 1. stop
        if (adverse <= self.stop) if d > 0 else (adverse >= self.stop):
            out.append(self._close_all(minute, self.stop, "STOP" if self.state == "INITIAL" else "TRAIL_STOP")); return out
        # 2. targets (partials) on the favourable extreme
        exit_cfg = self._exit_cfg
        if not self.tp1_done and ((favourable >= self.plan.tp1) if d > 0 else (favourable <= self.plan.tp1)):
            self.tp1_done = True; self.state = "PROFIT_1R"
            q = int(round(self.quantity * exit_cfg.tp1_fraction))
            if q > 0 and q >= self.remaining:
                out.append(self._close_all(minute, self.plan.tp1, "TARGET")); return out
            if q > 0:
                out.append(self._partial(minute, self.plan.tp1, q, "TP1"))
            if exit_cfg.breakeven_after_r and exit_cfg.breakeven_after_r <= 1.0:
                self._tighten(self.entry_price + d * exit_cfg.breakeven_buffer_atr * self._atr, minute, "breakeven")
        if not self.tp2_done and ((favourable >= self.plan.tp2) if d > 0 else (favourable <= self.plan.tp2)):
            self.tp2_done = True; self.state = "PROFIT_2R"
            q = int(round(self.quantity * exit_cfg.tp2_fraction))
            if q > 0 and q < self.remaining:
                out.append(self._partial(minute, self.plan.tp2, q, "TP2"))
            elif exit_cfg.tp2_fraction >= 1.0 or self.remaining <= 0 or exit_cfg.trail == "none":
                out.append(self._close_all(minute, self.plan.tp2, "TARGET")); return out
            if self.remaining > 0:
                self.state = "RUNNER"
        self.best = max(self.best, favourable) if d > 0 else min(self.best, favourable)
        # 3. session flatten on the close
        if minute >= flatten_minute:
            out.append(self._close_all(minute, c.close, "SESSION_FLATTEN"))
        return out

    # ---------------- setup-timeframe update (structure / trailing / time) ----------------
    def on_setup_close(self, minute: int, close: float, atr_setup: float, swings: List[Swing], choch_against: bool) -> List[Fill]:
        out: List[Fill] = []
        if not self.open:
            return out
        self._atr = atr_setup
        self.bars_since_entry_setup += 1
        d = self.direction; ex = self._exit_cfg
        r_now = self.r_at(close); r_best = self.r_at(self.best)
        if ex.invalidation_exit and choch_against and self.state == "INITIAL":
            out.append(self._close_all(minute, close, "STRUCTURE_INVALIDATED")); return out
        if ex.time_exit_bars and self.bars_since_entry_setup >= ex.time_exit_bars and r_best < ex.time_exit_progress_r:
            out.append(self._close_all(minute, close, "TIME_EXIT")); return out
        if ex.breakeven_after_r and r_best >= ex.breakeven_after_r:
            self._tighten(self.entry_price + d * ex.breakeven_buffer_atr * atr_setup, minute, "breakeven")
        if ex.trail != "none" and r_best >= ex.trail_after_r:
            if self.state == "INITIAL":
                self.state = "PROFIT_1R"
            cand = None
            if ex.trail in ("structure", "hybrid"):
                sw = [s for s in swings if s.confirmed_idx <= self._setup_idx and s.kind == ("L" if d > 0 else "H")]
                if sw:
                    lvl = sw[-1].price
                    cand = lvl - d * ex.breakeven_buffer_atr * atr_setup
                    if (d > 0 and cand <= self.stop) or (d < 0 and cand >= self.stop):
                        cand = None
            if cand is None and ex.trail in ("atr", "hybrid"):
                cand = self.best - d * ex.trail_atr_factor * atr_setup
            if cand is not None:
                self._tighten(cand, minute, f"trail:{ex.trail}")
        return out

    def _partial(self, minute: int, price: float, q: int, reason: str) -> Fill:
        self.remaining -= q
        return Fill(minute, price, q, reason, self.state)

    def _close_all(self, minute: int, price: float, reason: str) -> Fill:
        q = self.remaining; self.remaining = 0; self.state = "EXIT"
        return Fill(minute, price, q, reason, "EXIT")

    # injected by the engine
    _exit_cfg = None
    _atr: float = 1.0
    _setup_idx: int = 0
