"""
Setup lifecycle (spec section 23): LIQUIDITY_EVENT -> STRUCTURE_SHIFT -> WAITING_FOR_RETEST -> RETEST_OK ->
ENTRY_CONFIRMED -> RISK_VALIDATED, or EXPIRED / INVALIDATED with a reason. One setup per sweep id.
Setup-timeframe steps happen on setup candle closes; execution confirmation on execution candle closes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from .candles import Candle
from .liquidity import SweepEvent
from .structure import StructureEvent
from .swings import Swing


@dataclass
class Setup:
    sweep: SweepEvent
    direction: int
    state: str = "LIQUIDITY_EVENT"
    created_setup_idx: int = 0
    shift: Optional[StructureEvent] = None
    retest_seen: bool = False
    retest_idx: Optional[int] = None
    wait_exec_from: Optional[int] = None      # execution-minute index from which confirmation is awaited
    confirm_minute: Optional[int] = None
    confirm_kind: str = ""
    confirm_candle: Optional[Candle] = None
    reject_reason: str = ""
    score: int = 0
    score_parts: dict = field(default_factory=dict)
    invalidation: float = 0.0                 # structural invalidation price (sweep extreme)

    @property
    def id(self) -> str:
        return self.sweep.sweep_id


class SetupTracker:
    def __init__(self, cfg, log, tf_setup: int):
        self.cfg = cfg; self.log = log; self.tf = tf_setup
        self.active: List[Setup] = []
        self.done: List[Setup] = []

    def new_sweep(self, sw: SweepEvent, setup_idx: int, setup_candles: Sequence[Candle], ts):
        if any(s.id == sw.sweep_id for s in self.active):
            return
        s = Setup(sweep=sw, direction=sw.direction, created_setup_idx=setup_idx)
        # invalidation = extreme of the sweep excursion
        seg = setup_candles[sw.start_idx:sw.idx + 1]
        s.invalidation = max(x.high for x in seg) if sw.direction < 0 else min(x.low for x in seg)
        if not self.cfg.structure.require_shift:
            s.state = "WAITING_FOR_RETEST" if self.cfg.retest.enabled else "WAITING_CONFIRMATION"
            s.wait_exec_from = (setup_idx + 1) * self.tf
        self.active.append(s)
        self.log.log(ts, "LIQUIDITY_EVENT", setup=s.id, direction=sw.direction, level=sw.level.name, price=sw.level.price,
                     penetration_atr=sw.penetration_atr, close_location=sw.close_location, wick_ratio=sw.wick_ratio, bars_beyond=sw.bars_beyond)

    # ---------------- setup-timeframe close ----------------
    def on_setup_close(self, setup_idx: int, c_setup: Sequence[Candle], events: Sequence[StructureEvent], atr_value: float, ts):
        cur = c_setup[setup_idx]
        for s in list(self.active):
            age = setup_idx - s.created_setup_idx
            if s.state == "LIQUIDITY_EVENT":
                ev = [e for e in events if e.idx == setup_idx and e.direction == s.direction]
                if ev:
                    s.shift = ev[-1]; s.state = "WAITING_FOR_RETEST" if self.cfg.retest.enabled else "WAITING_CONFIRMATION"
                    s.wait_exec_from = (setup_idx + 1) * self.tf
                    self.log.log(ts, "STRUCTURE_SHIFT", setup=s.id, kind=ev[-1].kind, level=ev[-1].level, distance_atr=ev[-1].distance_atr)
                elif age > self.cfg.structure.max_bars_after_sweep:
                    self._expire(s, f"no structure shift within {self.cfg.structure.max_bars_after_sweep} setup candles", ts)
                # a close beyond the sweep extreme in the wrong direction kills the thesis
                elif (cur.close > s.invalidation) if s.direction < 0 else (cur.close < s.invalidation):
                    self._expire(s, "price accepted beyond the swept level", ts)
            elif s.state == "WAITING_FOR_RETEST":
                lvl = s.shift.level if s.shift else s.sweep.level.price
                zone = self.cfg.retest.zone_atr * atr_value
                inv = self.cfg.retest.invalidate_atr * atr_value
                deep = (cur.close > lvl + inv) if s.direction < 0 else (cur.close < lvl - inv)
                touched = (cur.high >= lvl - zone) if s.direction < 0 else (cur.low <= lvl + zone)
                since = setup_idx - (s.shift.idx if s.shift else s.sweep.idx)
                if deep:
                    self._expire(s, "retest reclaimed the broken structure", ts)
                elif touched:
                    s.retest_seen = True; s.retest_idx = setup_idx; s.state = "WAITING_CONFIRMATION"
                    s.wait_exec_from = (setup_idx + 1) * self.tf
                    self.log.log(ts, "RETEST", setup=s.id, level=lvl, close=cur.close)
                elif since > self.cfg.retest.expire_bars:
                    self._expire(s, f"no retest within {self.cfg.retest.expire_bars} setup candles", ts)
            elif s.state == "WAITING_CONFIRMATION":
                # still valid only while structure holds
                if (cur.close > s.invalidation) if s.direction < 0 else (cur.close < s.invalidation):
                    self._expire(s, "price accepted beyond the swept level while waiting for confirmation", ts)

    # ---------------- execution-timeframe close ----------------
    def on_exec_close(self, minute: int, c1: Sequence[Candle], exec_swings: Sequence[Swing], ts) -> List[Setup]:
        confirmed: List[Setup] = []
        cfg = self.cfg.confirm
        for s in list(self.active):
            if s.state != "WAITING_CONFIRMATION" or s.wait_exec_from is None or minute < s.wait_exec_from:
                continue
            if minute - s.wait_exec_from > cfg.max_bars:
                self._expire(s, f"no execution confirmation within {cfg.max_bars} candles", ts); continue
            cur, prev = c1[minute], c1[minute - 1]
            rng = cur.range
            if rng <= 0:
                continue
            body_ok = (cur.body / rng) >= cfg.body_frac
            dir_ok = cur.bullish if s.direction > 0 else cur.bearish
            rejection = dir_ok and body_ok and ((cur.close > prev.high) if s.direction > 0 else (cur.close < prev.low))
            micro = False
            lookback = [x for x in exec_swings if x.confirmed_idx <= minute and x.kind == ("H" if s.direction > 0 else "L")][-1:]
            if lookback:
                lv = lookback[0].price
                micro = (cur.close > lv) if s.direction > 0 else (cur.close < lv)
            if rejection or (micro and dir_ok):
                s.state = "ENTRY_CONFIRMED"; s.confirm_minute = minute; s.confirm_candle = cur
                s.confirm_kind = "rejection+micro_bos" if (rejection and micro) else ("rejection" if rejection else "micro_bos")
                confirmed.append(s)
        return confirmed

    def _expire(self, s: Setup, why: str, ts):
        s.state = "EXPIRED"; s.reject_reason = why
        self.active.remove(s); self.done.append(s)
        self.log.log(ts, "SETUP_EXPIRED", setup=s.id, reason=why)

    def resolve(self, s: Setup, state: str, why: str = ""):
        s.state = state; s.reject_reason = why
        if s in self.active:
            self.active.remove(s)
        self.done.append(s)
