"""
Per-session engine: consumes completed 1-minute candles, maintains context/setup/execution state, runs the
setup lifecycle, validates risk, and returns order intents and exit fills. Used identically by the backtester
(deterministic replay) and the live runner (spec section 24/25 pseudocode).
"""
from __future__ import annotations

import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional

from .candles import Candle, Session, aggregate, atr, parse_hhmm
from .config import EngineConfig
from .levels import LevelTracker
from .liquidity import detect_sweeps
from .logger import EventLog
from .position import Position, Fill
from .risk import plan as risk_plan
from .setup import Setup, SetupTracker
from .structure import compute_structure, regime, StructureState
from .swings import detect_swings


@dataclass
class OrderIntent:
    minute: int              # execution candle at whose close the decision was made; fill at next open
    setup: Setup
    plan: object
    score: int
    regime: str


@dataclass
class Decisions:
    intents: List[OrderIntent] = field(default_factory=list)
    fills: List[Fill] = field(default_factory=list)


class SessionEngine:
    def __init__(self, cfg: EngineConfig, session: Session, log: Optional[EventLog] = None):
        self.cfg = cfg; self.s = session; self.log = log or EventLog(None, keep=False)
        self.c1: List[Candle] = []
        self.levels = LevelTracker(cfg.levels, session)
        self.setups = SetupTracker(cfg, self.log, cfg.tf.setup)
        self.seen_sweeps = set()
        self.position: Optional[Position] = None
        self.closed: List[Position] = []
        self.regime_now = "TRANSITION"
        self.trend_ctx = "RANGE"; self.trend_setup = "RANGE"
        self.st_setup: StructureState = StructureState()
        self.swings_setup = []; self.atr_setup = 1.0; self.c_setup: List[Candle] = []
        self.level_list = []
        self.day_pnl_inr = 0.0; self.trades_today = 0; self.consec_losses = 0
        self.cooldown_until_setup_idx = -1
        self.halted_reason = ""
        self.entry_start = parse_hhmm(cfg.session.entry_start); self.entry_end = parse_hhmm(cfg.session.entry_end)
        self.flatten_min = parse_hhmm(cfg.session.flatten)

    # ---------------------------------------------------------------- main loop
    def on_candle(self, c: Candle) -> Decisions:
        self.c1.append(c); m = len(self.c1) - 1; ts = c.ts
        dec = Decisions()
        tf_s, tf_c = self.cfg.tf.setup, self.cfg.tf.context
        setup_close = (m + 1) % tf_s == 0
        ctx_close = (m + 1) % tf_c == 0

        # ---- execution timeframe: manage the open position first (adverse-first on this candle)
        if self.position and self.position.open:
            for f in self.position.on_candle(c, m, self.flatten_min):
                dec.fills.append(f); self._on_fill(f, m, ts)
        # ---- setup timeframe: swings, structure, levels, sweeps, setup lifecycle
        if setup_close:
            self.c_setup = aggregate(self.c1, tf_s, m); si = len(self.c_setup) - 1
            self.atr_setup = atr(self.c_setup, self.cfg.swing.atr_period)
            self.swings_setup = detect_swings(self.c_setup, self.atr_setup, self.cfg.swing.left, self.cfg.swing.right, self.cfg.swing.min_dist_atr)
            self.st_setup = compute_structure(self.c_setup, self.swings_setup, self.atr_setup, self.cfg.structure.bos_min_atr)
            self.trend_setup = self.st_setup.trend
            self.level_list = self.levels.build(self.c1, m, self.swings_setup, self.c_setup, self.atr_setup)
            for sw in detect_sweeps(self.c_setup, self.level_list, self.atr_setup, si, self.cfg.sweep, self.seen_sweeps):
                self.levels.record_touch(sw.level)
                self.setups.new_sweep(sw, si, self.c_setup, ts)
            self.setups.ctx = {"regime": self.regime_now, "trend_ctx": self.trend_ctx, "trend_setup": self.trend_setup}
            self.setups.on_setup_close(si, self.c_setup, self.st_setup.events, self.atr_setup, ts)
            if self.cfg.confirm.mode == "retest_close" and not (self.position and self.position.open) and not self.halted_reason \
                    and self.entry_start <= m <= self.entry_end:
                for s in [x for x in self.setups.active if x.state == "ENTRY_CONFIRMED" and x.confirm_minute == m]:
                    self._try_enter(s, m, c, dec)
            if self.position and self.position.open:
                self.position._setup_idx = si
                choch_against = any(e.idx == si and e.kind == "CHOCH" and e.direction == -self.position.direction for e in self.st_setup.events)
                for f in self.position.on_setup_close(m, c.close, self.atr_setup, self.swings_setup, choch_against):
                    dec.fills.append(f); self._on_fill(f, m, ts)
        # ---- context timeframe: regime
        if ctx_close:
            cc = aggregate(self.c1, tf_c, m)
            if len(cc) >= 6:
                a = atr(cc, self.cfg.swing.atr_period)
                sw = detect_swings(cc, a, self.cfg.swing.left, self.cfg.swing.right, self.cfg.swing.min_dist_atr)
                st = compute_structure(cc, sw, a, self.cfg.structure.bos_min_atr)
                self.trend_ctx = st.trend
                self.regime_now = regime(cc, sw, st, self.cfg.regime.fast_atr, self.cfg.regime.slow_atr, self.cfg.regime.highvol_atr_ratio, self.cfg.regime.range_overlap_atr)
        if self.position and self.position.open:
            return dec
        # ---- gates before looking for entries
        if self.halted_reason or not (self.entry_start <= m <= self.entry_end):
            return dec
        exec_swings = detect_swings(self.c1[-60:], max(1e-6, atr(self.c1[-30:], 14)), 2, 2, 0.0) if len(self.c1) >= 12 else []
        # re-index execution swings to absolute minutes
        off = len(self.c1) - len(self.c1[-60:])
        from .swings import Swing
        exec_swings = [Swing(x.idx + off, x.price, x.kind, x.confirmed_idx + off) for x in exec_swings]
        for s in self.setups.on_exec_close(m, self.c1, exec_swings, ts):
            self._try_enter(s, m, c, dec)
        return dec

    # ---------------------------------------------------------------- entry validation
    def _try_enter(self, s: Setup, m: int, c: Candle, dec: Decisions):
        ts = c.ts; cfg = self.cfg
        si = len(self.c_setup) - 1
        if si <= self.cooldown_until_setup_idx:
            self.setups.resolve(s, "REJECTED", "cooldown after last exit"); self.log.log(ts, "REJECTED", setup=s.id, reason="cooldown"); return
        if self.trades_today >= cfg.risk.max_trades_per_day:
            self.setups.resolve(s, "REJECTED", "max trades per day"); self.log.log(ts, "REJECTED", setup=s.id, reason="max trades per day"); return
        if self.regime_now == "RANGE" and cfg.structure.require_shift:
            pass   # range regime: only sweep + shift + retest setups (this chain) are allowed; nothing extra to do
        # score
        sc = cfg.score; parts = {}
        aligned_ctx = (self.trend_ctx == "BULL" and s.direction > 0) or (self.trend_ctx == "BEAR" and s.direction < 0)
        aligned_setup = (self.trend_setup == "BULL" and s.direction > 0) or (self.trend_setup == "BEAR" and s.direction < 0)
        parts["context"] = sc.context_structure if aligned_ctx else (1 if self.trend_ctx in ("TRANSITION", "RANGE") else 0)
        parts["setup_structure"] = sc.setup_structure if aligned_setup else (1 if self.trend_setup == "TRANSITION" else 0)
        parts["sweep"] = sc.sweep
        parts["shift"] = sc.shift if s.shift is not None else 0
        parts["retest"] = sc.retest if s.retest_seen else 0
        parts["micro_bos"] = sc.micro_bos if "micro" in s.confirm_kind else 0
        body = (c.body / c.range) if c.range > 0 else 0.0
        parts["momentum"] = sc.momentum if body >= cfg.confirm.body_frac else 0
        vols = [x.volume for x in self.c1[-21:-1] if x.volume > 0]
        parts["volume"] = sc.volume if (vols and c.volume > 1.2 * (sum(vols) / len(vols))) else 0
        # location: not immediately into opposing structure (>= 1 ATR to the nearest opposing level)
        from .risk import nearest_target
        tgt = nearest_target(s.direction, c.close, self.level_list, self.atr_setup, cfg.risk.min_target_atr)
        parts["location"] = sc.location if tgt is not None else 0
        s.score = sum(parts.values()); s.score_parts = parts
        if s.score < sc.entry_threshold:
            self.setups.resolve(s, "REJECTED", f"score {s.score} < {sc.entry_threshold}")
            self.log.log(ts, "REJECTED", setup=s.id, reason="score", score=s.score, parts=parts); return
        if self.regime_now == "HIGH_VOL" and cfg.risk.max_stop_atr < 4.0:
            pass  # high-vol handled by the ATR-scaled stop cap
        # risk
        plan, why = risk_plan(s.direction, c.close, s.invalidation, self.level_list, self.atr_setup, cfg.risk, cfg.cost, cfg.exit,
                              stop_reason=f"beyond sweep extreme {s.invalidation:.2f}")
        if plan is None:
            self.setups.resolve(s, "REJECTED", why); self.log.log(ts, "REJECTED", setup=s.id, reason=why, score=s.score); return
        s.state = "RISK_VALIDATED"
        self.log.log(ts, "ENTRY_CONFIRMED", setup=s.id, direction=s.direction, confirm=s.confirm_kind, score=s.score, parts=parts,
                     entry=c.close, stop=plan.stop, tp1=plan.tp1, tp2=plan.tp2, target=plan.target_name, rr=plan.rr, lots=plan.lots,
                     risk_inr=plan.risk_inr, regime=self.regime_now, trend_ctx=self.trend_ctx, trend_setup=self.trend_setup)
        dec.intents.append(OrderIntent(m, s, plan, s.score, self.regime_now))

    # ---------------------------------------------------------------- fills from the outside (backtester / live)
    def open_position(self, intent: OrderIntent, fill_price: float, fill_minute: int):
        p = Position(plan=intent.plan, entry_minute=fill_minute, entry_price=fill_price, quantity=intent.plan.quantity, setup_id=intent.setup.id)
        p._exit_cfg = self.cfg.exit; p._atr = self.atr_setup; p._setup_idx = len(self.c_setup) - 1
        # re-anchor stop/targets to the actual fill (spec 28: use the actual fill price)
        d = p.direction; slip = fill_price - intent.plan.entry
        p.plan.entry = fill_price
        p.plan.tp1 = intent.plan.tp1 + slip; p.plan.tp2 = intent.plan.tp2 + slip
        p.plan.risk_pts = abs(fill_price - p.stop)
        self.position = p; self.trades_today += 1
        self.setups.resolve(intent.setup, "POSITION_OPEN")
        self.log.log(self.c1[-1].ts + pd.Timedelta(minutes=1), "POSITION_OPEN", setup=intent.setup.id,
                     direction=d, fill=fill_price, stop=p.stop, tp1=p.plan.tp1, tp2=p.plan.tp2, quantity=p.quantity)

    def _on_fill(self, f: Fill, m: int, ts):
        self.log.log(ts, "EXIT" if f.state == "EXIT" else "PARTIAL_EXIT", reason=f.reason, price=f.price, quantity=f.quantity, minute=f.minute)
        if f.state == "EXIT" and self.position is not None:
            self.closed.append(self.position)
            self.cooldown_until_setup_idx = (len(self.c_setup) - 1) + self.cfg.risk.cooldown_bars
            self.position = None

    def record_result(self, net_inr: float):
        """Called by the backtester / live runner after a position closes, with realised P&L, for the day limits."""
        self.day_pnl_inr += net_inr
        if net_inr <= 0:
            self.consec_losses += 1
        else:
            self.consec_losses = 0
        if self.day_pnl_inr <= -self.cfg.risk.daily_loss_limit_inr:
            self.halted_reason = "daily loss limit"
        elif self.consec_losses >= self.cfg.risk.max_consecutive_losses:
            self.halted_reason = "consecutive losses"
        if self.halted_reason:
            self.log.log(self.c1[-1].ts, "HALTED", reason=self.halted_reason, day_pnl=self.day_pnl_inr)
