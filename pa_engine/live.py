"""
Live / paper runner (spec sections 25-27). One process per symbol. Each minute it pulls the last completed
1-minute spot candle from Kite, feeds it to the same SessionEngine the backtester uses, and acts on the
decisions: entry intents become option buys (paper: simulated at the live premium; live: MIS market orders),
exit fills become option sells. Safety: PAPER_TRADING flag, kill-switch file, token check, one position,
engine day limits, session flatten, persisted state and event log under runtime/.
"""
from __future__ import annotations

import json
import logging
import signal
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

from .broker_kite import IST, Kite
from .candles import Candle, Session, SESSION_MINUTES, minute_index
from .config import EngineConfig
from .costs import statutory
from .engine import OrderIntent, SessionEngine
from .env import ROOT, env_bool, load_env
from .logger import EventLog
from .telegram import Telegram

log = logging.getLogger("pa_engine.live")
RUNTIME = ROOT / "runtime"
KILL_FILE = RUNTIME / "STOP"


class LiveRunner:
    def __init__(self, cfg: EngineConfig, paper: bool, poll: float = 2.0):
        self.cfg = cfg; self.paper = paper; self.poll = poll
        RUNTIME.mkdir(exist_ok=True)
        self.day = datetime.now(IST).date()
        self.tg = Telegram()
        self.kite = Kite(cfg.symbol)
        lot = self.kite.lot_size()
        if lot and lot != cfg.risk.lot_size:
            log.info(f"lot size from instrument master: {lot} (config {cfg.risk.lot_size})"); self.cfg = cfg.override(**{"risk.lot_size": lot})
        self.events = EventLog(RUNTIME / f"events_{cfg.symbol}_{self.day}.jsonl", keep=False)
        self.trades_path = RUNTIME / f"trades_{cfg.symbol}.csv"
        self.session = Session(day=self.day, candles=[])
        self._prev_day_levels()
        self.engine = SessionEngine(self.cfg, self.session, self.events)
        self.pending: Optional[OrderIntent] = None
        self.contract: Optional[dict] = None       # option contract of the open position
        self.entry_premium = 0.0; self.entry_ts = None; self.pnl_day = 0.0
        self.stop = False
        signal.signal(signal.SIGINT, lambda *_: setattr(self, "stop", True)); signal.signal(signal.SIGTERM, lambda *_: setattr(self, "stop", True))

    # ---------------- setup
    def _prev_day_levels(self):
        now = datetime.now(IST)
        df = self.kite.minute_bars(now - timedelta(days=7), now.replace(hour=9, minute=14))
        if df.empty:
            log.warning("no previous-day bars; PDH/PDL unavailable"); return
        df["d"] = df["timestamp"].dt.date
        prev = df[df["d"] < self.day]
        if prev.empty:
            return
        last = prev[prev["d"] == prev["d"].max()]
        self.session.prev_high = float(last["high"].max()); self.session.prev_low = float(last["low"].min())
        self.session.prev_close = float(last["close"].iloc[-1]); self.session.prev_day = prev["d"].max()
        log.info(f"previous session {self.session.prev_day}: PDH {self.session.prev_high} PDL {self.session.prev_low}")

    def _catch_up(self):
        """Feed every completed candle of today that the engine has not seen yet."""
        now = datetime.now(IST)
        want_upto = now.replace(second=0, microsecond=0) - timedelta(minutes=1)
        have = len(self.session.candles)
        if minute_index(pd.Timestamp(want_upto)) < have:
            return
        start = now.replace(hour=9, minute=15, second=0, microsecond=0)
        df = self.kite.minute_bars(start, now)
        for r in df.itertuples(index=False):
            m = minute_index(r.timestamp)
            if m < 0 or m >= SESSION_MINUTES or r.timestamp >= pd.Timestamp(now.replace(second=0, microsecond=0)):
                continue
            while len(self.session.candles) < m:
                last = self.session.candles[-1]
                self.session.candles.append(Candle(ts=last.ts + pd.Timedelta(minutes=1), open=last.close, high=last.close, low=last.close, close=last.close))
                self._step(self.session.candles[-1])
            if len(self.session.candles) == m:
                c = Candle(ts=r.timestamp, open=float(r.open), high=float(r.high), low=float(r.low), close=float(r.close), volume=float(r.volume or 0))
                self.session.candles.append(c); self._step(c)

    # ---------------- one candle
    def _step(self, c: Candle):
        m = len(self.session.candles) - 1
        if self.pending is not None:                       # fill last minute's intent at this candle's open
            self._enter(self.pending, c, m); self.pending = None
        dec = self.engine.on_candle(c)
        for f in dec.fills:
            self._exit(f, c)
        if dec.intents:
            self.pending = dec.intents[0]
            it = dec.intents[0]
            self.tg.send(f"<b>{self.cfg.symbol} setup</b> {'CALL' if it.plan.direction > 0 else 'PUT'} score {it.score} regime {it.regime}\n"
                         f"entry ~{it.plan.entry:.1f} stop {it.plan.stop:.1f} ({it.plan.risk_pts:.1f} pts) tp1 {it.plan.tp1:.1f} tp2 {it.plan.tp2:.1f} [{it.plan.target_name}] R:R {it.plan.rr:.1f}")

    def _enter(self, it: OrderIntent, c: Candle, m: int):
        spot = self.kite.spot() or c.open
        con = self.kite.pick_option(it.plan.direction, spot, self.cfg.cost.delta)
        if con is None:
            log.error("no option contract found; skipping entry"); self.engine.setups.resolve(it.setup, "REJECTED", "no contract"); return
        qty = it.plan.lots * con["lot_size"]
        if self.paper:
            oid, status, avg = f"PAPER-{int(time.time())}", "COMPLETE", con["premium"]
        else:
            oid, status, avg = self.kite.market_order(con["tradingsymbol"], con["exchange"], "BUY", qty)
            if status != "COMPLETE":
                log.critical(f"entry order {oid} status {status}; not opening position"); self.tg.send(f"⚠️ entry order {status}: {con['tradingsymbol']}")
                self.engine.setups.resolve(it.setup, "REJECTED", f"order {status}"); return
        self.engine.open_position(it, spot, m)
        self.contract = con; self.entry_premium = avg or con["premium"]; self.entry_ts = c.ts
        self.events.log(c.ts, "ORDER", side="BUY", symbol=con["tradingsymbol"], qty=qty, premium=self.entry_premium, order_id=oid, paper=self.paper, spot=spot)
        self.tg.send(f"{'📝 PAPER' if self.paper else '🟢 LIVE'} BUY {qty} {con['tradingsymbol']} @ {self.entry_premium:.2f} (spot {spot:.1f}, delta {con['delta']})\nstop {it.plan.stop:.1f} tp1 {it.plan.tp1:.1f} tp2 {it.plan.tp2:.1f}")
        self._save_state()

    def _exit(self, f, c: Candle):
        if self.contract is None:
            return
        con = self.contract; key = f"{con['exchange']}:{con['tradingsymbol']}"
        prem = self.kite.ltp([key]).get(key, 0.0)
        if self.paper:
            oid, status, avg = f"PAPER-{int(time.time())}", "COMPLETE", prem
        else:
            oid, status, avg = self.kite.market_order(con["tradingsymbol"], con["exchange"], "SELL", f.quantity)
            if status != "COMPLETE":
                log.critical(f"EXIT ORDER {oid} status {status}: MANUAL INTERVENTION REQUIRED"); self.tg.send(f"🚨 exit order {status} for {con['tradingsymbol']} qty {f.quantity}: close manually")
        px = avg or prem
        pnl = (px - self.entry_premium) * f.quantity
        cost = statutory("option", self.entry_premium * f.quantity, px * f.quantity, self.cfg.cost.brokerage_per_order, self.cfg.cost.stt_sell_pct)
        net = pnl - cost; self.pnl_day += net
        self.events.log(c.ts, "ORDER", side="SELL", symbol=con["tradingsymbol"], qty=f.quantity, premium=avg or prem, order_id=oid, paper=self.paper, reason=f.reason, net=net)
        row = {"date": str(self.day), "symbol": con["tradingsymbol"], "qty": f.quantity, "entry_premium": self.entry_premium, "exit_premium": avg or prem,
               "entry_ts": str(self.entry_ts), "exit_ts": str(c.ts), "reason": f.reason, "spot_exit": f.price, "net": round(net, 2), "paper": self.paper}
        pd.DataFrame([row]).to_csv(self.trades_path, mode="a", header=not self.trades_path.exists(), index=False)
        self.tg.send(f"{'📝 PAPER' if self.paper else '🔴 LIVE'} SELL {f.quantity} {con['tradingsymbol']} @ {(avg or prem):.2f} [{f.reason}] net ₹{net:,.0f} | day ₹{self.pnl_day:,.0f}")
        if f.state == "EXIT":
            self.engine.record_result(net); self.contract = None
        self._save_state()

    def _save_state(self):
        st = {"day": str(self.day), "symbol": self.cfg.symbol, "paper": self.paper, "contract": self.contract, "entry_premium": self.entry_premium,
              "pnl_day": self.pnl_day, "trades_today": self.engine.trades_today, "halted": self.engine.halted_reason,
              "position": None if not self.engine.position else {"direction": self.engine.position.direction, "entry": self.engine.position.entry_price,
                                                                     "stop": self.engine.position.stop, "remaining": self.engine.position.remaining, "state": self.engine.position.state}}
        (RUNTIME / f"state_{self.cfg.symbol}.json").write_text(json.dumps(st, indent=1, default=str))

    # ---------------- loop
    def run(self):
        mode = "PAPER" if self.paper else "LIVE"
        log.info(f"{mode} runner for {self.cfg.symbol} on {self.day}; lot {self.cfg.risk.lot_size}; entries {self.cfg.session.entry_start}-{self.cfg.session.entry_end}, flatten {self.cfg.session.flatten}")
        self.tg.send(f"pa_engine v3 {mode} started for {self.cfg.symbol} ({self.day}) as {self.kite.user}")
        end = datetime.now(IST).replace(hour=15, minute=31, second=0, microsecond=0)
        while not self.stop and datetime.now(IST) < end:
            if KILL_FILE.exists():
                log.warning("kill file present; stopping"); break
            try:
                self._catch_up()
            except Exception as e:
                log.exception(f"loop error: {e}")
                time.sleep(5)
            time.sleep(self.poll)
        if self.engine.position and self.engine.position.open and self.contract is not None:
            log.warning("stopping with an open position: flattening at market")
            from .position import Fill
            q = self.engine.position.remaining; self.engine.position.remaining = 0; self.engine.position.state = "EXIT"
            last = self.session.candles[-1] if self.session.candles else None
            self._exit(Fill(len(self.session.candles) - 1, last.close if last else 0.0, q, "RUNNER_STOP", "EXIT"), last or Candle(pd.Timestamp.now(tz=IST), 0, 0, 0, 0))
        self.events.close(); self._save_state()
        self.tg.send(f"pa_engine v3 {mode} stopped for {self.cfg.symbol}: {self.engine.trades_today} trades, day ₹{self.pnl_day:,.0f}")


def main(cfg: EngineConfig, paper: bool = True, poll: float = 2.0):
    load_env()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    env_paper = env_bool("PAPER_TRADING", True)
    if not paper and env_paper:
        sys.exit("refusing LIVE mode: PAPER_TRADING=true in .env (set PAPER_TRADING=false explicitly to trade real money)")
    if not paper:
        print("*** LIVE MODE: real orders will be placed. Type LIVE to continue: ", end="", flush=True)
        if input().strip() != "LIVE":
            sys.exit("aborted")
    LiveRunner(cfg, paper=paper or env_paper, poll=poll).run()
