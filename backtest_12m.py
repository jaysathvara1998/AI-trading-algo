#!/usr/bin/env python
"""
backtest_12m.py - Replay the algo_vpin_v2 Track-1 decision pipeline over the bundled
12-month 1-minute index data (algo_vpin_v2/data/{nifty,sensex}_12m_1min.csv.gz).

Drives the REAL engines (GARCHEngine, VPINCalculator, MacroFeatureEngine, MultiTimeframeAnalyst,
SMCEngine, PatternRecognitionEngine, EnsembleBrain + skills, RiskManager) in the same order as
AlgoVPINRunner.process_incoming_bar, one session at a time with fresh per-day state like production.

What the code cannot supply offline and is therefore MODELLED here (see --help):
  * option premium  : P = P0 + delta * (spot - spot_entry) * dir - theta_per_min * minutes_held
                      P0 = spot * premium_pct (ITM weekly, delta ~0.72). No IV / gamma changes.
  * intra-bar exits : checked at the adverse extreme, then the favourable extreme, then the close
                      (conservative ordering); threshold exits fill at the threshold price.
  * costs           : Dhan flat brokerage per order + STT (sell side) + exchange charges + GST + stamp.

Usage:
  .venv/bin/python backtest_12m.py --symbol NIFTY --mode scalper --ml on  --out /tmp/x
  .venv/bin/python backtest_12m.py --symbol NIFTY --mode scalper --ml off --days 3
"""
import os
import sys
import json
import time
import argparse
import logging
from collections import deque
from pathlib import Path

# Neutralise side effects BEFORE the package (and its .env loader) is imported.
os.environ["ENABLE_TELEGRAM"] = "false"
os.environ["PAPER_TRADING"] = "true"

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.CRITICAL)
for name in ("algo_vpin_v2", "arch", "vision_chart_brain"):
    logging.getLogger(name).setLevel(logging.CRITICAL)

from algo_vpin_v2.config import CONFIG, TradeStrategyMode
from algo_vpin_v2.garch_engine import GARCHEngine, DirectionalSignal
from algo_vpin_v2.vpin import VPINCalculator
from algo_vpin_v2.mtf_analyst import MultiTimeframeAnalyst
from algo_vpin_v2.macro_features import MacroFeatureEngine
from algo_vpin_v2.pattern_engine import PatternRecognitionEngine
from algo_vpin_v2.smc_engine import SMCEngine
from algo_vpin_v2 import risk_manager as rm_mod
from algo_vpin_v2.risk_manager import RiskManager, PositionSide
from algo_vpin_v2.ensemble_brain import EnsembleBrain
from algo_vpin_v2.skills.pre_trade_sanity_skill import PreTradeSanitySkill

# --- hard safety patches: never touch Telegram, disk models, or today's trade history ---
rm_mod.notify_trailing_sl = lambda *a, **k: None
RiskManager._sync_from_trades_history = lambda self: None


def build_brain(ml_on: bool) -> EnsembleBrain:
    CONFIG.ensemble.xgb_min_prob_threshold = 0.80  # production value from adaptive_config.json
    brain = EnsembleBrain(CONFIG.ensemble)
    # never overwrite the production .joblib files
    brain.svm.save_model = lambda *a, **k: None
    brain.xgb.save_model = lambda *a, **k: None
    brain.ann.save_model = lambda *a, **k: None
    # ANN only affects Track 2; skip its (slow) online training
    brain.ann.update_bar = lambda *a, **k: None
    if not ml_on:
        brain.svm.predict = lambda v: (0, 0.0)
        brain.xgb.predict = lambda v: (0, 0.5)
        brain.ann.predict = lambda v: (0, 0.5, None)
        brain.update_bar = lambda *a, **k: None
    return brain


def load_sessions(symbol: str):
    df = pd.read_csv(ROOT / "algo_vpin_v2" / "data" / f"{symbol.lower()}_12m_1min.csv.gz")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    t = df["timestamp"].dt.time
    df = df[(t >= pd.Timestamp("09:15").time()) & (t < pd.Timestamp("15:30").time())].copy()
    df["date"] = df["timestamp"].dt.date
    sizes = df.groupby("date").size()
    keep = sizes[sizes >= 300].index
    df = df[df["date"].isin(keep)].reset_index(drop=True)
    return [g.reset_index(drop=True) for _, g in df.groupby("date", sort=True)]


def round_trip_costs(buy_value: float, sell_value: float, brokerage_per_order: float = 20.0) -> float:
    stt = 0.001 * sell_value                       # 0.1% STT on option sell premium
    txn = 0.00035 * (buy_value + sell_value)       # exchange transaction charges (approx.)
    sebi = 0.000001 * (buy_value + sell_value)
    stamp = 0.00003 * buy_value
    gst = 0.18 * (2 * brokerage_per_order + txn + sebi)
    return 2 * brokerage_per_order + stt + txn + sebi + stamp + gst


class Replay:
    def __init__(self, symbol, mode, ml_on, premium_pct, delta, theta_per_min_pct, warm_days, verbose):
        self.symbol = symbol.upper()
        CONFIG.market.set_symbol(self.symbol)
        self.mode = TradeStrategyMode.SCALPER if mode == "scalper" else TradeStrategyMode.INSTITUTIONAL_SWING
        self.ml_on = ml_on
        self.premium_pct = premium_pct
        self.delta = delta
        self.theta_pct = theta_per_min_pct
        self.warm_days = warm_days
        self.verbose = verbose
        self.slip = CONFIG.risk.slippage_pct
        self.lot = CONFIG.market.lot_size

        # engines that live across sessions (their windows are < 3 sessions anyway)
        self.garch = GARCHEngine(CONFIG.garch)
        self.vpin = VPINCalculator(CONFIG.vpin)
        self.smc = SMCEngine()
        self.macro = MacroFeatureEngine()
        self.brain = build_brain(ml_on)
        self.sanity = PreTradeSanitySkill()
        self.trades = []
        self.daily = []

    # ---------------- option premium model ----------------
    def premium_at(self, spot: float, pos_ctx: dict, ts) -> float:
        held_min = (ts - pos_ctx["entry_ts"]).total_seconds() / 60.0
        p = pos_ctx["p0"] + self.delta * pos_ctx["dir"] * (spot - pos_ctx["entry_spot"]) - pos_ctx["theta"] * held_min
        return max(0.5, round(p, 2))

    # ---------------- exit handling ----------------
    def try_exit(self, rm: RiskManager, pos_ctx, spot, ts, reversal=None, counter=None, tag=""):
        pos = rm.current_position
        if pos.side == PositionSide.FLAT:
            return False
        prem = self.premium_at(spot, pos_ctx, ts)
        exit_needed, reason = rm.check_exit_conditions(
            current_price=spot, option_premium=prem, current_time=ts.to_pydatetime(),
            reversal_signal=reversal, counter_trend_reason=counter
        )
        if not exit_needed:
            return False
        if "DISASTER" in reason and pos.hard_disaster_sl > 0:
            fill_ref = min(prem, pos.hard_disaster_sl) if prem < pos.hard_disaster_sl else pos.hard_disaster_sl
        elif "STOP" in reason or "TRAILING" in reason:
            fill_ref = pos.stop_loss if prem <= pos.stop_loss else prem
        elif "TAKE_PROFIT" in reason:
            fill_ref = pos.take_profit
        else:
            fill_ref = prem
            # At the adverse extreme a non-threshold exit (counter-trend / reversal) can report a price far
            # below the stop; under live 8-10 s polling the stop would have fired first at ~its level.
            if tag == "adverse" and pos.stop_loss > 0 and prem < pos.stop_loss:
                fill_ref = pos.stop_loss
        fill = round(max(0.05, fill_ref * (1.0 - self.slip)), 2)
        entry_fill = pos.entry_price
        qty = pos.quantity
        realized = rm.record_trade_close(fill)
        costs = round_trip_costs(entry_fill * qty, fill * qty)
        self.trades.append({
            "date": str(ts.date()), "entry_ts": str(pos_ctx["entry_ts"]), "exit_ts": str(ts),
            "side": "CALL" if pos_ctx["dir"] > 0 else "PUT", "mode": pos_ctx["mode"],
            "entry_spot": pos_ctx["entry_spot"], "exit_spot": spot,
            "entry_prem": entry_fill, "exit_prem": fill, "qty": qty,
            "gross_pnl": round(realized, 2), "costs": round(costs, 2), "net_pnl": round(realized - costs, 2),
            "hold_min": round((ts - pos_ctx["entry_ts"]).total_seconds() / 60.0, 1),
            "reason": reason.split(" (")[0].split(":")[0], "reason_full": reason, "stage": tag,
        })
        pos_ctx.clear()
        return True

    # ---------------- one session ----------------
    def run_day(self, day_df: pd.DataFrame, hist_df: pd.DataFrame, trade: bool):
        ts0 = day_df["timestamp"].iloc[0]
        # per-day fresh state, as in production (new process each morning)
        self.macro.initialize_from_history(hist_df)
        if len(hist_df) >= 375:
            self.vpin.update_bucket_volume(float(hist_df["volume"].sum()) / (len(hist_df) / 375.0))
        mtf = MultiTimeframeAnalyst()
        pattern = PatternRecognitionEngine()
        rm = RiskManager(CONFIG.risk, CONFIG.market, CONFIG.scalper, track_name="Track 1: Dual-Brain")
        recent = deque(maxlen=30)
        for _, r in hist_df.tail(30).iterrows():
            recent.append({"open": r.open, "high": r.high, "low": r.low, "close": r.close, "volume": r.volume})
        pos_ctx = {}
        day_pnl = 0.0
        n_trades_before = len(self.trades)
        m_open = ts0.replace(hour=9, minute=15, second=0)
        m_cut = ts0.replace(hour=CONFIG.market.last_entry_hour, minute=CONFIG.market.last_entry_minute, second=0)

        for i in range(len(day_df)):
            row = day_df.iloc[i]
            ts, o, h, l, c, v = row.timestamp, float(row.open), float(row.high), float(row.low), float(row.close), float(row.volume)

            # STEP 1: MTF update + exits on open position (adverse extreme -> favourable extreme -> close)
            mtf.update_bar(c, timestamp=ts)
            if rm.current_position.side != PositionSide.FLAT:
                is_call = pos_ctx["dir"] > 0
                is_counter, counter_msg = mtf.check_cumulative_counter_trend(holding_call=is_call, holding_put=not is_call, threshold_pts=20.0)
                adverse, favourable = (l, h) if is_call else (h, l)
                if not self.try_exit(rm, pos_ctx, adverse, ts, counter=counter_msg if is_counter else None, tag="adverse"):
                    if not self.try_exit(rm, pos_ctx, favourable, ts, tag="favourable"):
                        self.try_exit(rm, pos_ctx, c, ts, tag="close")

            # STEP 2: engines
            macro_st = self.macro.update_1min_bar(c, h, l)
            vpin_res = self.vpin.process_bar(close_price=c, volume=v, timestamp=ts)
            garch_res = self.garch.add_bar(close_price=c, timestamp=ts)
            smc_st = self.smc.update_bar(o, h, l, c, v)
            pattern_st = pattern.update_bar(o, h, l, c, v)
            recent.append({"open": o, "high": h, "low": l, "close": c, "volume": v})
            if garch_res is None:
                continue

            price_delta = c - (self.garch.prices[-2] if len(self.garch.prices) >= 2 else c)
            prev_open = recent[-2]["open"] if len(recent) >= 2 else o
            prev_close = recent[-2]["close"] if len(recent) >= 2 else c
            key_lvl = [macro_st.pdl, macro_st.pdh, round((macro_st.pdh + macro_st.pdl) / 2.0, 2)]

            # STEP 3: ensemble brain (+ skills) and online learning
            dec = self.brain.evaluate(
                garch_signal=garch_res.signal, price_delta=price_delta, rolling_vol=garch_res.sigma_next,
                garch_forecast=garch_res.mu_next, vpin=vpin_res.vpin, macro_state=macro_st, recent_bars=list(recent),
                smc_state=smc_st, open_p=o, high_p=h, low_p=l, close_p=c, prev_open=prev_open, prev_close=prev_close,
                recent_highs=[b["high"] for b in recent], recent_lows=[b["low"] for b in recent], key_levels=key_lvl,
                underlying=self.symbol, pattern_st=pattern_st,
            )
            self.brain.update_bar(current_price=c, price_delta=price_delta, rolling_vol=garch_res.sigma_next,
                                  garch_forecast=garch_res.mu_next, vpin=vpin_res.vpin, macro_state=macro_st)

            # reversal exit re-check (as main.py does after the brain)
            if rm.current_position.side != PositionSide.FLAT:
                rev = dec.final_action if not dec.is_vetoed else garch_res.signal
                self.try_exit(rm, pos_ctx, c, ts, reversal=rev, tag="reversal")

            if not trade:
                continue

            # STEP 4: entry
            active = (m_open <= ts <= m_cut)
            if not (active and not dec.is_vetoed and dec.final_action in (DirectionalSignal.BUY, DirectionalSignal.SELL)):
                continue
            if rm.current_position.side != PositionSide.FLAT:
                continue
            action = "BUY" if dec.final_action == DirectionalSignal.BUY else "SELL"
            ok, _ = mtf.validate_htf_entry_alignment(action, c)
            if not ok:
                continue
            p0 = round(c * self.premium_pct, 2)
            targets = rm.compute_trade_targets(
                signal=dec.final_action, current_price=c, garch_res=garch_res, vpin_res=vpin_res, is_option=True,
                option_premium=p0, available_cash=CONFIG.risk.capital_allocation, strategy_mode=self.mode,
                candle_high=h, candle_low=l,
            )
            if not targets:
                continue
            san = self.sanity.evaluate(intended_direction=action, entry_price=p0, stop_loss=targets.stop_loss,
                                       take_profit=targets.take_profit, current_spot=c, recent_bars=list(recent))
            if not san.is_favorable:
                continue
            unanimous = True if not self.ml_on else (dec.svm_prediction == dec.xgb_prediction and dec.xgb_prediction != 0)
            approved, _ = rm.validate_new_entry(signal=dec.final_action, vpin_res=vpin_res, current_time=ts.to_pydatetime(),
                                                is_unanimous=unanimous, volume_ratio=1.0)
            if not approved:
                continue
            # replicate OrderExecutionRouter.execute_entry bookkeeping
            fill = round(p0 * (1.0 + self.slip), 2)
            direction = 1 if action == "BUY" else -1
            pos = rm.current_position
            pos.side = PositionSide.LONG if direction > 0 else PositionSide.SHORT
            pos.entry_price = fill
            pos.quantity = targets.total_quantity
            pos.stop_loss = targets.stop_loss
            pos.take_profit = targets.take_profit
            pos.entry_time = ts.to_pydatetime()
            pos.entry_vpin = vpin_res.vpin
            pos.entry_regime = vpin_res.regime
            pos.symbol = f"{self.symbol} {int(round(c / CONFIG.market.step_size) * CONFIG.market.step_size)} {'CALL' if direction > 0 else 'PUT'}"
            pos.is_option = True
            pos.highest_price = fill
            pos.lowest_price = fill
            pos.strategy_mode = targets.strategy_mode
            pos.initial_risk_pts = targets.initial_risk_pts
            pos.current_trail_tier_r = 0.0
            pos.entry_spot_price = targets.entry_spot_price
            pos.spot_stop_loss = targets.spot_stop_loss
            pos.hard_disaster_sl = targets.hard_disaster_sl
            pos.is_runner_active = False
            pos_ctx.update({"entry_ts": ts, "entry_spot": c, "p0": fill, "dir": direction,
                            "theta": p0 * self.theta_pct, "mode": targets.strategy_mode.value})

        # force-close anything still open on the last bar (square-off should already have fired at 15:24)
        if rm.current_position.side != PositionSide.FLAT:
            last = day_df.iloc[-1]
            self.try_exit(rm, pos_ctx, float(last.close), last.timestamp, tag="eod-force") or None
            if rm.current_position.side != PositionSide.FLAT:
                prem = self.premium_at(float(last.close), pos_ctx, last.timestamp)
                pos = rm.current_position
                realized = rm.record_trade_close(prem)
                self.trades.append({"date": str(last.timestamp.date()), "entry_ts": str(pos_ctx["entry_ts"]), "exit_ts": str(last.timestamp),
                                    "side": "CALL" if pos_ctx["dir"] > 0 else "PUT", "mode": pos_ctx["mode"], "entry_spot": pos_ctx["entry_spot"],
                                    "exit_spot": float(last.close), "entry_prem": pos_ctx["p0"], "exit_prem": prem, "qty": pos.quantity or self.lot,
                                    "gross_pnl": round(realized, 2), "costs": 0.0, "net_pnl": round(realized, 2), "hold_min": 0.0,
                                    "reason": "EOD_FORCE_CLOSE", "reason_full": "EOD_FORCE_CLOSE", "stage": "eod"})
                pos_ctx.clear()

        day_trades = self.trades[n_trades_before:]
        day_pnl = sum(t["net_pnl"] for t in day_trades)
        self.daily.append({"date": str(ts0.date()), "trades": len(day_trades), "net_pnl": round(day_pnl, 2), "traded": trade})
        return day_pnl, len(day_trades)

    def run(self, sessions, max_days=None):
        t_start = time.time()
        n = len(sessions) if max_days is None else min(len(sessions), self.warm_days + max_days)
        for d in range(n):
            hist = pd.concat(sessions[max(0, d - 5):d]) if d > 0 else sessions[0].head(0)
            if d == 0:
                hist = sessions[0].head(0)
            trade = d >= self.warm_days
            if d == self.warm_days and self.ml_on:
                self.brain.train_models()
            pnl, k = self.run_day(sessions[d], hist if len(hist) else sessions[d].head(1), trade)
            if self.verbose:
                print(f"[{d+1:3d}/{n}] {sessions[d]['timestamp'].iloc[0].date()} trades={k:2d} net={pnl:9.2f} "
                      f"cum={sum(t['net_pnl'] for t in self.trades):10.2f} ({time.time()-t_start:5.0f}s)", file=sys.stderr, flush=True)

    # ---------------- reporting ----------------
    def summary(self) -> dict:
        tr = pd.DataFrame(self.trades)
        dy = pd.DataFrame([d for d in self.daily if d["traded"]])
        if tr.empty:
            return {"trades": 0}
        wins = tr[tr.net_pnl > 0]; losses = tr[tr.net_pnl <= 0]
        cum = tr.net_pnl.cumsum()
        dd = (cum - cum.cummax()).min()
        gp, gl = wins.net_pnl.sum(), -losses.net_pnl.sum()
        tr["month"] = pd.to_datetime(tr.date).dt.to_period("M").astype(str)
        monthly = tr.groupby("month").agg(trades=("net_pnl", "size"), net=("net_pnl", "sum")).round(0)
        reasons = tr.groupby("reason").agg(n=("net_pnl", "size"), net=("net_pnl", "sum"), avg=("net_pnl", "mean")).round(0).sort_values("n", ascending=False)
        return {
            "symbol": self.symbol, "mode": self.mode.value, "ml": self.ml_on,
            "sessions_traded": int(len(dy)), "trades": int(len(tr)), "trades_per_day": round(len(tr) / max(1, len(dy)), 2),
            "net_pnl": round(tr.net_pnl.sum(), 0), "gross_pnl": round(tr.gross_pnl.sum(), 0), "costs": round(tr.costs.sum(), 0),
            "win_rate": round(len(wins) / len(tr), 3), "avg_win": round(wins.net_pnl.mean() if len(wins) else 0, 0),
            "avg_loss": round(losses.net_pnl.mean() if len(losses) else 0, 0), "profit_factor": round(gp / gl, 2) if gl > 0 else None,
            "max_drawdown": round(dd, 0), "avg_hold_min": round(tr.hold_min.mean(), 1),
            "positive_days": int((dy.net_pnl > 0).sum()), "negative_days": int((dy.net_pnl < 0).sum()), "flat_days": int((dy.trades == 0).sum()),
            "best_day": round(dy.net_pnl.max(), 0), "worst_day": round(dy.net_pnl.min(), 0),
            "calls": int((tr.side == "CALL").sum()), "puts": int((tr.side == "PUT").sum()),
            "monthly": monthly.to_dict("index"), "exit_reasons": reasons.to_dict("index"),
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="NIFTY", choices=["NIFTY", "SENSEX"])
    ap.add_argument("--mode", default="scalper", choices=["scalper", "swing"])
    ap.add_argument("--ml", default="on", choices=["on", "off"], help="on = SVM/XGB online-trained vetoes as in production; off = rules only")
    ap.add_argument("--premium-pct", type=float, default=0.007, help="entry premium as fraction of spot (ITM weekly ~0.72 delta)")
    ap.add_argument("--delta", type=float, default=0.72)
    ap.add_argument("--theta-pct", type=float, default=0.00025, help="premium decay per minute as fraction of entry premium (~9%% per session)")
    ap.add_argument("--warm-days", type=int, default=5)
    ap.add_argument("--days", type=int, default=None, help="limit traded sessions (smoke test)")
    ap.add_argument("--out", default=None, help="output prefix for trades csv + summary json")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    sessions = load_sessions(a.symbol)
    rp = Replay(a.symbol, a.mode, a.ml == "on", a.premium_pct, a.delta, a.theta_pct, a.warm_days, not a.quiet)
    rp.run(sessions, max_days=a.days)
    s = rp.summary()
    if a.out:
        pd.DataFrame(rp.trades).to_csv(f"{a.out}_trades.csv", index=False)
        pd.DataFrame(rp.daily).to_csv(f"{a.out}_daily.csv", index=False)
        with open(f"{a.out}_summary.json", "w") as f:
            json.dump(s, f, indent=2, default=str)
    print(json.dumps(s, indent=2, default=str))


if __name__ == "__main__":
    main()
