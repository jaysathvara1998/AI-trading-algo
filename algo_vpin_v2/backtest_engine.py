"""
Backtest Engine for Algo VPIN v2.0
Runs comprehensive historical comparative backtests across:
1. Track A: Pure Scalper Mode (Sahil Masterclass 1:1.5 RR + 0.5R Step Trailing)
2. Track B: Pure Institutional Swing Mode (Open-ended 20-40 pt Trend Runner)
3. Track C: AI Adaptive Brain Mode (Brain dynamically selects Scalp vs Swing per setup)
"""

import sys
from pathlib import Path

# Add project root directory to path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging

from algo_vpin_v2.config import CONFIG, AppConfig, ExecutionMode, TradeStrategyMode
from algo_vpin_v2.garch_engine import GARCHEngine, DirectionalSignal
from algo_vpin_v2.vpin import VPINCalculator, ToxicityRegime
from algo_vpin_v2.macro_features import MacroFeatureEngine
from algo_vpin_v2.ensemble_brain import EnsembleBrain
from algo_vpin_v2.risk_manager import RiskManager, PositionSide
from algo_vpin_v2.humanoid_agent import HumanoidTraderAgent

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("algo_vpin_v2.backtest")


class StrategyTrackSimulator:
    """Simulates a single execution strategy track over historical price feeds"""

    def __init__(self, name: str, mode: ExecutionMode):
        self.name = name
        self.mode = mode
        self.cfg = AppConfig()
        self.cfg.risk.execution_mode = mode
        self.risk_mgr = RiskManager(risk_config=self.cfg.risk, market_config=self.cfg.market, scalper_config=self.cfg.scalper)
        
        self.spot_entry_price: float = 0.0
        self.is_call: bool = True
        self.trades = []
        self.equity_curve = [50000.0]
        self.lot_size = 65

    def check_trade_lifecycle(self, current_spot: float, bar_high: float, bar_low: float, timestamp: datetime):
        pos = self.risk_mgr.current_position
        if pos.side == PositionSide.FLAT:
            return

        # Realistic Option Delta ~0.70 tracking (Call gains on spot rise, Put gains on spot fall)
        if self.is_call:
            spot_delta = current_spot - self.spot_entry_price
        else:
            spot_delta = self.spot_entry_price - current_spot
        opt_gain_est = spot_delta * 0.70

        # Update peak option price
        current_opt_ltp = max(5.0, pos.entry_price + opt_gain_est)
        pos.highest_price = max(pos.highest_price, current_opt_ltp)
        pos.lowest_price = min(pos.lowest_price if pos.lowest_price > 0 else current_opt_ltp, current_opt_ltp)

        exit_needed, exit_reason = self.risk_mgr.check_exit_conditions(
            current_price=current_spot,
            option_premium=current_opt_ltp,
            current_time=timestamp
        )

        if exit_needed:
            # Slippage deduction ~0.10 pt
            fill_exit = max(1.0, current_opt_ltp - 0.10)
            entry_pr = pos.entry_price
            strat_mode = pos.strategy_mode.value if pos.strategy_mode else "UNKNOWN"
            realized_pnl = self.risk_mgr.record_trade_close(fill_exit)
            points_won = (fill_exit - entry_pr)

            trade_record = {
                "exit_time": timestamp,
                "strategy_mode": strat_mode,
                "entry_price": entry_pr,
                "exit_price": fill_exit,
                "points": points_won,
                "pnl_inr": realized_pnl,
                "reason": exit_reason
            }
            self.trades.append(trade_record)
            new_equity = self.equity_curve[-1] + realized_pnl
            self.equity_curve.append(new_equity)
            self.spot_entry_price = 0.0



class AlgoBacktestEngine:
    def __init__(self):
        self.tracks = {
            "Track A (Pure Scalper)": StrategyTrackSimulator("Pure Scalper (1:1.5 RR + 0.5R Trail)", ExecutionMode.SCALPER_ONLY),
            "Track B (Pure Swing)": StrategyTrackSimulator("Pure Institutional Swing", ExecutionMode.SWING_ONLY),
            "Track C (AI Adaptive Brain)": StrategyTrackSimulator("AI Adaptive Dynamic Selection", ExecutionMode.AUTO_BRAIN_SELECT)
        }
        self.garch = GARCHEngine()
        self.vpin_calc = VPINCalculator()
        self.macro = MacroFeatureEngine()
        self.brain = EnsembleBrain()
        self.humanoid = HumanoidTraderAgent()

    def generate_benchmark_dataset(self, n_days: int = 5, bars_per_day: int = 375) -> pd.DataFrame:
        """Generates multi-regime historical market dataset (Trending, Choppy, Reversal Days)"""
        records = []
        base_price = 23300.0
        start_date = datetime(2026, 9, 14, 9, 15)

        for day in range(n_days):
            day_start = start_date + timedelta(days=day)
            curr = base_price
            
            # Day regime definition
            if day % 3 == 0:
                drift = 0.35   # Trending Bull Day
                vol = 2.5
            elif day % 3 == 1:
                drift = -0.30  # Trending Bear Day
                vol = 2.8
            else:
                drift = 0.02   # Compressed Chop Day
                vol = 1.2

            for m in range(bars_per_day):
                bar_time = day_start + timedelta(minutes=m)
                shock = np.random.normal(drift, vol)
                curr += shock
                high = curr + abs(np.random.normal(1.5, 0.8))
                low = curr - abs(np.random.normal(1.5, 0.8))
                close = curr
                volume = max(5000.0, np.random.normal(25000.0, 8000.0))

                records.append({
                    "timestamp": bar_time,
                    "open": curr - shock,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": volume
                })
            base_price = curr

        return pd.DataFrame(records)

    def run_backtest(self, df: pd.DataFrame):
        logger.info(f"[BACKTEST] Starting historical simulation over {len(df)} 1-minute bars...")
        last_date = None

        for idx, row in df.iterrows():
            curr_spot = row["close"]
            bar_high = row["high"]
            bar_low = row["low"]
            vol = row["volume"]
            ts = row["timestamp"]

            # Daily Reset
            curr_date = ts.date()
            if curr_date != last_date:
                last_date = curr_date
                for track in self.tracks.values():
                    track.risk_mgr.total_trades_today = 0
                    track.risk_mgr.daily_realized_pnl = 0.0

            # Update core indicators
            macro_st = self.macro.update_1min_bar(curr_spot, bar_high, bar_low)
            vpin_res = self.vpin_calc.process_bar(curr_spot, vol, ts)
            garch_res = self.garch.add_bar(curr_spot, ts)


            if garch_res is None:
                continue

            price_delta = curr_spot - (self.garch.prices[-2] if len(self.garch.prices) >= 2 else curr_spot)
            
            # Brain Evaluation
            ensemble_dec = self.brain.evaluate(
                garch_signal=garch_res.signal,
                price_delta=price_delta,
                rolling_vol=garch_res.sigma_next,
                garch_forecast=garch_res.mu_next,
                vpin=vpin_res.vpin,
                macro_state=macro_st
            )

            # Check exits across all 3 tracks
            for track in self.tracks.values():
                track.check_trade_lifecycle(curr_spot, bar_high, bar_low, ts)

            # Check entry if signal fired
            if not ensemble_dec.is_vetoed and ensemble_dec.final_action in (DirectionalSignal.BUY, DirectionalSignal.SELL):
                sim_premium = 120.0  # Simulated ATM / ITM Option Premium

                for track_name, track in self.tracks.items():
                    if track.risk_mgr.current_position.side == PositionSide.FLAT:
                        # Determine Strategy Mode
                        chosen_mode, _ = self.humanoid.evaluate_execution_mode(
                            ensemble_dec=ensemble_dec,
                            heavyweight_st=None,
                            vpin_res=vpin_res,
                            configured_mode=track.mode
                        )

                        targets = track.risk_mgr.compute_trade_targets(
                            signal=ensemble_dec.final_action,
                            current_price=curr_spot,
                            garch_res=garch_res,
                            vpin_res=vpin_res,
                            is_option=True,
                            option_premium=sim_premium,
                            strategy_mode=chosen_mode
                        )

                        if targets:
                            pos = track.risk_mgr.current_position
                            pos.side = PositionSide.LONG
                            pos.entry_price = sim_premium
                            pos.quantity = targets.total_quantity
                            pos.stop_loss = targets.stop_loss
                            pos.take_profit = targets.take_profit
                            pos.entry_time = ts
                            pos.is_option = True
                            pos.highest_price = sim_premium
                            pos.lowest_price = sim_premium
                            pos.strategy_mode = targets.strategy_mode
                            pos.initial_risk_pts = targets.initial_risk_pts
                            pos.current_trail_tier_r = 0.0
                            track.spot_entry_price = curr_spot
                            track.is_call = (ensemble_dec.final_action == DirectionalSignal.BUY)
                            pos.symbol = "NIFTY_ATM_CE" if track.is_call else "NIFTY_ATM_PE"


        self.print_comparative_scorecard()

    def print_comparative_scorecard(self):
        print("\n" + "=" * 94)
        print("                   AURA-v2 QUANTITATIVE MULTI-MODE BACKTEST SCORECARD                   ")
        print("=" * 94)
        print(f"{'Strategy Track':<28} | {'Trades':<6} | {'Win Rate':<8} | {'Total Pts':<10} | {'Net P&L (INR)':<15} | {'Profit Factor':<12}")
        print("-" * 94)

        for track_name, track in self.tracks.items():
            trades = track.trades
            n_trades = len(trades)
            if n_trades == 0:
                print(f"{track_name:<28} | {0:<6} | {'0.0%':<8} | {'0.0 pts':<10} | {'+0.00 INR':<15} | {'N/A':<12}")
                continue

            wins = [t for t in trades if t["pnl_inr"] > 0]
            losses = [t for t in trades if t["pnl_inr"] <= 0]
            win_rate = (len(wins) / n_trades) * 100.0

            total_points = sum(t["points"] for t in trades)
            total_pnl = sum(t["pnl_inr"] for t in trades)

            gross_profit = sum(t["pnl_inr"] for t in wins) if wins else 0.0
            gross_loss = abs(sum(t["pnl_inr"] for t in losses)) if losses else 0.0
            profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (gross_profit if gross_profit > 0 else 1.0)

            print(f"{track_name:<28} | {n_trades:<6} | {win_rate:>6.1f}%  | {total_points:>+8.1f} pt | {total_pnl:>+12.2f} INR | {profit_factor:>8.2f}x")

        print("=" * 94)

        print("\n" + "-" * 94)
        print("                        DETAILED INDIVIDUAL TRADE LOGS                        ")
        print("-" * 94)
        for track_name, track in self.tracks.items():
            print(f"\n>> [{track_name.upper()}]:")
            if not track.trades:
                print("   No trades executed in this track.")
            for i, t in enumerate(track.trades, 1):
                pnl_tag = "WIN" if t["pnl_inr"] > 0 else ("BE" if abs(t["pnl_inr"]) < 50 else "LOSS")
                print(
                    f"   Trade #{i}: [{t['strategy_mode']}] "
                    f"Entry: INR {t['entry_price']:.2f} -> Exit: INR {t['exit_price']:.2f} "
                    f"({t['points']:+.2f} pts | INR {t['pnl_inr']:+.2f} | {pnl_tag}) "
                    f"| Reason: {t['reason']}"
                )
        print("-" * 94 + "\n")


if __name__ == "__main__":
    engine = AlgoBacktestEngine()
    data = engine.generate_benchmark_dataset(n_days=10, bars_per_day=375)
    engine.run_backtest(data)
