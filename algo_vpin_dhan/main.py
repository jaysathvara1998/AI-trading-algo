"""
Main Master Orchestration Engine
VPIN + GARCH(1,1) + SVM Quantitative Trading System on DhanHQ API
Based on Fang & Feng (2019) Market Microstructure Research
"""

import argparse
import asyncio
from datetime import datetime, time
import logging
from pathlib import Path
import sys
from typing import Optional
import pandas as pd
import pytz

# Automatically ensure project root is in sys.path when running directly
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from algo_vpin_dhan.config import AppConfig, CONFIG, InstrumentMode
from algo_vpin_dhan.data_feed import DhanDataFeed, BarOHLCV
from algo_vpin_dhan.vpin import VPINCalculator, ToxicityRegime
from algo_vpin_dhan.garch_engine import GARCHEngine, DirectionalSignal
from algo_vpin_dhan.svm_filter import SVMTradeFilter
from algo_vpin_dhan.risk_manager import RiskManager, PositionSide
from algo_vpin_dhan.execution import ExecutionEngine, OrderType

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("algo_vpin_dhan.main")


class QuantitativeTradingEngine:
    """
    Main algorithmic trading engine integrating:
    - Layer 1: BVC & VPIN (Volume-Synchronized Probability of Informed Trading)
    - Layer 2: Rolling GARCH(1,1) MLE Volatility & Directional Return Forecasting
    - Layer 3: SVM (RBF Kernel) Trade-Veto Filter
    - Dynamic Risk Management & DhanHQ Execution
    """

    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or CONFIG
        self.tz = pytz.timezone("Asia/Kolkata")

        # Initialize Subsystems
        self.data_feed = DhanDataFeed(self.config)
        self.vpin_calc = VPINCalculator(self.config.vpin)
        self.garch_engine = GARCHEngine(self.config.garch)
        self.svm_filter = SVMTradeFilter(self.config.svm)
        self.risk_manager = RiskManager(self.config.risk, self.config.market)
        self.execution = ExecutionEngine(self.config, self.risk_manager)

        self.is_warmed_up = False
        self.total_bars_processed = 0

    def warm_up(self, historical_bars: Optional[pd.DataFrame] = None):
        """
        Pre-market warm-up: Ingests historical bars to initialize BVC buckets,
        GARCH rolling parameters, and SVM trade filter.
        """
        logger.info("Initializing system warm-up phase...")

        if historical_bars is None or historical_bars.empty:
            historical_bars = self.data_feed.fetch_historical_bars(days=5)

        logger.info(f"Ingesting {len(historical_bars)} historical bars for warm-up...")

        # Calculate average daily volume to set volume bucket size V
        if "volume" in historical_bars.columns and len(historical_bars) >= 375:
            avg_daily_vol = historical_bars["volume"].tail(375 * 3).sum() / 3.0
            self.vpin_calc.update_bucket_volume(avg_daily_vol)
            logger.info(f"Calibrated adaptive bucket volume V = {self.vpin_calc.bucket_volume:.1f}")

        # Try loading pre-trained SVM model
        loaded = self.svm_filter.load_model()
        if loaded:
            logger.info("[SVM Engine] Pre-trained brain model successfully loaded. Fast warm-up engaged.")

        # Feed historical bars sequentially
        for idx, row in historical_bars.iterrows():
            ts = row.get("timestamp", pd.Timestamp.now(tz=self.tz))
            close_p = float(row["close"])
            vol = float(row["volume"])

            # 1. Update VPIN
            vpin_res = self.vpin_calc.process_bar(close_p, vol, ts)

            # 2. Update GARCH
            garch_res = self.garch_engine.add_bar(close_p, ts)

            # 3. Update SVM
            price_delta = close_p - (historical_bars.iloc[idx-1]["close"] if idx > 0 else close_p)
            vol_roll = garch_res.sigma_next if garch_res else 0.001
            garch_pred = garch_res.mu_next if garch_res else 0.0

            self.svm_filter.update_bar(
                current_price=close_p,
                price_delta=price_delta,
                rolling_vol=vol_roll,
                garch_forecast=garch_pred
            )

        if not self.svm_filter.is_trained:
            trained = self.svm_filter.train_model()
        else:
            trained = True

        logger.info(f"Warm-up complete. SVM Model trained: {trained}. Completed VPIN buckets: {len(self.vpin_calc.completed_buckets)}")
        self.is_warmed_up = True

    def _record_bar_to_csv(self, bar: BarOHLCV, vpin_res, garch_res, veto_decision):
        """Records every 1-minute candle and quantitative signal to local daily CSV"""
        try:
            date_str = datetime.now(self.tz).strftime("%Y%m%d")
            data_dir = ROOT_DIR / "algo_vpin_dhan" / "data"
            data_dir.mkdir(parents=True, exist_ok=True)
            csv_path = data_dir / f"{self.config.market.underlying.lower()}_1min_{date_str}.csv"

            row = {
                "timestamp": str(bar.timestamp),
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume,
                "vpin": round(vpin_res.vpin, 4),
                "vpin_regime": vpin_res.regime.value,
                "garch_mu": round(garch_res.mu_next, 6),
                "garch_vol": round(garch_res.annualized_vol, 4),
                "garch_signal": garch_res.signal.value,
                "svm_pred": veto_decision.svm_prediction,
                "svm_decision": "VETO" if veto_decision.is_vetoed else "APPROVED"
            }
            df_row = pd.DataFrame([row])
            df_row.to_csv(csv_path, mode="a", header=not csv_path.exists(), index=False)
        except Exception as e:
            logger.debug(f"Could not record bar to CSV: {e}")

    def process_incoming_bar(self, bar: BarOHLCV):
        """
        Core 1-minute bar evaluation cycle:
        1. Check Risk / Exit conditions on active position
        2. Compute Layer 1 VPIN & Toxicity Regime
        3. Compute Layer 2 GARCH(1,1) Directional Forecast
        4. Compute Layer 3 SVM Trade-Veto Filter
        5. Route Orders if Approved
        """
        self.total_bars_processed += 1
        curr_price = bar.close
        timestamp = bar.timestamp

        # --- STEP 1: Risk & Position Management Exit Check ---
        pos = self.risk_manager.current_position
        curr_opt_premium = None
        if pos.side != PositionSide.FLAT and pos.is_option and pos.symbol:
            curr_opt_premium = self.data_feed.fetch_option_ltp(pos.symbol, curr_price)

        exit_needed, exit_reason = self.risk_manager.check_exit_conditions(
            current_price=curr_price,
            option_premium=curr_opt_premium,
            current_time=timestamp.to_pydatetime() if hasattr(timestamp, "to_pydatetime") else None
        )
        if exit_needed:
            exit_exec_price = curr_opt_premium if (pos.is_option and curr_opt_premium) else curr_price
            self.execution.execute_exit(current_price=exit_exec_price, reason=exit_reason)

        # --- STEP 2: Layer 1 - BVC & VPIN Toxicity ---
        vpin_res = self.vpin_calc.process_bar(
            close_price=curr_price,
            volume=bar.volume,
            timestamp=timestamp
        )

        # --- STEP 3: Layer 2 - GARCH(1,1) Direction & Volatility Forecast ---
        garch_res = self.garch_engine.add_bar(
            close_price=curr_price,
            timestamp=timestamp
        )

        if garch_res is None:
            # Need more bars for GARCH warm-up
            return

        price_delta = curr_price - (self.garch_engine.prices[-2] if len(self.garch_engine.prices) >= 2 else curr_price)
        rolling_vol = garch_res.sigma_next
        garch_forecast = garch_res.mu_next

        # --- STEP 4: Layer 3 - SVM Trade-Veto Filter ---
        veto_decision = self.svm_filter.evaluate_signal(
            garch_signal=garch_res.signal,
            price_delta=price_delta,
            rolling_vol=rolling_vol,
            garch_forecast=garch_forecast
        )

        # Update SVM training dataset
        self.svm_filter.update_bar(
            current_price=curr_price,
            price_delta=price_delta,
            rolling_vol=rolling_vol,
            garch_forecast=garch_forecast
        )

        # Telemetry logging on every minute candle
        pos_str = f"{pos.symbol} (Qty {pos.quantity})" if pos.side != PositionSide.FLAT else "FLAT"
        logger.info(
            f"[BAR #{self.total_bars_processed:04d}] Price: {curr_price:.2f} | "
            f"VPIN: {vpin_res.vpin:.3f} ({vpin_res.regime.value}) | "
            f"GARCH: {garch_res.signal.value} (mu={garch_res.mu_next:+.6f}, vol={garch_res.annualized_vol*100:.1f}%) | "
            f"SVM: {'VETO' if veto_decision.is_vetoed else ('APPROVED' if garch_res.signal != DirectionalSignal.HOLD else 'NEUTRAL')} | "
            f"Pos: {pos_str}"
        )
        self._record_bar_to_csv(bar, vpin_res, garch_res, veto_decision)

        # Check Emergency Signal Reversal on open position
        if pos.side != PositionSide.FLAT and not veto_decision.is_vetoed:
            rev_exit, rev_reason = self.risk_manager.check_exit_conditions(
                current_price=curr_price,
                option_premium=curr_opt_premium,
                current_time=timestamp.to_pydatetime() if hasattr(timestamp, "to_pydatetime") else None,
                reversal_signal=veto_decision.final_action
            )
            if rev_exit:
                exit_exec_price = curr_opt_premium if (pos.is_option and curr_opt_premium) else curr_price
                self.execution.execute_exit(current_price=exit_exec_price, reason=rev_reason)

        # --- STEP 5: Order Execution on Approved Signals (Options Trading) ---
        if not veto_decision.is_vetoed and veto_decision.final_action in (DirectionalSignal.BUY, DirectionalSignal.SELL):
            is_option = (self.config.market.instrument_mode == InstrumentMode.OPTIONS_BUYING)
            target_symbol = None
            opt_premium = None
            action_str = "BUY" if veto_decision.final_action == DirectionalSignal.BUY else "SELL"

            if is_option:
                target_symbol = self.data_feed.resolve_option_strike(action_str, curr_price)
                opt_premium = self.data_feed.fetch_option_ltp(target_symbol, curr_price)
                exec_price = opt_premium
                logger.info(f"[OPTIONS BUYING] Signal {veto_decision.final_action.value} -> Contract: {target_symbol} | Live Premium: ₹{opt_premium:.2f}")
            else:
                exec_price = curr_price

            # Compute volatility-scaled trade targets & position size on Option Premium
            targets = self.risk_manager.compute_trade_targets(
                signal=veto_decision.final_action,
                current_price=curr_price,
                garch_res=garch_res,
                vpin_res=vpin_res,
                is_option=is_option,
                option_premium=opt_premium
            )

            if targets:
                self.execution.execute_entry(
                    signal=veto_decision.final_action,
                    current_price=exec_price,
                    trade_target=targets,
                    vpin_res=vpin_res,
                    order_type=OrderType.MARKET,
                    target_symbol=target_symbol
                )

    def run_simulation(self, bars_df: pd.DataFrame):
        """
        Runs backtest / offline paper trading simulation across historical/synthetic bars.
        """
        logger.info(f"Starting simulation run for {self.config.market.underlying} Options across {len(bars_df)} bars...")
        self.warm_up(bars_df.head(150))

        for idx, row in bars_df.iloc[150:].iterrows():
            bar = BarOHLCV(
                timestamp=row.get("timestamp", pd.Timestamp.now(tz=self.tz)),
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]),
                symbol=self.config.market.symbol
            )
            self.process_incoming_bar(bar)

        # Print performance summary
        logger.info("=" * 60)
        logger.info(f"SIMULATION RUN COMPLETE ({self.config.market.underlying} OPTIONS)")
        logger.info(f"Total Orders Executed: {len(self.execution.order_book)}")
        logger.info(f"Final Realized PnL: {self.risk_manager.daily_realized_pnl:+.2f} INR")
        logger.info(f"Kill Switch Triggered: {self.risk_manager.is_kill_switch_active}")
        logger.info("=" * 60)
    async def run_live_session(self):
        """
        Main live market loop with Dual-Speed Monitoring:
        - 1-Minute Bar Engine for VPIN, GARCH, and SVM ML Brain.
        - Fast 3-Second Micro-Loop for active Position Risk (catches quick wicks, spikes, and flash dumps).
        """
        logger.info(f"Starting Live Algo VPIN v1.0 Options Engine for {self.config.market.underlying}...")
        self.warm_up()

        last_bar_minute = -1

        while True:
            now_ist = datetime.now(self.tz)
            market_open = now_ist.replace(hour=self.config.market.market_open_hour, minute=self.config.market.market_open_minute, second=0)
            market_close = now_ist.replace(hour=self.config.market.market_close_hour, minute=self.config.market.market_close_minute, second=0)

            if now_ist < market_open:
                wait_sec = (market_open - now_ist).total_seconds()
                logger.info(f"Market pre-open. Waiting {wait_sec:.0f}s until 09:15 IST...")
                await asyncio.sleep(min(wait_sec, 30))
                continue

            if now_ist >= market_close:
                logger.info("Market closed for the day (15:30 IST). Generating post-session report...")
                break

            pos = self.risk_manager.current_position

            # --- FAST GUARDIAN: Check active position every 3-5 seconds for instant wick exits ---
            if pos.side != PositionSide.FLAT and pos.symbol:
                try:
                    curr_spot = self.data_feed.fetch_live_spot_price()
                    curr_opt_ltp = self.data_feed.fetch_option_ltp(pos.symbol, curr_spot)
                    exit_needed, exit_reason = self.risk_manager.check_exit_conditions(
                        current_price=curr_spot,
                        option_premium=curr_opt_ltp,
                        current_time=now_ist
                    )
                    if exit_needed:
                        logger.info(f"[FAST GUARDIAN EXIT] Instant trigger: {exit_reason} @ LTP INR {curr_opt_ltp:.2f}")
                        self.execution.execute_exit(current_price=curr_opt_ltp, reason=exit_reason)
                except Exception as e:
                    logger.debug(f"Fast guardian check: {e}")

            # --- 1-MINUTE ENGINE: Process Bar when minute rolls over ---
            current_minute = now_ist.minute
            if current_minute != last_bar_minute:
                last_bar_minute = current_minute
                try:
                    live_spot = self.data_feed.fetch_live_spot_price()
                    hist = self.data_feed.fetch_historical_bars(days=1)
                    if not hist.empty:
                        latest = hist.iloc[-1]
                        bar = BarOHLCV(
                            timestamp=pd.Timestamp(now_ist),
                            open=live_spot,
                            high=max(live_spot, float(latest.get("high", live_spot))),
                            low=min(live_spot, float(latest.get("low", live_spot))),
                            close=live_spot,
                            volume=float(latest.get("volume", 25000.0)),
                            symbol=self.config.market.symbol
                        )
                        self.process_incoming_bar(bar)
                except Exception as e:
                    logger.error(f"Error in 1-minute polling cycle: {e}")

            # Sleep short interval (3s if in trade, 5s if flat) for high-frequency responsiveness
            sleep_duration = 3 if pos.side != PositionSide.FLAT else 5
            await asyncio.sleep(sleep_duration)


def main():
    parser = argparse.ArgumentParser(description="VPIN + GARCH(1,1) + SVM Quantitative Options Trading Engine")
    parser.add_argument("--mode", choices=["live", "simulation"], default="simulation", help="Run mode: live or simulation")
    parser.add_argument("--symbol", choices=["NIFTY", "SENSEX"], default="NIFTY", help="Underlying Index (NIFTY or SENSEX)")
    parser.add_argument("--bars", type=int, default=500, help="Number of bars for simulation")
    args = parser.parse_args()

    engine = QuantitativeTradingEngine()
    if args.symbol:
        engine.config.market.set_symbol(args.symbol)

    if args.mode == "simulation":
        start_price = 24000.0 if engine.config.market.underlying == "NIFTY" else 79000.0
        bars_df = engine.data_feed.generate_synthetic_bars(n_bars=args.bars, start_price=start_price)
        engine.run_simulation(bars_df)
    else:
        asyncio.run(engine.run_live_session())


if __name__ == "__main__":
    main()
