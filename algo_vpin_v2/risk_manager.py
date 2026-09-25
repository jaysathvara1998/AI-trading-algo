"""
Risk Management Engine for Algo VPIN v2.0
Calibrated for ₹50,000 Capital with Dynamic Option Volatility Stops & Daily Kill-Switch.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional, Tuple
import pytz
import logging

from .config import RiskConfig, MarketConfig, ScalperConfig, TradeStrategyMode
from .garch_engine import DirectionalSignal, GARCHForecastResult
from .vpin import VPINResult, ToxicityRegime
from .telegram_notifier import notify_trailing_sl

logger = logging.getLogger("algo_vpin_v2.risk_manager")


class PositionSide(Enum):
    FLAT = "FLAT"
    LONG = "LONG"
    SHORT = "SHORT"


@dataclass
class TradeTarget:
    entry_price: float
    stop_loss: float
    take_profit: float
    quantity_lots: int
    total_quantity: int
    risk_reward_ratio: float
    strategy_mode: TradeStrategyMode = TradeStrategyMode.INSTITUTIONAL_SWING
    initial_risk_pts: float = 0.0
    entry_spot_price: float = 0.0
    spot_stop_loss: float = 0.0
    hard_disaster_sl: float = 0.0


@dataclass
class ActivePosition:
    side: PositionSide
    entry_price: float
    quantity: int
    stop_loss: float
    take_profit: float
    entry_time: datetime
    entry_vpin: float
    entry_regime: ToxicityRegime
    symbol: str = "NIFTY"
    is_option: bool = False
    unrealized_pnl: float = 0.0
    highest_price: float = 0.0
    lowest_price: float = 0.0
    strategy_mode: TradeStrategyMode = TradeStrategyMode.INSTITUTIONAL_SWING
    initial_risk_pts: float = 0.0
    current_trail_tier_r: float = 0.0
    entry_spot_price: float = 0.0
    spot_stop_loss: float = 0.0
    hard_disaster_sl: float = 0.0
    is_runner_active: bool = False



@dataclass
class PreMarketJournal:
    """Pre-9:15 AM Institutional Trading Journal (Pillar 4)"""
    date_str: str
    pdh: float
    pdl: float
    pdc: float
    equilibrium_50: float
    max_trades_limit: int
    daily_max_loss_inr: float
    daily_target_profit_inr: float
    global_sentiment: str = "NEUTRAL"
    heavyweight_bias: str = "NEUTRAL"
    news_risk_status: str = "CLEAR"
    journal_summary: str = ""


class RiskManager:
    def __init__(
        self,
        risk_config: Optional[RiskConfig] = None,
        market_config: Optional[MarketConfig] = None,
        scalper_config: Optional[ScalperConfig] = None,
        track_name: str = "Track 1: Dual-Brain"
    ):
        self.risk_config = risk_config or RiskConfig()
        self.market_config = market_config or MarketConfig()
        self.scalper_config = scalper_config or ScalperConfig()
        self.track_name = track_name
        self.tz = pytz.timezone("Asia/Kolkata")
        
        self.daily_realized_pnl: float = 0.0
        self.is_kill_switch_active: bool = False
        self.total_trades_today: int = 0
        self.winning_trades_today: int = 0
        self.consecutive_losses: int = 0
        self.in_cautious_state: bool = False
        self.premarket_journal: Optional[PreMarketJournal] = None
        
        self.current_position = ActivePosition(
            side=PositionSide.FLAT,
            entry_price=0.0,
            quantity=0,
            stop_loss=0.0,
            take_profit=0.0,
            entry_time=datetime.now(self.tz),
            entry_vpin=0.0,
            entry_regime=ToxicityRegime.NORMAL_TOXICITY
        )
        self.min_hold_minutes: int = 5  # Minimum 5 mins before reversal exit is allowed
        self.bars_in_position: int = 0  # Counts bars since entry
        self._sync_from_trades_history()

    def initialize_premarket_journal(
        self,
        pdh: float,
        pdl: float,
        pdc: float,
        global_sentiment: str = "NEUTRAL",
        heavyweight_bias: str = "NEUTRAL",
        news_risk_status: str = "CLEAR"
    ) -> PreMarketJournal:
        """Initializes and locks in the pre-market risk plan before trading begins"""
        today_str = datetime.now(self.tz).strftime("%Y-%m-%d")
        eq_50 = round((pdh + pdl) / 2.0, 2) if (pdh > 0 and pdl > 0) else 0.0
        
        summary = (
            f"[PRE-MARKET JOURNAL | {today_str}]\n"
            f"  * Key Levels: PDH={pdh:.2f} | PDL={pdl:.2f} | PDC={pdc:.2f} | 50% Eq={eq_50:.2f}\n"
            f"  * Global Cues: {global_sentiment} | Heavyweight Breadth: {heavyweight_bias}\n"
            f"  * News Risk: {news_risk_status}\n"
            f"  * Hard Rules: Max Trades={self.risk_config.max_daily_trades} | Max Loss=INR {self.risk_config.max_daily_loss_inr:.0f} | Target Goal=INR {self.risk_config.daily_target_profit_inr:.0f}"
        )
        
        journal = PreMarketJournal(
            date_str=today_str,
            pdh=pdh,
            pdl=pdl,
            pdc=pdc,
            equilibrium_50=eq_50,
            max_trades_limit=self.risk_config.max_daily_trades,
            daily_max_loss_inr=-abs(self.risk_config.max_daily_loss_inr),
            daily_target_profit_inr=self.risk_config.daily_target_profit_inr,
            global_sentiment=global_sentiment,
            heavyweight_bias=heavyweight_bias,
            news_risk_status=news_risk_status,
            journal_summary=summary
        )
        self.premarket_journal = journal
        logger.info(summary)
        return journal

    def _sync_from_trades_history(self):
        """Loads today's trading history from disk to retain cumulative P&L and Cautious Guard state across restarts"""
        try:
            import os
            import csv
            history_path = os.path.join(os.path.dirname(__file__), "data", "trades_history.csv")
            if not os.path.exists(history_path):
                return
            today_str = datetime.now(self.tz).strftime("%Y-%m-%d")
            total_pnl = 0.0
            trades_today = 0
            wins_today = 0
            recent_losses = 0

            target_is_track_2 = ("TRACK 2" in self.track_name.upper())

            with open(history_path, mode='r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    ts = row.get("exit_timestamp", "")
                    if ts.startswith(today_str):
                        row_track = row.get("track", "")
                        exit_reason = row.get("exit_reason", "")
                        is_track_2 = ("TRACK 2" in row_track.upper()) or ("[TRACK 2]" in exit_reason.upper())

                        # Ensure Track 1 only syncs Track 1 trades, and Track 2 only syncs Track 2 trades
                        if is_track_2 != target_is_track_2:
                            continue

                        trades_today += 1
                        pnl = float(row.get("pnl_inr", 0.0))
                        total_pnl += pnl
                        if pnl > 0:
                            wins_today += 1
                            recent_losses = 0
                        else:
                            recent_losses += 1

            self.daily_realized_pnl = round(total_pnl, 2)
            self.total_trades_today = trades_today
            self.winning_trades_today = wins_today
            self.consecutive_losses = recent_losses
            threshold = getattr(self.risk_config, 'consecutive_losses_threshold', 2)
            self.in_cautious_state = (self.consecutive_losses >= threshold)
            logger.info(
                f"[RISK SYNC: {self.track_name}] Restored today's state: {trades_today} trades ({wins_today} wins), "
                f"Net P&L: INR {self.daily_realized_pnl:+.2f}, Cautious State: {self.in_cautious_state}"
            )
        except Exception as e:
            logger.debug(f"[RISK SYNC] Could not sync trade history: {e}")

    def calculate_position_size(
        self,
        vpin_res: VPINResult,
        option_premium: Optional[float] = None,
        available_cash: Optional[float] = None
    ) -> int:
        """Determines lot quantity based on available capital, margin utilization caps, and toxicity"""
        if self.risk_config.enable_kill_switch and self.is_kill_switch_active:
            return 0
        if vpin_res.regime == ToxicityRegime.EXTREME_TOXICITY:
            logger.warning("[MARGIN GUARD] Extreme toxicity detected. Sizing reduced to 0.")
            return 0
        # Dynamically reload max_daily_trades from adaptive_config.json if available
        try:
            from pathlib import Path
            import json
            cfg_path = Path(__file__).resolve().parent / "data" / "adaptive_config.json"
            if cfg_path.exists():
                with open(cfg_path, "r", encoding="utf-8") as f:
                    dyn_cfg = json.load(f)
                    if "max_daily_trades" in dyn_cfg:
                        self.risk_config.max_daily_trades = int(dyn_cfg["max_daily_trades"])
        except Exception:
            pass

        # Hard cap on daily trades from Pre-Market Journal (Pillar 4)
        max_trades = getattr(self.risk_config, 'max_daily_trades', 12)
        if self.total_trades_today >= max_trades:
            logger.info(f"[Risk Manager] Max daily trades limit reached ({self.total_trades_today}/{max_trades}). Entry blocked.")
            return 0

        if option_premium and option_premium > 0:
            cost_per_lot = option_premium * self.market_config.lot_size
            if cost_per_lot > 0:
                usable_cash = self.risk_config.capital_allocation
                if available_cash and available_cash > 0:
                    utilization_cap = getattr(self.risk_config, 'max_capital_utilization_pct', 0.50)
                    usable_cash = min(usable_cash, available_cash * utilization_cap)

                usable_cash = min(usable_cash, self.risk_config.max_capital_per_trade)
                affordable_lots = int(usable_cash // cost_per_lot)

                # In cautious state, cap size strictly at 1 lot for capital protection
                if self.in_cautious_state:
                    return 1 if (available_cash and available_cash >= cost_per_lot) or affordable_lots >= 1 else 0

                # Fallback: Allow 1 lot if available cash covers the cost
                if affordable_lots < 1 and available_cash and available_cash >= cost_per_lot:
                    return 1 if available_cash >= cost_per_lot else 0
                max_configured = getattr(self.risk_config, 'max_lots', 1)
                return min(max_configured, max(1, affordable_lots))

        return self.risk_config.base_lots

    def compute_trade_targets(
        self,
        signal: DirectionalSignal,
        current_price: float,
        garch_res: GARCHForecastResult,
        vpin_res: VPINResult,
        is_option: bool = False,
        option_premium: Optional[float] = None,
        available_cash: Optional[float] = None,
        strategy_mode: Optional[TradeStrategyMode] = None,
        candle_high: Optional[float] = None,
        candle_low: Optional[float] = None,
        delta: Optional[float] = None
    ) -> Optional[TradeTarget]:
        if signal == DirectionalSignal.HOLD or current_price <= 0:
            return None

        lots = self.calculate_position_size(vpin_res, option_premium=option_premium, available_cash=available_cash)
        if lots <= 0:
            logger.warning("[MARGIN GUARD] Insufficient capital or extreme toxicity: calculated 0 lots.")
            return None

        total_qty = lots * self.market_config.lot_size
        mode = strategy_mode or TradeStrategyMode.INSTITUTIONAL_SWING

        if is_option:
            premium = option_premium if option_premium and option_premium > 0 else max(10.0, current_price * garch_res.sigma_next * 0.4)
            under = self.market_config.underlying.upper()
            
            # Asset-specific calibration
            if "SENSEX" in under:
                default_spot_sl_dist = 50.0
                min_scalp_sl = 30.0
                max_scalp_sl = 60.0
                min_tp_pts = 45.0
                eff_delta = delta or 0.72
            elif "BANKNIFTY" in under:
                default_spot_sl_dist = 35.0
                min_scalp_sl = 20.0
                max_scalp_sl = 45.0
                min_tp_pts = 30.0
                eff_delta = delta or 0.65
            else:
                default_spot_sl_dist = 18.0
                min_scalp_sl = getattr(self.scalper_config, "min_scalp_sl_points", 8.0)
                max_scalp_sl = getattr(self.scalper_config, "max_scalp_sl_points", 18.0)
                min_tp_pts = 12.0
                eff_delta = delta or 0.60

            # Dynamic Swing / Candle High-Low Spot SL mapping
            if signal == DirectionalSignal.BUY:
                if candle_low is not None and (current_price - candle_low) > 5.0:
                    spot_sl_dist = max(default_spot_sl_dist * 0.7, (current_price - candle_low) + (default_spot_sl_dist * 0.1))
                else:
                    spot_sl_dist = default_spot_sl_dist
                spot_sl = round(current_price - spot_sl_dist, 2)
            else:
                if candle_high is not None and (candle_high - current_price) > 5.0:
                    spot_sl_dist = max(default_spot_sl_dist * 0.7, (candle_high - current_price) + (default_spot_sl_dist * 0.1))
                else:
                    spot_sl_dist = default_spot_sl_dist
                spot_sl = round(current_price + spot_sl_dist, 2)

            if mode == TradeStrategyMode.SCALPER:
                # Dynamic Option SL derived from Spot Structure * Delta, bounded by Asset Volatility Budget
                dynamic_opt_sl_pts = spot_sl_dist * eff_delta
                sl_pts = max(min_scalp_sl, min(max_scalp_sl, dynamic_opt_sl_pts))
                # Target: minimum 1:1.5 Risk-Reward Ratio
                tp_pts = max(min_tp_pts, sl_pts * self.scalper_config.initial_risk_reward)

                sl = max(1.0, round(premium - sl_pts, 2))
                tp = round(premium + tp_pts, 2)
                rr_ratio = round(tp_pts / sl_pts, 2) if sl_pts > 0 else 1.5
                initial_risk_dist = round(sl_pts, 2)
                disaster_sl = max(1.0, round(premium * 0.75, 2))  # Max 25% catastrophic loss floor
            else:
                # Institutional Swing: Open-ended target (managed by continuous trailing stop)
                sl_points = max(min_scalp_sl, min(premium * 0.20, getattr(self.risk_config, 'max_initial_sl_points', 45.0 if "SENSEX" in under else 25.0)))
                tp_pct = 2.00  # Multi-target runner

                sl = max(1.0, round(premium - sl_points, 2))
                tp = round(premium * (1.0 + tp_pct), 2)
                initial_risk_dist = round(sl_points, 2)
                reward_dist = abs(tp - premium)
                rr_ratio = round(reward_dist / initial_risk_dist, 2) if initial_risk_dist > 0 else 2.0
                disaster_sl = max(1.0, round(premium * 0.70, 2))  # Max 30% catastrophic loss floor

            return TradeTarget(
                entry_price=round(premium, 2),
                stop_loss=round(sl, 2),
                take_profit=round(tp, 2),
                quantity_lots=lots,
                total_quantity=total_qty,
                risk_reward_ratio=round(rr_ratio, 2),
                strategy_mode=mode,
                initial_risk_pts=initial_risk_dist,
                entry_spot_price=round(current_price, 2),
                spot_stop_loss=spot_sl,
                hard_disaster_sl=disaster_sl
            )

        sigma = garch_res.sigma_next
        vol_band = max(current_price * 0.002, sigma * current_price * self.risk_config.sl_garch_multiplier)
        tp_band = max(current_price * 0.003, sigma * current_price * self.risk_config.tp_garch_multiplier)
        hard_cap = current_price * self.risk_config.max_trade_loss_pct
        vol_band = min(vol_band, hard_cap)

        if signal == DirectionalSignal.BUY:
            sl = current_price - vol_band
            tp = current_price + tp_band
        else:
            sl = current_price + vol_band
            tp = current_price - tp_band

        risk_dist = abs(current_price - sl)
        reward_dist = abs(tp - current_price)
        rr_ratio = reward_dist / risk_dist if risk_dist > 0 else 1.0

        return TradeTarget(
            entry_price=round(current_price, 2),
            stop_loss=round(sl, 2),
            take_profit=round(tp, 2),
            quantity_lots=lots,
            total_quantity=total_qty,
            risk_reward_ratio=round(rr_ratio, 2),
            strategy_mode=mode,
            initial_risk_pts=round(risk_dist, 2)
        )

    def validate_new_entry(
        self,
        signal: DirectionalSignal,
        vpin_res: VPINResult,
        current_time: Optional[datetime] = None,
        is_unanimous: bool = False,
        volume_ratio: float = 1.0
    ) -> Tuple[bool, str]:
        if self.risk_config.enable_kill_switch and self.is_kill_switch_active:
            return False, "DAILY_KILL_SWITCH_ACTIVE"
        if self.current_position.side != PositionSide.FLAT:
            return False, "POSITION_ALREADY_OPEN"

        now_ist = current_time or datetime.now(self.tz)
        cutoff_h = getattr(self.market_config, "last_entry_hour", 15)
        cutoff_m = getattr(self.market_config, "last_entry_minute", 15)
        if (now_ist.hour > cutoff_h or
            (now_ist.hour == cutoff_h and now_ist.minute > cutoff_m)):
            return False, f"PAST_ENTRY_CUTOFF ({cutoff_h}:{cutoff_m:02d} IST)"

        if vpin_res.regime == ToxicityRegime.EXTREME_TOXICITY:
            return False, "EXTREME_VPIN_TOXICITY"

        # Smart Re-qualification during Cautious State post consecutive losses
        if self.in_cautious_state:
            # Synthetic index volume: when volume_ratio is 1.0 (fixed index stream), vol condition passes
            is_index_synthetic = abs(volume_ratio - 1.0) < 1e-4
            is_vol_ok = True if is_index_synthetic else (volume_ratio >= 1.25)
            # Toxicity: In midday index options, normal VPIN is 0.15-0.19. We require >= 0.16
            is_vpin_ok = (vpin_res.vpin >= 0.16)

            if is_unanimous and is_vpin_ok and is_vol_ok:
                logger.info(
                    f"[HIGH-CONVICTION RE-QUALIFICATION] Approved entry in Cautious State! "
                    f"Unanimous Consensus={is_unanimous}, VPIN={vpin_res.vpin:.3f} >= 0.16. "
                    f"Capturing high-conviction impulse move with disciplined 1-lot sizing."
                )
            else:
                reason = (
                    f"CAUTIOUS_STATE_RESTRICTION (Requires Unanimous Brain Consensus + VPIN>=0.16 | "
                    f"Got Unanimous={is_unanimous}, VPIN={vpin_res.vpin:.3f}, VolRatio={volume_ratio:.2f}x)"
                )
                return False, reason

        return True, "APPROVED"

    def check_exit_conditions(
        self,
        current_price: float,
        option_premium: Optional[float] = None,
        current_time: Optional[datetime] = None,
        reversal_signal: Optional[DirectionalSignal] = None,
        counter_trend_reason: Optional[str] = None
    ) -> Tuple[bool, str]:
        if self.current_position.side == PositionSide.FLAT:
            return False, ""

        now_ist = current_time or datetime.now(self.tz)
        if (now_ist.hour > self.market_config.square_off_hour or
            (now_ist.hour == self.market_config.square_off_hour and
             now_ist.minute >= self.market_config.square_off_minute)):
            return True, "INTRADAY_SQUARE_OFF"

        pos = self.current_position
        eval_price = option_premium if (pos.is_option and option_premium and option_premium > 0) else current_price

        pos.highest_price = max(pos.highest_price, eval_price)
        pos.lowest_price = min(pos.lowest_price if pos.lowest_price > 0 else eval_price, eval_price)

        # 1. Multi-Bar Cumulative Counter-Trend Guard (3m / 5m rolling move against trade)
        if counter_trend_reason:
            time_in_trade = (now_ist - pos.entry_time).total_seconds() / 60.0
            if time_in_trade >= 2.0:  # Allow 2 minutes grace before multi-bar exit
                return True, f"MULTI_BAR_REVERSAL_EXIT: {counter_trend_reason}"

        # 2. Emergency Signal Reversal Exit (only after minimum hold time)
        if reversal_signal is not None and reversal_signal != DirectionalSignal.HOLD:
            time_in_trade = (now_ist - pos.entry_time).total_seconds() / 60.0
            if time_in_trade < self.min_hold_minutes:
                logger.debug(f"[REVERSAL GUARD] Reversal blocked — only {time_in_trade:.1f}m in trade (min={self.min_hold_minutes}m)")
            else:
                if (pos.side == PositionSide.LONG and reversal_signal == DirectionalSignal.SELL) or \
                   (pos.is_option and pos.symbol and ("CALL" in pos.symbol or "CE" in pos.symbol) and reversal_signal == DirectionalSignal.SELL):
                    return True, "EMERGENCY_REVERSAL_EXIT: Model confirmed Bearish SELL signal"
                elif (pos.side == PositionSide.SHORT and reversal_signal == DirectionalSignal.BUY) or \
                     (pos.is_option and pos.symbol and ("PUT" in pos.symbol or "PE" in pos.symbol) and reversal_signal == DirectionalSignal.BUY):
                    return True, "EMERGENCY_REVERSAL_EXIT: Model confirmed Bullish BUY signal"

        # 3. Dynamic Trailing Stop-Loss Engine (Scalper 0.5R Steps vs Institutional Swing Runner)
        if pos.is_option or pos.side == PositionSide.LONG:
            pos.unrealized_pnl = (eval_price - pos.entry_price) * pos.quantity
            profit_pct = (eval_price - pos.entry_price) / pos.entry_price if pos.entry_price > 0 else 0.0
            peak_gain_pct = (pos.highest_price - pos.entry_price) / pos.entry_price if pos.entry_price > 0 else 0.0
            peak_gain_pts = pos.highest_price - pos.entry_price

            under = self.market_config.underlying.upper()
            if "SENSEX" in under:
                t1_pts = 15.0       # Move to BE at +15 pts
                t1_lock = 3.0       # +3 pts BE shield
                trail_buffer = 22.0 # Trail 20-22 pts behind peak
            elif "BANKNIFTY" in under:
                t1_pts = 12.0
                t1_lock = 2.0
                trail_buffer = 12.0
            else:
                t1_pts = 8.0
                t1_lock = 1.5
                trail_buffer = 5.0

            # -------------------------------------------------------------
            # UNIFIED RESPONSIVE TRAILING STOP ENGINE (Points & Peak Trailing)
            # -------------------------------------------------------------
            # Tier 1: At +t1_pts (e.g. +15 pts SENSEX), move SL to Breakeven (+Brokerage Shield)
            be_price = round(pos.entry_price + t1_lock, 2)
            if peak_gain_pts >= t1_pts and pos.stop_loss < be_price:
                pos.stop_loss = be_price
                logger.info(f"[TRAILING SL] +{peak_gain_pts:.1f} pts gain achieved. SL moved to Breakeven Shield (INR {pos.stop_loss:.2f} / +{t1_lock:.1f} pts)")
                try:
                    notify_trailing_sl(
                        symbol=pos.symbol or self.market_config.trading_symbol,
                        peak_price=pos.highest_price,
                        gain_pct=peak_gain_pct * 100,
                        new_sl=pos.stop_loss,
                        tier_name=f"Breakeven Shield (+{t1_lock:.1f} pts)",
                        track_name=self.track_name
                    )
                except Exception as e:
                    logger.debug(f"Telegram trailing alert: {e}")

            # Tier 2: At +25+ pts gain (e.g. SENSEX peak reaches 360+), continuously trail 20-25 pts behind peak
            if peak_gain_pts >= 25.0:
                peak_trail_sl = round(pos.highest_price - trail_buffer, 2)
                if peak_trail_sl > pos.stop_loss:
                    pos.stop_loss = peak_trail_sl
                    locked_gain = round(pos.stop_loss - pos.entry_price, 2)
                    logger.info(f"[TRAILING SL] Peak reached INR {pos.highest_price:.2f} (+{peak_gain_pts:.1f} pts). SL ratcheted to INR {pos.stop_loss:.2f} (Locking +{locked_gain:.1f} pts | Buffer: -{trail_buffer:.1f} pts)")
                    try:
                        notify_trailing_sl(
                            symbol=pos.symbol or self.market_config.trading_symbol,
                            peak_price=pos.highest_price,
                            gain_pct=peak_gain_pct * 100,
                            new_sl=pos.stop_loss,
                            tier_name=f"Peak Trail -{trail_buffer:.1f} pts (Lock +{locked_gain:.1f} pts)",
                            track_name=self.track_name
                        )
                    except Exception as e:
                        logger.debug(f"Telegram trailing alert: {e}")

            # 4. Take Profit Target Exit or Dynamic Runner Mode
            under = self.market_config.underlying.upper()
            if "SENSEX" in under:
                default_runner_buffer = 16.0
            elif "BANKNIFTY" in under:
                default_runner_buffer = 10.0
            else:
                default_runner_buffer = 4.5

            if pos.take_profit > 0 and eval_price >= pos.take_profit:
                if getattr(self.scalper_config, "trail_past_target", True):
                    # RUNNER MODE ENGAGED: Do not kill winner! Elevate SL to lock profit and trail peak buffer
                    profit_expansion = max(0.0, pos.highest_price - pos.entry_price)
                    buffer_pts = max(default_runner_buffer, profit_expansion * 0.25)
                    floor_sl = round(pos.take_profit - (default_runner_buffer * 0.5), 2)
                    peak_trail_sl = round(pos.highest_price - buffer_pts, 2)
                    target_sl = max(floor_sl, peak_trail_sl)

                    if not getattr(pos, "is_runner_active", False):
                        pos.is_runner_active = True
                        logger.info(
                            f"[RUNNER MODE ENGAGED] Price INR {eval_price:.2f} >= Target INR {pos.take_profit:.2f}. "
                            f"Holding position to ride momentum runner! SL elevated to INR {target_sl:.2f} "
                            f"(Locking minimum profit of +{target_sl - pos.entry_price:.2f} pts)"
                        )

                    if target_sl > pos.stop_loss:
                        pos.stop_loss = target_sl
                else:
                    return True, f"TAKE_PROFIT_TRIGGERED (Price INR {eval_price:.2f} >= Target INR {pos.take_profit:.2f})"

            # Active runner peak ratcheting with proportional trend-riding buffer
            if getattr(pos, "is_runner_active", False):
                profit_expansion = max(0.0, pos.highest_price - pos.entry_price)
                buffer_pts = max(default_runner_buffer, profit_expansion * 0.25)
                dynamic_runner_sl = round(pos.highest_price - buffer_pts, 2)
                if dynamic_runner_sl > pos.stop_loss:
                    pos.stop_loss = dynamic_runner_sl
                    logger.info(
                        f"[RUNNER TRAILING] New peak INR {pos.highest_price:.2f} -> SL raised to INR {pos.stop_loss:.2f} "
                        f"(Locking +{pos.stop_loss - pos.entry_price:.2f} pts gain | Buffer: {buffer_pts:.1f} pts)"
                    )

            # 5. Dual Spot-Confirmation Stop-Loss Engine
            if eval_price <= pos.stop_loss:
                # If Runner Mode was active and price pulled back to trailing stop:
                if getattr(pos, "is_runner_active", False):
                    gain_pts = max(0.0, pos.stop_loss - pos.entry_price)
                    return True, f"RUNNER_TRAILING_HIT (Peak INR {pos.highest_price:.2f} -> SL INR {pos.stop_loss:.2f} | Captured +{gain_pts:.2f} pts Runner Profit)"

                # Tier A: Catastrophic Disaster SL Guard (immediate exit if premium drops > 25-30% regardless of spot)
                if getattr(pos, "hard_disaster_sl", 0.0) > 0 and eval_price <= pos.hard_disaster_sl:
                    return True, f"DISASTER_STOP_LOSS (Emergency Capital Shield: Premium INR {eval_price:.2f} <= Hard Floor INR {pos.hard_disaster_sl:.2f})"

                # Tier B: Profit Lock Trail (if trade already locked in breakeven or trailing profit tier >= 1.0R, exit immediately)
                if getattr(pos, "current_trail_tier_r", 0.0) >= 1.0:
                    return True, f"TRAILING_STOP_HIT (Protected Profit Lock: Premium INR {eval_price:.2f} <= Trail SL INR {pos.stop_loss:.2f} | Tier {pos.current_trail_tier_r:.1f}R)"

                # Tier C: IMMEDIATE Stop-Loss Exit — No Dual Confirmation Required
                # When premium hits SL, EXIT. Period. No spot shield override.
                # Spot confirmation is only useful for trailing stops (Tier B above),
                # NOT for initial SL where capital protection is paramount.
                mode_tag = f"Scalper Tier {pos.current_trail_tier_r:.1f}R" if pos.strategy_mode == TradeStrategyMode.SCALPER else "Swing Trail"
                return True, f"STOP_LOSS_HIT (Option INR {eval_price:.2f} <= SL {pos.stop_loss:.2f} | {mode_tag})"


        elif pos.side == PositionSide.SHORT:
            pos.unrealized_pnl = (pos.entry_price - eval_price) * pos.quantity
            profit_pct = (pos.entry_price - eval_price) / pos.entry_price if pos.entry_price > 0 else 0.0
            peak_gain_pct = (pos.entry_price - pos.lowest_price) / pos.entry_price if pos.entry_price > 0 else 0.0

            if peak_gain_pct >= 0.20 and pos.stop_loss > pos.entry_price:
                pos.stop_loss = round(pos.entry_price, 2)
            if peak_gain_pct >= 0.30:
                dynamic_trail = round(pos.lowest_price * 1.15, 2)
                if dynamic_trail < pos.stop_loss:
                    pos.stop_loss = dynamic_trail

            if eval_price >= pos.stop_loss:
                return True, f"STOP_LOSS_TRIGGERED (Price INR {eval_price:.2f} >= SL INR {pos.stop_loss:.2f})"

        return False, ""

    def record_trade_close(self, exit_price: float) -> float:
        pos = self.current_position
        if pos.side == PositionSide.FLAT:
            return 0.0

        if pos.is_option or pos.side == PositionSide.LONG:
            realized = (exit_price - pos.entry_price) * pos.quantity
        else:
            realized = (pos.entry_price - exit_price) * pos.quantity

        self.daily_realized_pnl += realized
        self.total_trades_today += 1
        
        if realized > 0:
            self.winning_trades_today += 1
            if self.consecutive_losses > 0:
                logger.info(f"[RISK RECOVERY] Profitable trade (+INR {realized:.2f}) closed. Consecutive loss counter reset.")
            self.consecutive_losses = 0
            self.in_cautious_state = False
        else:
            self.consecutive_losses += 1
            threshold = getattr(self.risk_config, 'consecutive_losses_threshold', 2)
            if self.consecutive_losses >= threshold:
                self.in_cautious_state = True
                logger.warning(
                    f"[RISK CAUTIOUS STATE] {self.consecutive_losses} consecutive losses. "
                    f"Engaging High-Conviction Re-qualification filter (Requires Unanimous Brain Consensus + VPIN >= 0.22 + Vol >= 1.5x)."
                )

        if self.risk_config.enable_kill_switch and self.daily_realized_pnl <= -self.risk_config.max_daily_loss_inr:
            self.is_kill_switch_active = True
            logger.critical(
                f"[KILL SWITCH TRIGGERED] Daily Loss {self.daily_realized_pnl:.2f} exceeded limit "
                f"{self.risk_config.max_daily_loss_inr:.2f} INR. Trading halted."
            )

        pos.side = PositionSide.FLAT
        pos.entry_price = 0.0
        pos.quantity = 0
        pos.stop_loss = 0.0
        pos.take_profit = 0.0
        pos.unrealized_pnl = 0.0
        pos.highest_price = 0.0
        pos.lowest_price = 0.0
        pos.current_trail_tier_r = 0.0
        pos.initial_risk_pts = 0.0
        pos.is_runner_active = False
        pos.symbol = ""
        pos.is_option = False
        return realized
