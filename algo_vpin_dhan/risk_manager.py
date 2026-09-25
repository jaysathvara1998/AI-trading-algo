"""
Risk Management Engine
Handles position sizing throttles, VPIN toxicity adjustments,
volatility-based dynamic Stop-Loss/Take-Profit, and daily drawdown kill-switch.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional, Tuple
import pytz

from .config import RiskConfig, MarketConfig
from .vpin import ToxicityRegime, VPINResult
from .garch_engine import GARCHForecastResult, DirectionalSignal


class PositionSide(Enum):
    FLAT = "FLAT"
    LONG = "LONG"
    SHORT = "SHORT"


@dataclass
class TradeTarget:
    """Calculated entry, stop-loss, and take-profit prices"""
    entry_price: float
    stop_loss: float
    take_profit: float
    quantity_lots: int
    total_quantity: int
    risk_reward_ratio: float


@dataclass
class ActivePosition:
    """Current open position tracking"""
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


class RiskManager:
    """
    Manages capital allocation, position sizing based on VPIN toxicity,
    volatility-based stop loss adjustments, and daily loss kill-switch.
    """

    def __init__(self, risk_config: Optional[RiskConfig] = None, market_config: Optional[MarketConfig] = None):
        self.risk_config = risk_config or RiskConfig()
        self.market_config = market_config or MarketConfig()
        self.tz = pytz.timezone("Asia/Kolkata")

        self.daily_realized_pnl = 0.0
        self.is_kill_switch_active = False
        self.kill_switch_reason = ""
        self.current_position = ActivePosition(
            side=PositionSide.FLAT,
            entry_price=0.0,
            quantity=0,
            stop_loss=0.0,
            take_profit=0.0,
            entry_time=datetime.now(self.tz),
            entry_vpin=0.5,
            entry_regime=ToxicityRegime.NORMAL_TOXICITY
        )

    def calculate_position_size(self, vpin_result: VPINResult) -> int:
        """
        Determines position size in lots based on VPIN toxicity:
        - High toxicity (VPIN > delta_2): Halve position size (0.5x)
        - Low toxicity (VPIN < delta_3): Full position size (1.0x)
        - Normal toxicity: Standard size (0.75x)
        """
        base_lots = self.risk_config.max_position_lots
        size_mult = vpin_result.size_multiplier

        lots = int(round(base_lots * size_mult))
        return max(1, min(lots, self.risk_config.max_position_lots))

    def compute_trade_targets(
        self,
        signal: DirectionalSignal,
        current_price: float,
        garch_res: GARCHForecastResult,
        vpin_res: VPINResult,
        is_option: bool = False,
        option_premium: Optional[float] = None
    ) -> Optional[TradeTarget]:
        """
        Computes dynamic SL and TP based on GARCH conditional volatility sqrt(h_{t+1}).
        If is_option=True, targets are calibrated to the actual Option Premium.
        """
        if signal == DirectionalSignal.HOLD or current_price <= 0:
            return None

        lots = self.calculate_position_size(vpin_res)
        total_qty = lots * self.market_config.lot_size

        # Options Premium Target Calculation
        if is_option:
            premium = option_premium if option_premium and option_premium > 0 else max(10.0, current_price * garch_res.sigma_next * 0.4)
            # Dynamic SL on option premium: 20% to 30% stop based on volatility
            sl_pct = min(0.35, max(0.15, garch_res.sigma_next * 100 * self.risk_config.sl_garch_multiplier * 0.1))
            tp_pct = min(0.80, max(0.30, garch_res.sigma_next * 100 * self.risk_config.tp_garch_multiplier * 0.1))

            sl = max(1.0, premium * (1.0 - sl_pct))
            tp = premium * (1.0 + tp_pct)
            risk_dist = abs(premium - sl)
            reward_dist = abs(tp - premium)
            rr_ratio = reward_dist / risk_dist if risk_dist > 0 else 2.0

            return TradeTarget(
                entry_price=round(premium, 2),
                stop_loss=round(sl, 2),
                take_profit=round(tp, 2),
                quantity_lots=lots,
                total_quantity=total_qty,
                risk_reward_ratio=round(rr_ratio, 2)
            )

        # Underlying / Futures Target Calculation
        sigma = garch_res.sigma_next
        vol_band = max(current_price * 0.002, sigma * current_price * self.risk_config.sl_garch_multiplier)
        tp_band = max(current_price * 0.003, sigma * current_price * self.risk_config.tp_garch_multiplier)

        hard_cap = current_price * self.risk_config.max_trade_loss_pct
        vol_band = min(vol_band, hard_cap)

        if signal == DirectionalSignal.BUY:
            sl = current_price - vol_band
            tp = current_price + tp_band
        else:  # SELL
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
            risk_reward_ratio=round(rr_ratio, 2)
        )

    def check_exit_conditions(
        self,
        current_price: float,
        option_premium: Optional[float] = None,
        current_time: Optional[datetime] = None,
        reversal_signal: Optional[DirectionalSignal] = None
    ) -> Tuple[bool, str]:
        """
        Evaluates active position against:
        1. Stop-Loss & Take-Profit limits
        2. Dynamic Trailing Stop-Loss (locks in profit as trade moves favorably)
        3. Emergency Signal Reversal (exits immediately if model confirms opposite trend)
        4. Intraday Cutoff (15:24 IST)
        """
        if self.current_position.side == PositionSide.FLAT:
            return False, ""

        # 1. Check Mandatory Intraday Square-off Time (15:24 IST)
        now_ist = current_time or datetime.now(self.tz)
        if (now_ist.hour > self.market_config.square_off_hour or
            (now_ist.hour == self.market_config.square_off_hour and
             now_ist.minute >= self.market_config.square_off_minute)):
            return True, "INTRADAY_SQUARE_OFF"

        pos = self.current_position
        eval_price = option_premium if (pos.is_option and option_premium and option_premium > 0) else current_price

        # Update peak prices for trailing metrics
        pos.highest_price = max(pos.highest_price, eval_price)
        pos.lowest_price = min(pos.lowest_price if pos.lowest_price > 0 else eval_price, eval_price)

        # 2. Emergency Signal Reversal Exit
        if reversal_signal is not None and reversal_signal != DirectionalSignal.HOLD:
            if (pos.side == PositionSide.LONG and reversal_signal == DirectionalSignal.SELL) or \
               (pos.is_option and "CALL" in pos.symbol and reversal_signal == DirectionalSignal.SELL):
                return True, f"EMERGENCY_REVERSAL_EXIT: Model confirmed Bearish SELL signal"
            elif (pos.side == PositionSide.SHORT and reversal_signal == DirectionalSignal.BUY) or \
                 (pos.is_option and "PUT" in pos.symbol and reversal_signal == DirectionalSignal.BUY):
                return True, f"EMERGENCY_REVERSAL_EXIT: Model confirmed Bullish BUY signal"

        # 3. Dynamic Trailing Stop-Loss for Options & Futures
        if pos.is_option or pos.side == PositionSide.LONG:
            pos.unrealized_pnl = (eval_price - pos.entry_price) * pos.quantity
            profit_pct = (eval_price - pos.entry_price) / pos.entry_price if pos.entry_price > 0 else 0.0

            # If trade hits +20% gain, trail SL to Breakeven (entry price)
            if profit_pct >= 0.20 and pos.stop_loss < pos.entry_price:
                pos.stop_loss = round(pos.entry_price, 2)
                logger.info(f"[TRAILING SL] Trade up +{profit_pct*100:.1f}%. Trailing SL moved to Breakeven (INR {pos.stop_loss:.2f})")

            # If trade hits +40% gain, trail SL to lock in +20% profit
            elif profit_pct >= 0.40 and pos.stop_loss < (pos.entry_price * 1.20):
                pos.stop_loss = round(pos.entry_price * 1.20, 2)
                logger.info(f"[TRAILING SL] Trade up +{profit_pct*100:.1f}%. Trailing SL moved to +20% profit (INR {pos.stop_loss:.2f})")

            if eval_price <= pos.stop_loss:
                return True, f"STOP_LOSS_TRIGGERED (Price INR {eval_price:.2f} <= SL INR {pos.stop_loss:.2f})"
            if eval_price >= pos.take_profit:
                return True, f"TAKE_PROFIT_TRIGGERED (Price INR {eval_price:.2f} >= TP INR {pos.take_profit:.2f})"

        elif pos.side == PositionSide.SHORT:
            pos.unrealized_pnl = (pos.entry_price - eval_price) * pos.quantity
            profit_pct = (pos.entry_price - eval_price) / pos.entry_price if pos.entry_price > 0 else 0.0

            if profit_pct >= 0.20 and pos.stop_loss > pos.entry_price:
                pos.stop_loss = round(pos.entry_price, 2)
            elif profit_pct >= 0.40 and pos.stop_loss > (pos.entry_price * 0.80):
                pos.stop_loss = round(pos.entry_price * 0.80, 2)

            if eval_price >= pos.stop_loss:
                return True, f"STOP_LOSS_TRIGGERED (Price INR {eval_price:.2f} >= SL INR {pos.stop_loss:.2f})"
            if eval_price <= pos.take_profit:
                return True, f"TAKE_PROFIT_TRIGGERED (Price INR {eval_price:.2f} <= TP INR {pos.take_profit:.2f})"

        return False, ""

    def record_trade_close(self, exit_price: float) -> float:
        """
        Closes current active position, records realized P&L, and evaluates kill-switch.
        """
        if self.current_position.side == PositionSide.FLAT:
            return 0.0

        pos = self.current_position
        if pos.side == PositionSide.LONG:
            pnl = (exit_price - pos.entry_price) * pos.quantity
        else:
            pnl = (pos.entry_price - exit_price) * pos.quantity

        self.daily_realized_pnl += pnl

        # Check Kill Switch
        if self.daily_realized_pnl <= -self.risk_config.max_daily_loss:
            self.is_kill_switch_active = True
            self.kill_switch_reason = f"MAX_DAILY_LOSS_EXCEEDED (Loss: {self.daily_realized_pnl:.2f} <= Limit: -{self.risk_config.max_daily_loss:.2f})"

        # Reset active position
        self.current_position = ActivePosition(
            side=PositionSide.FLAT,
            entry_price=0.0,
            quantity=0,
            stop_loss=0.0,
            take_profit=0.0,
            entry_time=datetime.now(self.tz),
            entry_vpin=0.5,
            entry_regime=ToxicityRegime.NORMAL_TOXICITY
        )

        return pnl

    def can_open_new_trade(self, current_time: Optional[datetime] = None) -> Tuple[bool, str]:
        """
        Validates whether system is permitted to enter a new position.
        """
        if self.is_kill_switch_active:
            return False, f"KILL_SWITCH_ACTIVE: {self.kill_switch_reason}"

        if self.current_position.side != PositionSide.FLAT:
            return False, "POSITION_ALREADY_OPEN"

        now_ist = current_time or datetime.now(self.tz)

        # Before market open 09:15
        if (now_ist.hour < self.market_config.market_open_hour or
            (now_ist.hour == self.market_config.market_open_hour and
             now_ist.minute < self.market_config.market_open_minute)):
            return False, "BEFORE_MARKET_OPEN (09:15 IST)"

        # Past square-off time 15:15
        if (now_ist.hour > self.market_config.square_off_hour or
            (now_ist.hour == self.market_config.square_off_hour and
             now_ist.minute >= self.market_config.square_off_minute)):
            return False, "PAST_ENTRY_CUTOFF (15:15 IST)"

        return True, "OK"
