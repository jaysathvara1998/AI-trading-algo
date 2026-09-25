"""
Order Execution Engine for DhanHQ & Dhan-Tradehull API
Supports live routing for NSE_FNO, Super Bracket Orders, Telegram Alerting, and Paper Trading Simulation.
Supports automated PIN + TOTP login and Access Token authentication.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import logging
from typing import Dict, List, Optional
import pytz

try:
    from Dhan_Tradehull import Tradehull
except ImportError:
    Tradehull = None

try:
    from dhanhq import dhanhq
except ImportError:
    dhanhq = None

from .config import AppConfig, InstrumentMode
from .garch_engine import DirectionalSignal
from .risk_manager import PositionSide, TradeTarget, RiskManager
from .vpin import VPINResult, ToxicityRegime

logger = logging.getLogger("algo_vpin_dhan.execution")


class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"


class OrderStatus(Enum):
    SUBMITTED = "SUBMITTED"
    FILLED = "FILLED"
    REJECTED = "REJECTED"
    CANCELLED = "CANCELLED"


@dataclass
class OrderReceipt:
    """Execution receipt details"""
    order_id: str
    symbol: str
    action: str                 # "BUY" or "SELL"
    order_type: OrderType
    requested_price: float
    filled_price: float
    quantity: int
    status: OrderStatus
    timestamp: datetime
    slippage: float
    tag: str


class ExecutionEngine:
    """
    Handles live Dhan-Tradehull & DhanHQ API order placement, bracket orders,
    instant cancellation, and simulated Paper Trading execution.
    """

    def __init__(self, config: Optional[AppConfig] = None, risk_manager: Optional[RiskManager] = None):
        self.config = config or AppConfig()
        self.risk_manager = risk_manager or RiskManager(self.config.risk, self.config.market)
        self.tz = pytz.timezone("Asia/Kolkata")
        self.is_paper = self.config.dhan.paper_trading

        self.tradehull_client = None
        self.dhan_client = None

        if not self.is_paper and self.config.dhan.client_id and self.config.dhan.client_id != "YOUR_DHAN_CLIENT_ID":
            # Attempt Dhan-Tradehull initialization with TOTP or Access Token
            if Tradehull is not None:
                try:
                    if self.config.dhan.auth_mode == "pin_totp" and self.config.dhan.pin != "YOUR_DHAN_PIN":
                        self.tradehull_client = Tradehull(
                            ClientCode=self.config.dhan.client_id,
                            mode="pin_totp",
                            pin=self.config.dhan.pin,
                            totp_secret=self.config.dhan.totp_secret
                        )
                    else:
                        self.tradehull_client = Tradehull(
                            ClientCode=self.config.dhan.client_id,
                            token_id=self.config.dhan.access_token,
                            mode="access_token"
                        )
                    logger.info("Connected to Live Dhan-Tradehull execution gateway.")
                except Exception as e:
                    logger.warning(f"Could not connect to Dhan-Tradehull: {e}")

            # Fallback to direct dhanhq
            if self.tradehull_client is None and dhanhq is not None:
                try:
                    self.dhan_client = dhanhq(
                        client_id=self.config.dhan.client_id,
                        access_token=self.config.dhan.access_token
                    )
                    logger.info("Connected to Live DhanHQ API direct execution gateway.")
                except Exception as e:
                    logger.error(f"Failed to connect to DhanHQ: {e}. Switching to Paper mode.")
                    self.is_paper = True
        else:
            self.is_paper = True
            logger.info("Operating in PAPER TRADING simulation mode.")

        # Order history
        self.order_book: List[OrderReceipt] = []
        self._order_counter = 1000

    def notify(self, message: str):
        """
        Sends telegram alert via Tradehull if enabled.
        """
        logger.info(f"[NOTIFY] {message}")
        if self.config.dhan.enable_telegram and self.tradehull_client is not None:
            if self.config.dhan.telegram_bot_token and self.config.dhan.telegram_chat_id:
                try:
                    self.tradehull_client.send_telegram_alert(
                        message=message,
                        receiver_chat_id=self.config.dhan.telegram_chat_id,
                        bot_token=self.config.dhan.telegram_bot_token
                    )
                except Exception as e:
                    logger.error(f"Failed to send Telegram alert: {e}")

    def emergency_cancel_all(self):
        """
        Cancels all open orders across DhanHQ using Tradehull cancel_all_orders()
        """
        if self.tradehull_client is not None:
            try:
                res = self.tradehull_client.cancel_all_orders()
                logger.info(f"[EMERGENCY] Cancelled all pending orders: {res}")
            except Exception as e:
                logger.error(f"Failed to cancel all orders: {e}")

    def calculate_limit_price(
        self,
        current_price: float,
        action: DirectionalSignal,
        vpin_res: VPINResult
    ) -> float:
        """
        Calculates limit order price adjusted by VPIN toxicity entry bands.
        """
        spread_base = current_price * 0.0005
        adjusted_offset = spread_base * vpin_res.entry_band_multiplier

        if action == DirectionalSignal.BUY:
            return round(current_price - adjusted_offset, 2)
        else:
            return round(current_price + adjusted_offset, 2)

    def execute_entry(
        self,
        signal: DirectionalSignal,
        current_price: float,
        trade_target: TradeTarget,
        vpin_res: VPINResult,
        order_type: OrderType = OrderType.MARKET,
        target_symbol: Optional[str] = None
    ) -> Optional[OrderReceipt]:
        """
        Executes a new entry order (BUY or SELL).
        """
        can_open, reason = self.risk_manager.can_open_new_trade()
        if not can_open:
            logger.warning(f"Trade entry rejected by risk manager: {reason}")
            return None

        self._order_counter += 1
        order_id = f"DHAN_{self._order_counter}"
        action_str = "BUY" if signal == DirectionalSignal.BUY else "SELL"
        symbol = target_symbol or self.config.market.trading_symbol

        if order_type == OrderType.LIMIT:
            limit_price = self.calculate_limit_price(current_price, signal, vpin_res)
        else:
            limit_price = current_price

        # Paper Trading simulation path
        if self.is_paper:
            slippage_bps = 0.0003 if vpin_res.regime == ToxicityRegime.HIGH_TOXICITY else 0.0001
            slippage = current_price * slippage_bps
            fill_price = current_price + slippage if signal == DirectionalSignal.BUY else current_price - slippage
            fill_price = round(fill_price, 2)

            receipt = OrderReceipt(
                order_id=order_id,
                symbol=symbol,
                action=action_str,
                order_type=order_type,
                requested_price=limit_price,
                filled_price=fill_price,
                quantity=trade_target.total_quantity,
                status=OrderStatus.FILLED,
                timestamp=datetime.now(self.tz),
                slippage=round(abs(fill_price - current_price), 2),
                tag=f"VPIN_{vpin_res.regime.value}"
            )

            # Update Risk Manager Active Position
            self.risk_manager.current_position.side = PositionSide.LONG if signal == DirectionalSignal.BUY else PositionSide.SHORT
            self.risk_manager.current_position.entry_price = fill_price
            self.risk_manager.current_position.quantity = trade_target.total_quantity
            self.risk_manager.current_position.stop_loss = trade_target.stop_loss
            self.risk_manager.current_position.take_profit = trade_target.take_profit
            self.risk_manager.current_position.entry_time = receipt.timestamp
            self.risk_manager.current_position.entry_vpin = vpin_res.vpin
            self.risk_manager.current_position.entry_regime = vpin_res.regime
            self.risk_manager.current_position.symbol = symbol
            self.risk_manager.current_position.is_option = (self.config.market.instrument_mode == InstrumentMode.OPTIONS_BUYING)
            self.risk_manager.current_position.highest_price = fill_price
            self.risk_manager.current_position.lowest_price = fill_price

            self.order_book.append(receipt)
            total_cost = fill_price * trade_target.total_quantity
            msg = (
                f"[PAPER ENTRY] BUY {trade_target.total_quantity} qty {symbol} @ Premium ₹{fill_price:.2f} "
                f"(Total: ₹{total_cost:,.2f}) | SL: ₹{trade_target.stop_loss:.2f} | TP: ₹{trade_target.take_profit:.2f} | Regime: {vpin_res.regime.value}"
            )
            logger.info(msg)
            self.notify(msg)
            return receipt

        # 1. Try Live Dhan-Tradehull order placement
        if self.tradehull_client is not None:
            try:
                if self.config.risk.use_super_bracket_orders:
                    resp = self.tradehull_client.place_super_order(
                        tradingsymbol=symbol,
                        exchange=self.config.market.exchange_segment,
                        transaction_type=action_str,
                        quantity=trade_target.total_quantity,
                        order_type="MARKET" if order_type == OrderType.MARKET else "LIMIT",
                        trade_type="INTRADAY",
                        price=limit_price if order_type == OrderType.LIMIT else 0,
                        target_price=trade_target.take_profit,
                        stop_loss_price=trade_target.stop_loss,
                        trailing_jump=0.0
                    )
                else:
                    resp = self.tradehull_client.order_placement(
                        tradingsymbol=symbol,
                        exchange=self.config.market.exchange_segment,
                        quantity=trade_target.total_quantity,
                        price=limit_price if order_type == OrderType.LIMIT else 0,
                        trigger_price=0,
                        order_type="LIMIT" if order_type == OrderType.LIMIT else "MARKET",
                        transaction_type=action_str,
                        product_type="INTRADAY"
                    )

                logger.info(f"[TRADEHULL LIVE ENTRY] Response: {resp}")
                receipt = OrderReceipt(
                    order_id=str(resp),
                    symbol=symbol,
                    action=action_str,
                    order_type=order_type,
                    requested_price=limit_price,
                    filled_price=current_price,
                    quantity=trade_target.total_quantity,
                    status=OrderStatus.SUBMITTED,
                    timestamp=datetime.now(self.tz),
                    slippage=0.0,
                    tag=f"VPIN_{vpin_res.regime.value}"
                )
                self.order_book.append(receipt)
                self.notify(f"[LIVE ENTRY] {action_str} {trade_target.total_quantity} qty {symbol} submitted.")
                return receipt
            except Exception as e:
                logger.error(f"Error in Tradehull order placement: {e}")

        # 2. Try raw DhanHQ client fallback
        if self.dhan_client is not None:
            try:
                dhan_txn_type = self.dhan_client.BUY if signal == DirectionalSignal.BUY else self.dhan_client.SELL
                dhan_order_type = self.dhan_client.MARKET if order_type == OrderType.MARKET else self.dhan_client.LIMIT

                resp = self.dhan_client.place_order(
                    security_id=self.config.market.security_id,
                    exchange_segment=self.config.market.exchange_segment,
                    transaction_type=dhan_txn_type,
                    quantity=trade_target.total_quantity,
                    order_type=dhan_order_type,
                    product_type=self.dhan_client.INTRA,
                    price=limit_price if order_type == OrderType.LIMIT else 0.0,
                    trigger_price=0.0,
                    validity="DAY"
                )

                if resp and resp.get("status") == "success":
                    live_order_id = resp.get("data", {}).get("orderId", order_id)
                    receipt = OrderReceipt(
                        order_id=live_order_id,
                        symbol=symbol,
                        action=action_str,
                        order_type=order_type,
                        requested_price=limit_price,
                        filled_price=current_price,
                        quantity=trade_target.total_quantity,
                        status=OrderStatus.SUBMITTED,
                        timestamp=datetime.now(self.tz),
                        slippage=0.0,
                        tag=f"VPIN_{vpin_res.regime.value}"
                    )
                    self.order_book.append(receipt)
                    self.notify(f"[LIVE ENTRY] {action_str} {trade_target.total_quantity} qty submitted (ID: {live_order_id})")
                    return receipt
            except Exception as e:
                logger.error(f"Exception during DhanHQ live order execution: {e}")

        return None

    def execute_exit(
        self,
        current_price: float,
        reason: str
    ) -> Optional[OrderReceipt]:
        """
        Closes existing open position (Square-off).
        """
        pos = self.risk_manager.current_position
        if pos.side == PositionSide.FLAT:
            return None

        self._order_counter += 1
        order_id = f"DHAN_EXIT_{self._order_counter}"
        exit_action_str = "SELL" if pos.side == PositionSide.LONG else "BUY"

        if self.is_paper:
            slippage = current_price * 0.0001
            fill_price = current_price - slippage if pos.side == PositionSide.LONG else current_price + slippage
            fill_price = round(fill_price, 2)

            pnl = self.risk_manager.record_trade_close(fill_price)

            receipt = OrderReceipt(
                order_id=order_id,
                symbol=self.config.market.trading_symbol,
                action=exit_action_str,
                order_type=OrderType.MARKET,
                requested_price=current_price,
                filled_price=fill_price,
                quantity=pos.quantity,
                status=OrderStatus.FILLED,
                timestamp=datetime.now(self.tz),
                slippage=round(abs(fill_price - current_price), 2),
                tag=reason
            )
            self.order_book.append(receipt)
            msg = (
                f"[PAPER EXIT] {exit_action_str} {pos.quantity} qty @ {fill_price} | "
                f"Reason: {reason} | Realized PnL: {pnl:+.2f} | Total Daily PnL: {self.risk_manager.daily_realized_pnl:+.2f}"
            )
            logger.info(msg)
            self.notify(msg)

            if self.risk_manager.is_kill_switch_active:
                self.emergency_cancel_all()
            return receipt

        # Live exit via Tradehull
        if self.tradehull_client is not None:
            try:
                self.tradehull_client.order_placement(
                    tradingsymbol=self.config.market.trading_symbol,
                    exchange=self.config.market.exchange_segment,
                    quantity=pos.quantity,
                    price=0,
                    trigger_price=0,
                    order_type="MARKET",
                    transaction_type=exit_action_str,
                    product_type="INTRADAY"
                )
                self.risk_manager.record_trade_close(current_price)
                self.notify(f"[LIVE EXIT] {exit_action_str} {pos.quantity} qty executed. Reason: {reason}")
                if self.risk_manager.is_kill_switch_active:
                    self.emergency_cancel_all()
                return None
            except Exception as e:
                logger.error(f"Failed live exit on Tradehull: {e}")

        # Live exit via DhanHQ direct
        if self.dhan_client is not None:
            try:
                dhan_txn_type = self.dhan_client.SELL if pos.side == PositionSide.LONG else self.dhan_client.BUY
                self.dhan_client.place_order(
                    security_id=self.config.market.security_id,
                    exchange_segment=self.config.market.exchange_segment,
                    transaction_type=dhan_txn_type,
                    quantity=pos.quantity,
                    order_type=self.dhan_client.MARKET,
                    product_type=self.dhan_client.INTRA,
                    price=0.0,
                    trigger_price=0.0,
                    validity="DAY"
                )
                self.risk_manager.record_trade_close(current_price)
                self.notify(f"[LIVE EXIT] {exit_action_str} {pos.quantity} qty submitted. Reason: {reason}")
            except Exception as e:
                logger.error(f"Failed to execute live exit on DhanHQ: {e}")

        return None
