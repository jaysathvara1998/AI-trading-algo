"""
Execution Router & Paper Simulator for Algo VPIN v2.0
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional, List
import pytz
import logging

from .config import AppConfig, MarketConfig, DhanAPIConfig, RiskConfig, InstrumentMode
from .garch_engine import DirectionalSignal
from .risk_manager import RiskManager, TradeTarget, PositionSide
from .vpin import VPINResult
from .telegram_notifier import notify_trade_entry, notify_trade_exit

logger = logging.getLogger("algo_vpin_v2.execution")


class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"


class OrderStatus(Enum):
    FILLED = "FILLED"
    REJECTED = "REJECTED"
    CANCELLED = "CANCELLED"


@dataclass
class OrderReceipt:
    order_id: str
    symbol: str
    action: str
    quantity: int
    price: float
    order_type: OrderType
    status: OrderStatus
    timestamp: datetime
    stop_loss: float
    take_profit: float
    realized_pnl: float = 0.0
    rejection_reason: str = ""


class OrderExecutionRouter:
    def __init__(self, config: Optional[AppConfig] = None, risk_manager: Optional[RiskManager] = None, tradehull_client=None, track_label: str = "Track 1: Dual-Brain", data_feed=None):
        self.config = config or AppConfig()
        self.dhan_cfg: DhanAPIConfig = self.config.dhan
        self.market_cfg: MarketConfig = self.config.market
        self.risk_cfg: RiskConfig = self.config.risk
        self.risk_manager = risk_manager or RiskManager(self.risk_cfg, self.market_cfg)
        self.tradehull_client = tradehull_client
        self.data_feed = data_feed  # Live data feed for fresh LTP fetch at exit time
        self.track_label = track_label
        self.tz = pytz.timezone("Asia/Kolkata")
        
        self.is_paper = self.dhan_cfg.paper_trading
        self.order_book: List[OrderReceipt] = []
        self.order_counter: int = 1000

        if self.is_paper:
            logger.info(f"[{self.track_label}] Operating in PAPER TRADING simulation mode.")
        else:
            logger.warning(f"[{self.track_label}] Operating in LIVE DhanHQ Order Execution Mode.")

    def notify(self, message: str):
        logger.info(f"[NOTIFY] {message}")

    def get_available_capital(self) -> float:
        """Uses simulation capital (₹50,000) in Paper Trading mode, and queries live Dhan balance in Live mode."""
        if self.is_paper:
            return self.risk_cfg.capital_allocation

        if self.tradehull_client is not None:
            try:
                fund_limit = self.tradehull_client.get_balance()
                if fund_limit is not None and isinstance(fund_limit, (int, float)) and fund_limit > 0:
                    return float(fund_limit)
                elif isinstance(fund_limit, dict):
                    avail = fund_limit.get("availabelBalance") or fund_limit.get("availableBalance") or fund_limit.get("cashBalance")
                    if avail:
                        return float(avail)
            except Exception as e:
                logger.debug(f"Could not fetch balance from Tradehull: {e}")
        return self.risk_cfg.capital_allocation

    def execute_entry(
        self,
        signal: DirectionalSignal,
        current_price: float,
        trade_target: TradeTarget,
        vpin_res: VPINResult,
        order_type: OrderType = OrderType.MARKET,
        target_symbol: Optional[str] = None,
        is_unanimous: bool = False,
        volume_ratio: float = 1.0
    ) -> Optional[OrderReceipt]:
        is_approved, reason = self.risk_manager.validate_new_entry(
            signal=signal,
            vpin_res=vpin_res,
            is_unanimous=is_unanimous,
            volume_ratio=volume_ratio
        )
        if not is_approved:
            logger.warning(f"Trade entry rejected by risk manager: {reason}")
            return None

        symbol = target_symbol or self.market_cfg.trading_symbol
        action_str = "BUY"
        slippage = current_price * self.risk_cfg.slippage_pct
        fill_price = round(current_price + slippage, 2)
        
        order_id = f"V2_PAPER_{self.order_counter}"
        self.order_counter += 1

        # Live DhanHQ Order Dispatch
        if not self.is_paper:
            if self.tradehull_client is not None:
                try:
                    exch = self.market_cfg.exchange_segment
                    if "NIFTY" in symbol or "BANKNIFTY" in symbol:
                        exch = "NSE_FNO"
                    elif "SENSEX" in symbol:
                        exch = "BSE_FNO"

                    resp = self.tradehull_client.order_placement(
                        tradingsymbol=symbol,
                        exchange=exch,
                        quantity=trade_target.total_quantity,
                        price=0,
                        trigger_price=0,
                        order_type="MARKET",
                        transaction_type=action_str,
                        product_type="INTRADAY"
                    )
                    logger.info(f"[{self.track_label}] Live DhanHQ Order Executed: {resp}")
                    order_id = f"DHAN_{resp}"
                except Exception as e:
                    logger.error(f"[{self.track_label}] Live order placement error: {e}")
                    order_id = f"DHAN_ERR_{self.order_counter}"
            else:
                logger.error(f"[{self.track_label}] Live trading mode active but Tradehull client is None! Order not dispatched.")
                order_id = f"NO_CLIENT_{self.order_counter}"

        receipt = OrderReceipt(
            order_id=order_id,
            symbol=symbol,
            action=action_str,
            quantity=trade_target.total_quantity,
            price=fill_price,
            order_type=order_type,
            status=OrderStatus.FILLED if self.is_paper else OrderStatus.FILLED,
            timestamp=datetime.now(self.tz),
            stop_loss=trade_target.stop_loss,
            take_profit=trade_target.take_profit
        )

        self.risk_manager.current_position.side = PositionSide.LONG if signal == DirectionalSignal.BUY else PositionSide.SHORT
        self.risk_manager.current_position.entry_price = fill_price
        self.risk_manager.current_position.quantity = trade_target.total_quantity
        self.risk_manager.current_position.stop_loss = trade_target.stop_loss
        self.risk_manager.current_position.take_profit = trade_target.take_profit
        self.risk_manager.current_position.entry_time = receipt.timestamp
        self.risk_manager.current_position.entry_vpin = vpin_res.vpin
        self.risk_manager.current_position.entry_regime = vpin_res.regime
        self.risk_manager.current_position.symbol = symbol
        self.risk_manager.current_position.is_option = (self.market_cfg.instrument_mode == InstrumentMode.OPTIONS_BUYING)
        self.risk_manager.current_position.highest_price = fill_price
        self.risk_manager.current_position.lowest_price = fill_price
        self.risk_manager.current_position.strategy_mode = getattr(trade_target, 'strategy_mode', None)
        self.risk_manager.current_position.initial_risk_pts = getattr(trade_target, 'initial_risk_pts', 0.0)
        self.risk_manager.current_position.current_trail_tier_r = 0.0
        self.risk_manager.current_position.entry_spot_price = getattr(trade_target, 'entry_spot_price', 0.0)
        self.risk_manager.current_position.spot_stop_loss = getattr(trade_target, 'spot_stop_loss', 0.0)
        self.risk_manager.current_position.hard_disaster_sl = getattr(trade_target, 'hard_disaster_sl', 0.0)

        self.order_book.append(receipt)
        total_cost = fill_price * trade_target.total_quantity
        mode_label = "SCALPER [1:1.5 R:R | 0.5R Trail]" if getattr(trade_target, 'strategy_mode', None) and trade_target.strategy_mode.value == "SCALPER" else "INSTITUTIONAL SWING RUNNER"
        spot_sl_str = f" | Spot SL: {trade_target.spot_stop_loss:.2f}" if getattr(trade_target, 'spot_stop_loss', 0.0) > 0 else ""

        # Clean, Universal ASCII Trade Entry Banner
        entry_banner = (
            f"\n"
            f"+================================================================================+\n"
            f"| >>> [TRADE ENTRY TRIGGERED] -> BUY {symbol} ({trade_target.total_quantity} Qty)\n"
            f"+--------------------------------------------------------------------------------+\n"
            f"| * Order ID       : {order_id}\n"
            f"| * Strategy Mode  : {mode_label}\n"
            f"| * Entry Premium  : INR {fill_price:.2f} (Total Capital: INR {total_cost:,.2f})\n"
            f"| * Stop-Loss (SL) : INR {trade_target.stop_loss:.2f} (Risk: {trade_target.initial_risk_pts:.1f} pts{spot_sl_str})\n"
            f"| * Target (TP)    : INR {trade_target.take_profit:.2f} (R:R = 1:{trade_target.risk_reward_ratio:.1f})\n"
            f"| * Market Regime  : {vpin_res.regime.value} (VPIN: {vpin_res.vpin:.3f})\n"
            f"+================================================================================+"
        )
        logger.info(entry_banner)

        try:
            notify_trade_entry(
                symbol=symbol,
                action=action_str,
                quantity=trade_target.total_quantity,
                entry_price=fill_price,
                sl=trade_target.stop_loss,
                tp=trade_target.take_profit,
                vpin=vpin_res.vpin,
                regime=vpin_res.regime.value,
                total_capital=total_cost,
                track_name=self.track_label
            )
        except Exception as e:
            logger.debug(f"Telegram entry alert: {e}")

        try:
            import winsound
            winsound.Beep(1200, 400)
            winsound.Beep(1500, 400)
        except Exception:
            pass
        return receipt

    def execute_exit(self, current_price: float, reason: str = "EXIT_SIGNAL") -> Optional[OrderReceipt]:
        pos = self.risk_manager.current_position
        if pos.side == PositionSide.FLAT:
            return None

        # Capture snapshot before record_trade_close resets the active position
        pos_symbol = pos.symbol or self.market_cfg.trading_symbol
        pos_entry_price = pos.entry_price
        pos_entry_time = pos.entry_time
        pos_side_val = pos.side.value
        pos_qty = pos.quantity
        is_long_or_option = pos.is_option or pos.side == PositionSide.LONG

        # --- FRESH LTP FETCH: Bypass the 2s cache at exit time for accurate fill price ---
        fresh_exit_price = current_price  # Default to the passed-in price
        if self.data_feed is not None and pos_symbol:
            try:
                fresh_ltp = self.data_feed.fetch_fresh_ltp(pos_symbol)
                if fresh_ltp > 0:
                    fresh_exit_price = fresh_ltp
                    if abs(fresh_ltp - current_price) > 0.5:
                        logger.info(
                            f"[{self.track_label}] EXIT FRESH LTP: ₹{fresh_ltp:.2f} "
                            f"(stale cached was ₹{current_price:.2f}, diff {fresh_ltp - current_price:+.2f})"
                        )
            except Exception as e:
                logger.debug(f"[{self.track_label}] Fresh LTP fetch at exit failed, using cached: {e}")

        # Slippage: LONG/option exit = SELL order = worse fill (subtract slippage)
        #           SHORT exit = BUY order = worse fill (add slippage)
        slippage = fresh_exit_price * self.risk_cfg.slippage_pct
        if is_long_or_option:
            fill_price = round(max(0.05, fresh_exit_price - slippage), 2)
        else:
            fill_price = round(fresh_exit_price + slippage, 2)

        logger.info(
            f"[{self.track_label}] EXIT FILL: {pos_symbol} | "
            f"Fresh LTP=₹{fresh_exit_price:.2f} | Slippage=₹{slippage:.2f} | Fill=₹{fill_price:.2f}"
        )
        realized_pnl = self.risk_manager.record_trade_close(fill_price)
        pnl_pct = round(((fill_price - pos_entry_price) / pos_entry_price) * 100, 2) if pos_entry_price > 0 else 0.0

        order_id = f"V2_PAPER_EXIT_{self.order_counter}"
        self.order_counter += 1

        # Live DhanHQ Exit Dispatch
        if not self.is_paper:
            if self.tradehull_client is not None:
                try:
                    exch = self.market_cfg.exchange_segment
                    if "NIFTY" in pos_symbol or "BANKNIFTY" in pos_symbol:
                        exch = "NSE_FNO"
                    elif "SENSEX" in pos_symbol:
                        exch = "BSE_FNO"

                    exit_resp = self.tradehull_client.order_placement(
                        tradingsymbol=pos_symbol,
                        exchange=exch,
                        quantity=pos_qty,
                        price=0,
                        trigger_price=0,
                        order_type="MARKET",
                        transaction_type="SELL",
                        product_type="INTRADAY"
                    )
                    logger.info(f"[{self.track_label}] Live DhanHQ Exit Order Executed: {exit_resp}")
                    order_id = f"DHAN_EXIT_{exit_resp}"
                except Exception as e:
                    logger.error(f"[{self.track_label}] Live exit order placement error: {e}")
                    order_id = f"DHAN_EXIT_ERR_{self.order_counter}"
            else:
                logger.error(f"[{self.track_label}] Live exit requested but Tradehull client is None!")
                order_id = f"NO_CLIENT_EXIT_{self.order_counter}"

        receipt = OrderReceipt(
            order_id=order_id,
            symbol=pos_symbol,
            action="SELL",
            quantity=pos_qty,
            price=fill_price,
            order_type=OrderType.MARKET,
            status=OrderStatus.FILLED,
            timestamp=datetime.now(self.tz),
            stop_loss=pos.stop_loss,
            take_profit=pos.take_profit,
            realized_pnl=realized_pnl,
            rejection_reason=reason
        )
        self.order_book.append(receipt)

        # Clean, Universal ASCII Trade Exit Banner
        pnl_tag = "[PROFIT]" if realized_pnl >= 0 else "[LOSS/SL]"
        exit_banner = (
            f"\n"
            f"+================================================================================+\n"
            f"| <<< [{self.track_label}] -> CLOSED {pos_symbol} {pnl_tag}\n"
            f"+--------------------------------------------------------------------------------+\n"
            f"| * Exit Reason    : {reason}\n"
            f"| * Exit Price     : INR {fill_price:.2f}\n"
            f"| * Realized P&L   : {realized_pnl:+.2f} INR\n"
            f"| * Total Day P&L  : {self.risk_manager.daily_realized_pnl:+.2f} INR\n"
            f"+================================================================================+"
        )
        logger.info(exit_banner)
        try:
            notify_trade_exit(
                symbol=pos_symbol,
                exit_price=fill_price,
                pnl_inr=realized_pnl,
                pnl_pct=pnl_pct,
                reason=reason,
                day_pnl=self.risk_manager.daily_realized_pnl,
                track_name=self.track_label
            )
        except Exception as e:
            logger.debug(f"Telegram exit alert: {e}")
        try:
            import winsound
            winsound.Beep(800, 300)
        except Exception:
            pass

        # Permanent CSV Trade Logging for 1-week / 1-month analysis
        try:
            from pathlib import Path
            import pandas as pd
            log_dir = Path(__file__).resolve().parent / "data"
            log_dir.mkdir(parents=True, exist_ok=True)
            trade_csv = log_dir / "trades_history.csv"
            trade_record = {
                "exit_timestamp": str(receipt.timestamp),
                "entry_time": str(pos_entry_time),
                "track": self.track_label,
                "symbol": pos_symbol,
                "side": pos_side_val,
                "quantity": pos_qty,
                "entry_price": pos_entry_price,
                "exit_price": fill_price,
                "pnl_inr": realized_pnl,
                "pnl_pct": pnl_pct,
                "exit_reason": reason,
                "cumulative_daily_pnl": self.risk_manager.daily_realized_pnl
            }
            df_trade = pd.DataFrame([trade_record])
            df_trade.to_csv(trade_csv, mode="a", header=not trade_csv.exists(), index=False)
        except Exception as e:
            logger.debug(f"Error saving trade to CSV: {e}")

        return receipt
