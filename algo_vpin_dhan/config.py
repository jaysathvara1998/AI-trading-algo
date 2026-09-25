"""
Trading System Configuration & Global Constants
Based on Fang & Feng (2019) Market Microstructure Architecture
Static credentials & advanced Dhan-Tradehull TOTP integration.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import logging

logger = logging.getLogger("algo_vpin_dhan.config")

import os
from pathlib import Path

# ==============================================================================
# CREDENTIALS & ACCOUNT CONFIGURATION (loaded from environment / .env file)
# ==============================================================================
# Secrets are never stored in source. Put them in a `.env` file at the project
# root (see `.env.example`) or export them as environment variables.

def _load_dotenv() -> None:
    """Minimal .env loader (no external dependency). Never overrides existing env vars."""
    for directory in Path(__file__).resolve().parents:
        env_file = directory / ".env"
        if env_file.is_file():
            for raw in env_file.read_text().splitlines():
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
            return


_load_dotenv()


def _env_bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


# Authentication Modes: "pin_totp" (Fully automated), "access_token" (Manual token), or "api_key"
STATIC_AUTH_MODE = os.getenv("DHAN_AUTH_MODE", "pin_totp")

# 1. Primary TOTP-Based Login (Automated daily login via Dhan-Tradehull)
STATIC_DHAN_CLIENT_ID = os.getenv("DHAN_CLIENT_ID", "")
STATIC_DHAN_PIN = os.getenv("DHAN_PIN", "")
STATIC_DHAN_TOTP_SECRET = os.getenv("DHAN_TOTP_SECRET", "")

# 2. Access Token Fallback (If using manual daily access token)
STATIC_DHAN_ACCESS_TOKEN = os.getenv("DHAN_ACCESS_TOKEN", "")
STATIC_IP = os.getenv("STATIC_IP", "")                     # Optional static IP

# Trading Engine Mode
STATIC_PAPER_TRADING = _env_bool("PAPER_TRADING", True)    # True for simulation, False for Live orders
STATIC_LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Telegram Notifications (Optional)
STATIC_ENABLE_TELEGRAM = _env_bool("ENABLE_TELEGRAM", False)
STATIC_TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
STATIC_TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

if not STATIC_DHAN_CLIENT_ID:
    logger.warning("DHAN_CLIENT_ID is not set. Copy .env.example to .env and fill in your credentials.")


class InstrumentMode(Enum):
    OPTIONS_BUYING = "OPTIONS_BUYING"   # Trades ATM Call on BUY and ATM Put on SELL (Default)
    FUTURES = "FUTURES"                 # Disabled for now


@dataclass
class MarketConfig:
    """NSE/BSE Market Timing & Options Instrument Specifications"""
    symbol: str = "NIFTY"                           # "NIFTY" or "SENSEX"
    underlying: str = "NIFTY"                       # "NIFTY" or "SENSEX"
    trading_symbol: str = "NIFTY"                   # Auto-resolved to active Option contract
    exchange_segment: str = "NFO"                   # "NFO" for Nifty, "BFO" for Sensex
    instrument_type: str = "OPTIDX"
    instrument_mode: InstrumentMode = InstrumentMode.OPTIONS_BUYING  # Options buying active
    option_strike_mode: str = "ATM"                 # "ATM", "ITM", or "OTM"
    lot_size: int = 65                              # 65 for Nifty, 20 for Sensex
    step_size: int = 50                             # 50 for Nifty, 100 for Sensex
    
    # Market Trading Hours (IST: 09:15 to 15:30)
    market_open_hour: int = 9
    market_open_minute: int = 15
    market_close_hour: int = 15
    market_close_minute: int = 30
    
    # Intraday Square-off & Order Cutoff
    square_off_hour: int = 15
    square_off_minute: int = 24
    warmup_bars_required: int = 100

    def set_symbol(self, symbol_name: str):
        """Switches underlying between NIFTY and SENSEX"""
        self.underlying = symbol_name.upper()
        self.symbol = self.underlying
        if self.underlying == "SENSEX":
            self.exchange_segment = "BFO"
            self.lot_size = 20
            self.step_size = 100
        else:
            self.underlying = "NIFTY"
            self.symbol = "NIFTY"
            self.exchange_segment = "NFO"
            self.lot_size = 65
            self.step_size = 50

    def sync_instrument_and_lot_size(self, tradehull_client) -> int:
        """
        Dynamically synchronizes current active contract & lot size using Dhan-Tradehull
        """
        if self.underlying == "SENSEX":
            self.exchange_segment = "BFO"
            self.lot_size = 20
            self.step_size = 100
        else:
            self.exchange_segment = "NFO"
            self.lot_size = 65
            self.step_size = 50

        if tradehull_client is not None:
            try:
                # Query nearest option strike to get current lot size
                ce_sym, pe_sym, strike = tradehull_client.ATM_Strike_Selection(self.underlying, 0)
                if ce_sym:
                    fetched_size = tradehull_client.get_lot_size(ce_sym)
                    if fetched_size > 0:
                        self.lot_size = int(fetched_size)
                        logger.info(f"[Tradehull] Synchronized {self.underlying} Options lot size: {self.lot_size}")
                        return self.lot_size
            except Exception as e:
                logger.warning(f"Tradehull lot sync for {self.underlying} options: {e}. Using default: {self.lot_size}")
        return self.lot_size


@dataclass
class VPINConfig:
    """
    Layer 1: Bulk Volume Classification (BVC) & VPIN Parameters
    Fang & Feng (2019) Specifications
    """
    n_buckets: int = 50                 # N = 50 volume buckets
    sigma_window: int = 50              # Rolling window for Delta P standard deviation
    delta_2: float = 0.55               # High Toxicity Threshold -> halve size, widen entry
    delta_3: float = 0.25               # Low Toxicity Threshold  -> full allocation, tighten entry
    default_bucket_volume: float = 25000.0  # Default volume V per bucket for Nifty index
    adaptive_bucket_volume: bool = True     # Recalculate V as Daily Volume / N_buckets


@dataclass
class GARCHConfig:
    """
    Layer 2: GARCH(1,1) Volatility Forecasting Parameters
    h_t = omega + alpha * eps_{t-1}^2 + beta * h_{t-1}
    """
    p: int = 1                          # GARCH lag order
    q: int = 1                          # ARCH lag order
    mean_model: str = "AR"              # 'AR' or 'Constant'
    ar_lags: int = 1                    # AR(1) lag for return forecasting
    dist: str = "normal"                # Distribution ('normal' or 't')
    delta_1: float = 0.00015            # Directional threshold for 1-step ahead return mu_{t+1}
    rolling_window: int = 375           # Rolling fitting window (approx 1 full trading day of 1-min bars)
    min_fit_samples: int = 60           # Minimum observations before fitting


@dataclass
class SVMConfig:
    """
    Layer 3: Support Vector Machine (RBF Kernel) Trade-Veto Filter
    Features: [price_delta, rolling_volatility, garch_forecast]
    """
    kernel: str = "rbf"
    c_param: float = 1.0                # Regularization C
    gamma: float = 0.0001               # RBF Kernel bandwidth parameter sigma/gamma = 0.0001
    lookback_days: int = 30             # 30-day rolling feature matrix
    bars_per_day: int = 375             # 375 1-minute bars per standard NSE session
    max_training_samples: int = 375 * 10  # Trailing sample cap for efficient online retraining
    retrain_interval_bars: int = 15     # Retrain SVM every N completed bars


@dataclass
class RiskConfig:
    """Risk Management & Capital Preservation (Calibrated for ₹50,000 Capital)"""
    max_capital: float = 50000.0        # ₹50,000 Total Trading Capital
    max_daily_loss: float = 2500.0      # ₹2,500 Max Daily Loss Limit (5% Max Drawdown Kill-Switch)
    max_position_lots: int = 1          # 1 Lot Max Allocation for ₹50k Capital
    high_toxicity_size_multiplier: float = 1.0   # Sized to 1 lot
    low_toxicity_size_multiplier: float = 1.0    # 1 lot
    normal_toxicity_size_multiplier: float = 1.0
    
    # Dynamic Volatility Stop-Loss & Take-Profit Multipliers
    sl_garch_multiplier: float = 2.0     # SL = Entry - sl_multiplier * sqrt(h_{t+1}) * P_t
    tp_garch_multiplier: float = 3.0     # TP = Entry + tp_multiplier * sqrt(h_{t+1}) * P_t
    max_trade_loss_pct: float = 0.015    # Hard stop loss per trade (1.5%)
    use_super_bracket_orders: bool = False  # If True, places Tradehull place_super_order() with server-side SL/TP


@dataclass
class DhanAPIConfig:
    """Static DhanHQ & Dhan-Tradehull API Settings"""
    auth_mode: str = STATIC_AUTH_MODE
    client_id: str = STATIC_DHAN_CLIENT_ID
    pin: str = STATIC_DHAN_PIN
    totp_secret: str = STATIC_DHAN_TOTP_SECRET
    access_token: str = STATIC_DHAN_ACCESS_TOKEN
    static_ip: Optional[str] = STATIC_IP
    paper_trading: bool = STATIC_PAPER_TRADING
    log_level: str = STATIC_LOG_LEVEL
    enable_telegram: bool = STATIC_ENABLE_TELEGRAM
    telegram_bot_token: str = STATIC_TELEGRAM_BOT_TOKEN
    telegram_chat_id: str = STATIC_TELEGRAM_CHAT_ID


@dataclass
class AppConfig:
    """Master System Configuration"""
    market: MarketConfig = field(default_factory=MarketConfig)
    vpin: VPINConfig = field(default_factory=VPINConfig)
    garch: GARCHConfig = field(default_factory=GARCHConfig)
    svm: SVMConfig = field(default_factory=SVMConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    dhan: DhanAPIConfig = field(default_factory=DhanAPIConfig)


# Global instance
CONFIG = AppConfig()
