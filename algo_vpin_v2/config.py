"""
Global Configuration for Algo VPIN v2.0
Combines Dhan credentials, Multi-Timeframe Parameters, and Ensemble (SVM + XGBoost) Settings.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import logging

logger = logging.getLogger("algo_vpin_v2.config")

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
STATIC_ENABLE_TELEGRAM = _env_bool("ENABLE_TELEGRAM", True)
STATIC_TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
STATIC_TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# Google Gemini AI Agent API Key (Optional - for advanced LLM reflection & autonomous skill generation)
STATIC_GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

if not STATIC_DHAN_CLIENT_ID:
    logger.warning("DHAN_CLIENT_ID is not set. Copy .env.example to .env and fill in your credentials.")


class InstrumentMode(Enum):
    OPTIONS_BUYING = "OPTIONS_BUYING"   # Trades ATM Call on BUY and ATM Put on SELL (Default)
    FUTURES = "FUTURES"                 # Disabled


class ExecutionMode(Enum):
    AUTO_BRAIN_SELECT = "AUTO_BRAIN_SELECT"  # AI Brain dynamically chooses Swing vs Scalper
    SCALPER_ONLY = "SCALPER_ONLY"            # Strictly Sahil Scalper (1:1.5 RR + 0.5R Trail)
    SWING_ONLY = "SWING_ONLY"                # Institutional Trend Runner


class TradeStrategyMode(Enum):
    SCALPER = "SCALPER"                      # 1:1.5 Initial R:R + 0.5R Step Trailing
    INSTITUTIONAL_SWING = "INSTITUTIONAL_SWING"  # Open-ended 20-40 pt Runner



@dataclass
class MarketConfig:
    """NSE/BSE Market Timing & Options Instrument Specifications"""
    symbol: str = "NIFTY"                           # "NIFTY" or "SENSEX"
    underlying: str = "NIFTY"                       # "NIFTY" or "SENSEX"
    trading_symbol: str = "NIFTY"                   # Auto-resolved to active Option contract
    exchange_segment: str = "NFO"                   # "NFO" for Nifty, "BFO" for Sensex
    instrument_type: str = "OPTIDX"
    instrument_mode: InstrumentMode = InstrumentMode.OPTIONS_BUYING
    option_strike_mode: str = "ITM"                 # "ATM", "ITM", or "DELTA_70"
    target_delta: float = 0.72                      # Pro Trader Scalper Delta Sweet-Spot (~0.70 - 0.76)
    lot_size: int = 65                              # 65 for Nifty, 20 for Sensex
    step_size: int = 50                             # 50 for Nifty, 100 for Sensex
    
    # Market Trading Hours (IST: 09:15 to 15:30)
    market_open_hour: int = 9
    market_open_minute: int = 15
    market_close_hour: int = 15
    market_close_minute: int = 30
    
    # New Trade Cutoff (no new trades after 15:15 IST)
    last_entry_hour: int = 15
    last_entry_minute: int = 15

    # Intraday Square-off & Order Cutoff (auto square off active trades at 15:24 IST)
    square_off_hour: int = 15
    square_off_minute: int = 24
    warmup_bars_required: int = 100

    def set_symbol(self, symbol_name: str):
        """Switches underlying between NIFTY and SENSEX"""
        self.underlying = symbol_name.upper()
        self.symbol = self.underlying
        if "NATGAS" in self.underlying or "NATURAL" in self.underlying:
            self.exchange_segment = "MCX_COMM"
            self.instrument_type = "OPTFUT"
            self.lot_size = 250
            self.step_size = 5
            self.market_open_hour = 9
            self.market_open_minute = 0
            self.market_close_hour = 23
            self.market_close_minute = 30
            self.last_entry_hour = 23
            self.last_entry_minute = 15
            self.square_off_hour = 23
            self.square_off_minute = 24
        elif self.underlying == "SENSEX":
            self.exchange_segment = "BFO"
            self.lot_size = 20
            self.step_size = 100
        elif "BANK" in self.underlying:
            self.exchange_segment = "NFO"
            self.lot_size = 30
            self.step_size = 100
        else:
            self.underlying = "NIFTY"
            self.symbol = "NIFTY"
            self.exchange_segment = "NFO"
            self.lot_size = 65
            self.step_size = 50

    def sync_instrument_and_lot_size(self, tradehull_client) -> int:
        under = self.underlying.upper()
        if "NATGAS" in under or "NATURAL" in under:
            self.exchange_segment = "MCX_COMM"
            self.lot_size = 250
            self.step_size = 5
            return self.lot_size
        elif "SENSEX" in under:
            self.exchange_segment = "BFO"
            self.lot_size = 20
            self.step_size = 100
        elif "BANKNIFTY" in under:
            self.exchange_segment = "NFO"
            self.lot_size = 30
            self.step_size = 100
        else:
            self.exchange_segment = "NFO"
            self.lot_size = 65
            self.step_size = 50

        if tradehull_client is not None and hasattr(tradehull_client, "instrument_df"):
            try:
                inst_df = tradehull_client.instrument_df
                if inst_df is not None and not inst_df.empty:
                    match_sym = "SENSEX" if "SENSEX" in under else ("BANKNIFTY" if "BANKNIFTY" in under else "NIFTY")
                    row = inst_df[
                        (inst_df["SEM_INSTRUMENT_NAME"] == "OPTIDX") & 
                        (inst_df["SEM_TRADING_SYMBOL"].str.startswith(match_sym))
                    ]
                    if not row.empty and "SEM_LOT_UNITS" in row.columns:
                        fetched_lot = int(row.iloc[0]["SEM_LOT_UNITS"])
                        if fetched_lot > 0:
                            self.lot_size = fetched_lot
                            logger.info(f"[Tradehull] Synchronized {self.underlying} Options lot size: {self.lot_size}")
                            return self.lot_size
            except Exception as e:
                logger.debug(f"Tradehull lot sync notice: {e}")
        return self.lot_size


@dataclass
class VPINConfig:
    """Layer 1: BVC & VPIN Parameters"""
    n_buckets: int = 50
    default_daily_volume: float = 10000000.0
    sigma_delta_window: int = 30
    regime_low_threshold: float = 0.25
    regime_high_threshold: float = 0.55
    regime_extreme_threshold: float = 0.75


@dataclass
class GARCHConfig:
    """Layer 2: GARCH(1,1) Volatility & Momentum Parameters"""
    p_order: int = 1
    q_order: int = 1
    min_obs_for_fit: int = 50
    refit_interval: int = 10
    buy_threshold_ret: float = 0.00004   # ~1.0 pt/min momentum for NIFTY (responsive to 25-40 pt intraday legs)
    sell_threshold_ret: float = -0.00004 # -1.0 pt/min downward velocity  # Symmetric sell threshold
    dist: str = "normal"


@dataclass
class EnsembleConfig:
    """Layer 3: Ensemble Brain (SVM + XGBoost) Configuration"""
    # SVM Settings
    svm_kernel: str = "rbf"
    svm_c: float = 1.0
    svm_gamma: float = 0.0001
    
    # XGBoost Settings
    xgb_n_estimators: int = 100
    xgb_max_depth: int = 4
    xgb_learning_rate: float = 0.05
    xgb_min_prob_threshold: float = 0.72     # Strict high-conviction >=72% probability threshold to filter noise
    
    # Retraining & History
    retrain_interval: int = 15
    max_history_samples: int = 1000
    consensus_mode: str = "unanimous"        # "unanimous" (both must agree) or "weighted"


@dataclass
class ScalperConfig:
    """Mr Star Sahil Scalping Masterclass Configuration (1:1.5 RR + 0.5R Step Trailing)"""
    initial_risk_reward: float = 1.5           # 1:1.5 Initial Risk-Reward Ratio
    initial_sl_points: float = 10.0            # Default 10 pt Stop-Loss for NIFTY Options
    breakeven_trigger_r: float = 1.0           # Shift SL to Cost at +1.0 R (+10 pts)
    trail_step_r: float = 0.5                  # Step-wise trail SL by +0.5 R for every +0.5 R profit expansion
    brokerage_buffer_pts: float = 1.50         # +1.50 pts above entry to guarantee net-green exit
    min_scalp_sl_points: float = 6.0           # Minimum SL floor to prevent suffocating tight stops
    max_scalp_sl_points: float = 15.0          # Maximum initial SL cap in scalper mode
    trail_past_target: bool = True             # RUNNER MODE: Don't kill winners at target; trail dynamic SL to catch big runners
    runner_trail_buffer_pts: float = 5.5       # Dynamic trailing buffer (pts) behind highest peak once target is reached


@dataclass
class RiskConfig:
    """Institutional Risk Limits Calibrated for ₹50,000 Capital"""
    execution_mode: ExecutionMode = ExecutionMode.AUTO_BRAIN_SELECT
    capital_allocation: float = 50000.0
    max_capital_per_trade: float = 40000.0
    max_daily_loss_inr: float = 2500.0       # 5% Max Daily Drawdown Kill-Switch (if enabled)
    daily_target_profit_inr: float = 5000.0  # Daily Profit Goal Target Lock
    max_daily_trades: int = 10               # Dynamic Daily Trades Limit (Expanded for Expiry Days)
    enable_kill_switch: bool = False         # False = Disabled for continuous trade monitoring
    max_open_positions: int = 1
    max_lots: int = 1
    base_lots: int = 1
    
    # Dynamic Option Volatility Multipliers
    sl_garch_multiplier: float = 1.5
    tp_garch_multiplier: float = 2.5
    max_initial_sl_points: float = 25.0      # Hybrid SL point cap (₹1,625 max risk per lot)
    max_trade_loss_pct: float = 0.30         # -30% Option Stop-Loss Hard Cap
    slippage_pct: float = 0.001

    # Dynamic Capital Allocation & Margin Guard
    max_capital_utilization_pct: float = 0.50 # Max 50% available cash utilized per trade
    consecutive_losses_threshold: int = 2     # Triggers High-Conviction Re-qualification Cautious State


@dataclass
class DhanAPIConfig:
    client_id: str = STATIC_DHAN_CLIENT_ID
    pin: str = STATIC_DHAN_PIN
    totp_secret: str = STATIC_DHAN_TOTP_SECRET
    access_token: str = STATIC_DHAN_ACCESS_TOKEN
    auth_mode: str = STATIC_AUTH_MODE
    paper_trading: bool = STATIC_PAPER_TRADING
    ip_address: str = STATIC_IP


@dataclass
class TelegramConfig:
    enabled: bool = STATIC_ENABLE_TELEGRAM
    bot_token: str = STATIC_TELEGRAM_BOT_TOKEN
    chat_id: str = STATIC_TELEGRAM_CHAT_ID


@dataclass
class GeminiConfig:
    api_key: str = STATIC_GEMINI_API_KEY
    model: str = "gemini-2.5-flash"


@dataclass
class AppConfig:
    dhan: DhanAPIConfig = field(default_factory=DhanAPIConfig)
    market: MarketConfig = field(default_factory=MarketConfig)
    vpin: VPINConfig = field(default_factory=VPINConfig)
    garch: GARCHConfig = field(default_factory=GARCHConfig)
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    scalper: ScalperConfig = field(default_factory=ScalperConfig)
    telegram: TelegramConfig = field(default_factory=TelegramConfig)
    gemini: GeminiConfig = field(default_factory=GeminiConfig)


CONFIG = AppConfig()

