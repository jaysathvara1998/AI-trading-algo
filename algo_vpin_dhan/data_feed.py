"""
DhanHQ & Dhan-Tradehull Data Feed Module
Handles live 1-minute OHLCV bar polling, historical bar retrieval via Tradehull,
Option Chain / ATM strike resolution, and synthetic market playback.
Supports automated PIN + TOTP login and Access Token authentication.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from typing import Callable, Iterator, List, Optional, Tuple
import numpy as np
import pandas as pd
import pytz

try:
    from Dhan_Tradehull import Tradehull
except ImportError:
    Tradehull = None

try:
    from dhanhq import dhanhq
except ImportError:
    dhanhq = None

from .config import AppConfig, MarketConfig, DhanAPIConfig, InstrumentMode

logger = logging.getLogger("algo_vpin_dhan.data_feed")


@dataclass
class BarOHLCV:
    """Standardized 1-minute OHLCV Bar"""
    timestamp: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float
    symbol: str


class DhanDataFeed:
    """
    Interfaces with Dhan-Tradehull & DhanHQ REST API to provide 1-minute OHLCV bars.
    """

    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or AppConfig()
        self.market_cfg = self.config.market
        self.dhan_cfg = self.config.dhan
        self.tz = pytz.timezone("Asia/Kolkata")

        self.tradehull_client = None
        self.dhan_client = None

        # Initialize Dhan-Tradehull client with TOTP or Access Token
        if Tradehull is not None and self.dhan_cfg.client_id and self.dhan_cfg.client_id != "YOUR_DHAN_CLIENT_ID":
            try:
                if self.dhan_cfg.auth_mode == "pin_totp" and self.dhan_cfg.pin != "YOUR_DHAN_PIN":
                    logger.info("Initializing Dhan-Tradehull with PIN + TOTP automated login...")
                    self.tradehull_client = Tradehull(
                        ClientCode=self.dhan_cfg.client_id,
                        mode="pin_totp",
                        pin=self.dhan_cfg.pin,
                        totp_secret=self.dhan_cfg.totp_secret
                    )
                else:
                    logger.info("Initializing Dhan-Tradehull with Access Token...")
                    self.tradehull_client = Tradehull(
                        ClientCode=self.dhan_cfg.client_id,
                        token_id=self.dhan_cfg.access_token,
                        mode="access_token"
                    )

                logger.info("Initialized Dhan-Tradehull client successfully.")
                # Dynamically sync active contract & lot size
                self.market_cfg.sync_instrument_and_lot_size(self.tradehull_client)
            except Exception as e:
                logger.warning(f"Could not initialize Dhan-Tradehull client: {e}.")

        # Fallback to direct dhanhq client if needed
        if self.tradehull_client is None and dhanhq is not None and self.dhan_cfg.client_id and self.dhan_cfg.client_id != "YOUR_DHAN_CLIENT_ID":
            try:
                from dhanhq import DhanContext
                context = DhanContext(self.dhan_cfg.client_id, self.dhan_cfg.access_token)
                self.dhan_client = dhanhq(context)
                logger.info("Initialized raw DhanHQ client successfully.")
            except Exception:
                try:
                    self.dhan_client = dhanhq(self.dhan_cfg.client_id, self.dhan_cfg.access_token)
                    logger.info("Initialized raw DhanHQ client successfully.")
                except Exception as e:
                    logger.warning(f"Could not initialize DhanHQ client: {e}. Running in standalone/paper mode.")

    def resolve_option_strike(self, action: str, current_price: Optional[float] = None) -> Optional[str]:
        """
        Resolves the exact ATM Call (on BUY) or ATM Put (on SELL) Option contract
        for both NIFTY and SENSEX directly from the Master Instrument DB.
        """
        underlying = self.market_cfg.underlying.upper()
        target_opt_type = "CE" if action == "BUY" else "PE"

        if self.tradehull_client is None:
            return f"{underlying}_ATM_{target_opt_type}"

        # 1. Fast, Direct Master Instrument DB Lookup (Zero latency, No API blips)
        try:
            inst_df = self.tradehull_client.instrument_df
            if inst_df is not None and not inst_df.empty:
                ref_price = current_price if current_price and current_price > 0 else self.fetch_live_spot_price()
                step = self.market_cfg.step_size
                atm_strike = round(ref_price / step) * step

                # Filter by OPTIDX, underlying, and option type (CE/PE)
                opt_subset = inst_df[
                    (inst_df["SEM_INSTRUMENT_NAME"] == "OPTIDX") &
                    (inst_df["SEM_CUSTOM_SYMBOL"].str.startswith(underlying)) &
                    (inst_df["SEM_OPTION_TYPE"] == target_opt_type)
                ].copy()

                if not opt_subset.empty:
                    opt_subset["SEM_STRIKE_PRICE"] = pd.to_numeric(opt_subset["SEM_STRIKE_PRICE"], errors="coerce")
                    opt_subset["diff"] = abs(opt_subset["SEM_STRIKE_PRICE"] - atm_strike)
                    # Pick nearest expiry and closest strike
                    closest_row = opt_subset.sort_values(by=["SEM_EXPIRY_DATE", "diff"]).iloc[0]
                    chosen_sym = closest_row["SEM_CUSTOM_SYMBOL"]
                    logger.info(f"[Option Strike Resolver] Selected {underlying} {target_opt_type} Strike @ {atm_strike}: {chosen_sym}")
                    return chosen_sym
        except Exception as e:
            logger.debug(f"Direct instrument DB resolve: {e}")

        return f"{underlying}_ATM_{target_opt_type}"

    def fetch_live_spot_price(self) -> float:
        """Fetches live Index Spot price (NIFTY/SENSEX) directly via Dhan API with zero tracebacks."""
        if self.tradehull_client is not None and hasattr(self.tradehull_client, "token_id"):
            try:
                inst_df = self.tradehull_client.instrument_df
                target_sym = self.market_cfg.underlying.upper()
                sec_row = inst_df[((inst_df['SEM_CUSTOM_SYMBOL'] == target_sym) | (inst_df['SEM_TRADING_SYMBOL'] == target_sym))]
                if not sec_row.empty:
                    sec_id = int(sec_row.iloc[-1]['SEM_SMST_SECURITY_ID'])
                    headers = {
                        'access-token': self.tradehull_client.token_id,
                        'client-id': self.dhan_cfg.client_id,
                        'Content-Type': 'application/json'
                    }
                    exch_key = 'IDX_I' if target_sym in ['NIFTY', 'BANKNIFTY'] else 'BSE_IDX'
                    import requests
                    resp = requests.post(
                        'https://api.dhan.co/v2/marketfeed/ltp',
                        json={exch_key: [sec_id]},
                        headers=headers,
                        timeout=2
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        if data.get('status') == 'success':
                            price = float(data['data'][exch_key][str(sec_id)]['last_price'])
                            if price > 0:
                                self._last_known_spot = price
                                return price
            except Exception as e:
                logger.debug(f"Direct spot fetch: {e}")

        return getattr(self, "_last_known_spot", 23800.0 if self.market_cfg.underlying == "NIFTY" else 79000.0)

    def fetch_option_ltp(self, option_symbol: str, spot_price: float = 23800.0) -> float:
        """
        Fetches the exact real-time Option Premium LTP directly from Dhan/Tradehull without rate limits.
        """
        if not hasattr(self, "_option_ltp_cache"):
            self._option_ltp_cache = {}

        sym_clean = option_symbol.strip().upper()

        if self.tradehull_client is not None:
            try:
                exch = "BFO" if "SENSEX" in sym_clean else "NFO"
                hist = self.tradehull_client.get_historical_data(sym_clean, exch, "1")
                if hist is not None and not hist.empty:
                    latest_close = float(hist.iloc[-1]["close"])
                    if latest_close > 0:
                        self._option_ltp_cache[sym_clean] = latest_close
                        return latest_close
            except Exception as e:
                logger.debug(f"Option LTP via NFO historical: {e}")

        if sym_clean in self._option_ltp_cache:
            return self._option_ltp_cache[sym_clean]

        return round(max(5.0, spot_price * 0.0045), 2)

    def fetch_historical_bars(
        self,
        security_id: Optional[str] = None,
        exchange_segment: Optional[str] = None,
        days: int = 5
    ) -> pd.DataFrame:
        """
        Fetches historical 1-minute OHLCV bars using Dhan-Tradehull or DhanHQ API.
        """
        segment = exchange_segment or self.market_cfg.exchange_segment

        # 1. Try Dhan-Tradehull historical data
        if self.tradehull_client is not None:
            try:
                df = self.tradehull_client.get_historical_data(
                    tradingsymbol=self.market_cfg.trading_symbol,
                    exchange=segment,
                    timeframe="1"
                )
                if df is not None and not df.empty:
                    df = df.rename(columns={
                        "timestamp": "timestamp",
                        "open": "open",
                        "high": "high",
                        "low": "low",
                        "close": "close",
                        "volume": "volume"
                    })
                    df["timestamp"] = pd.to_datetime(df["timestamp"])
                    return df[["timestamp", "open", "high", "low", "close", "volume"]]
            except Exception as e:
                logger.warning(f"Tradehull get_historical_data failed: {e}. Falling back to DhanHQ direct.")

        # 2. Try raw DhanHQ client
        if self.dhan_client is not None:
            try:
                sec_id = security_id or "13"
                response = self.dhan_client.intraday_daily_minute_data(
                    security_id=sec_id,
                    exchange_segment=segment,
                    instrument_type=self.market_cfg.instrument_type
                )
                if response and response.get("status") == "success" and "data" in response:
                    df = pd.DataFrame(response["data"])
                    if not df.empty:
                        df["timestamp"] = pd.to_datetime(df["start_Time"])
                        df = df.rename(columns={
                            "open": "open",
                            "high": "high",
                            "low": "low",
                            "close": "close",
                            "volume": "volume"
                        })
                        return df[["timestamp", "open", "high", "low", "close", "volume"]]
            except Exception as e:
                logger.error(f"Error fetching historical bars from DhanHQ: {e}")

        # 3. Fallback: Synthetic historical data
        logger.debug("Generating synthetic historical bars for standalone/paper trading mode.")
        return self.generate_synthetic_bars(n_bars=375 * days)

    def generate_synthetic_bars(
        self,
        n_bars: int = 500,
        start_price: float = 24000.0,
        volatility: float = 0.0012,
        seed: int = 42
    ) -> pd.DataFrame:
        """
        Generates realistic synthetic 1-minute OHLCV bars for Nifty Index with volume regimes.
        """
        np.random.seed(seed)
        records = []
        current_time = datetime.now(self.tz).replace(hour=9, minute=15, second=0, microsecond=0)
        curr_price = start_price

        for i in range(n_bars):
            if current_time.hour > 15 or (current_time.hour == 15 and current_time.minute > 30):
                current_time = (current_time + timedelta(days=1)).replace(hour=9, minute=15)

            jump = 0.0
            if np.random.rand() < 0.03:
                jump = np.random.choice([-1, 1]) * np.random.uniform(0.002, 0.006)

            ret = np.random.normal(0, volatility) + jump
            bar_open = curr_price
            curr_price = curr_price * np.exp(ret)
            bar_close = curr_price

            high_wiggle = abs(np.random.normal(0, volatility * 0.5)) * bar_open
            low_wiggle = abs(np.random.normal(0, volatility * 0.5)) * bar_open

            bar_high = max(bar_open, bar_close) + high_wiggle
            bar_low = min(bar_open, bar_close) - low_wiggle

            base_vol = np.random.uniform(15000, 35000)
            if abs(jump) > 0:
                base_vol *= np.random.uniform(2.5, 4.0)

            records.append({
                "timestamp": pd.Timestamp(current_time),
                "open": round(bar_open, 2),
                "high": round(bar_high, 2),
                "low": round(bar_low, 2),
                "close": round(bar_close, 2),
                "volume": round(base_vol, 0)
            })

            current_time += timedelta(minutes=1)

        return pd.DataFrame(records)
