"""
Zerodha Kite Connect broker adapter and data feed for Algo VPIN v2.

KiteBroker  - thin wrapper exposing the calls the execution router uses (order_placement,
              get_order_status, get_balance) plus the raw KiteConnect client as `.kite`.
KiteDataFeed - drop-in replacement for DhanDataFeed: same public methods used by main.py and
              execution.py (get_ltp, get_multiple_ltp, fetch_fresh_ltp, resolve_option_strike,
              fetch_historical_bars, fetch_last_completed_minute_bar, generate_synthetic_bars).

Daily login: Kite access tokens expire every morning. Run `python kite_login.py` once per day; it writes
Dependencies/kite_token_<YYYY-MM-DD>.txt which this feed picks up (or set KITE_ACCESS_TOKEN in .env).
"""
from __future__ import annotations

import glob
import logging
import math
import time
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import numpy as np
import pandas as pd
import pytz

try:
    from kiteconnect import KiteConnect
except ImportError:  # pragma: no cover
    KiteConnect = None

from .config import AppConfig
from .data_feed import DhanDataFeed

logger = logging.getLogger("algo_vpin_v2.kite_feed")
IST = pytz.timezone("Asia/Kolkata")
ROOT = Path(__file__).resolve().parents[1]

# Kite instrument tokens and quote keys for the index spots
SPOT = {
    "NIFTY":     {"token": 256265, "quote": "NSE:NIFTY 50",   "name": "NIFTY",     "opt_exch": "NFO"},
    "NIFTY 50":  {"token": 256265, "quote": "NSE:NIFTY 50",   "name": "NIFTY",     "opt_exch": "NFO"},
    "BANKNIFTY": {"token": 260105, "quote": "NSE:NIFTY BANK", "name": "BANKNIFTY", "opt_exch": "NFO"},
    "SENSEX":    {"token": 265,    "quote": "BSE:SENSEX",     "name": "SENSEX",    "opt_exch": "BFO"},
}
SESSION_OPEN, SESSION_CLOSE = (9, 15), (15, 30)


def load_kite_token(explicit: str = "") -> Tuple[str, str]:
    """Returns (access_token, source). Explicit env value wins; else today's Dependencies/kite_token_<date>.txt."""
    if explicit:
        return explicit.strip(), "env"
    today = date.today().isoformat()
    f = ROOT / "Dependencies" / f"kite_token_{today}.txt"
    if f.exists():
        parts = f.read_text().strip().split("|")
        if len(parts) >= 2 and parts[0] == today and parts[1]:
            return parts[1], str(f)
    return "", "none"


# ---------------------------------------------------------------------------------------------- broker
class KiteBroker:
    """Order/account calls with the same method names the execution router already uses."""

    def __init__(self, api_key: str, access_token: str):
        if KiteConnect is None:
            raise RuntimeError("kiteconnect is not installed (pip install kiteconnect)")
        self.kite = KiteConnect(api_key=api_key)
        self.kite.set_access_token(access_token)
        self.profile = self.kite.profile()          # raises on an invalid/expired token
        self.user_id = self.profile.get("user_id", "")
        self.instrument_df: Optional[pd.DataFrame] = None   # compat: Dhan-specific code checks this attr and skips

    def order_placement(self, tradingsymbol: str, exchange: str, quantity: int, price=0, trigger_price=0,
                        order_type: str = "MARKET", transaction_type: str = "BUY", trade_type: str = "MIS", **_) -> Optional[str]:
        """Kite place_order with MIS product. Returns order id or None (never raises)."""
        try:
            oid = self.kite.place_order(
                variety=self.kite.VARIETY_REGULAR, exchange=exchange.upper(), tradingsymbol=tradingsymbol.upper(),
                transaction_type=transaction_type.upper(), quantity=int(quantity), product=self.kite.PRODUCT_MIS,
                order_type=order_type.upper(), price=None if order_type.upper() == "MARKET" else float(price),
                validity=self.kite.VALIDITY_DAY, tag="algovpin",
            )
            return str(oid) if oid else None
        except Exception as e:
            logger.critical(f"[Kite] place_order failed for {transaction_type} {quantity} {tradingsymbol}: {e}")
            return None

    def get_order_status(self, order_id: str) -> str:
        try:
            time.sleep(0.6)
            hist = self.kite.order_history(str(order_id))
            return str(hist[-1].get("status", "UNKNOWN")).upper() if hist else "UNKNOWN"
        except Exception as e:
            return f"UNKNOWN ({e})"

    def get_balance(self) -> float:
        try:
            m = self.kite.margins("equity")
            avail = m.get("available", {})
            return float(avail.get("live_balance") or avail.get("cash") or m.get("net") or 0.0)
        except Exception as e:
            logger.debug(f"[Kite] margins: {e}")
            return 0.0

    def get_positions(self):
        return self.kite.positions()


# ---------------------------------------------------------------------------------------------- feed
def _bs_call_put(spot, strike, t_years, sigma, r=0.065):
    if t_years <= 0 or sigma <= 0:
        return max(0.0, spot - strike), max(0.0, strike - spot), 1.0 if spot > strike else 0.0
    d1 = (math.log(spot / strike) + (r + 0.5 * sigma ** 2) * t_years) / (sigma * math.sqrt(t_years))
    d2 = d1 - sigma * math.sqrt(t_years)
    N = lambda x: 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
    call = spot * N(d1) - strike * math.exp(-r * t_years) * N(d2)
    put = strike * math.exp(-r * t_years) * N(-d2) - spot * N(-d1)
    return call, put, N(d1)


def _implied_vol(price, spot, strike, t_years, is_call):
    lo, hi = 0.02, 2.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        c, p, _ = _bs_call_put(spot, strike, t_years, mid)
        if (c if is_call else p) > price:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


class KiteDataFeed(DhanDataFeed):
    """Kite-backed feed. Inherits the offline helpers (synthetic bars, local gz history fallback) from DhanDataFeed."""

    def __init__(self, config: Optional[AppConfig] = None):
        self.broker: Optional[KiteBroker] = None
        self.kite = None
        self.authenticated = False
        self.last_delta: Optional[float] = None
        self._inst_cache: Dict[str, pd.DataFrame] = {}
        self._inst_cache_day: Optional[date] = None
        self._ltp_cache: Dict[str, float] = {}
        self._ltp_time_cache: Dict[str, float] = {}
        self._option_ltp_cache: Dict[str, float] = {}
        self._option_time_cache: Dict[str, float] = {}
        self._last_api_call_ts = 0.0
        super().__init__(config)

    # ---------------- session ----------------
    def _init_clients(self):
        self.tradehull_client = None      # compat attribute: Dhan-specific code paths check it and skip
        self.dhan_client = None
        kc = self.config.kite
        token, source = load_kite_token(kc.access_token)
        if not kc.api_key:
            logger.critical("[Kite] KITE_API_KEY is not set in .env")
            return
        if not token:
            logger.critical("[Kite] No access token for today. Run `python kite_login.py` first (or set KITE_ACCESS_TOKEN). Feed is OFFLINE.")
            return
        try:
            self.broker = KiteBroker(kc.api_key, token)
            self.kite = self.broker.kite
            self.tradehull_client = self.broker   # execution router / scanners receive this object
            self.authenticated = True
            logger.info(f"[Kite] Logged in as {self.broker.profile.get('user_name', self.broker.user_id)} (token from {source}).")
            self.market_cfg.sync_instrument_and_lot_size(None)
            self._sync_lot_size_from_instruments()
        except Exception as e:
            logger.critical(f"[Kite] Login with today's token failed: {e}. Run `python kite_login.py` again.")

    # ---------------- instruments ----------------
    def _instruments(self, exchange: str) -> pd.DataFrame:
        today = datetime.now(IST).date()
        if self._inst_cache_day != today:
            self._inst_cache = {}; self._inst_cache_day = today
        if exchange not in self._inst_cache:
            df = pd.DataFrame(self.kite.instruments(exchange))
            if not df.empty:
                df["expiry"] = pd.to_datetime(df["expiry"]).dt.date
            self._inst_cache[exchange] = df
        return self._inst_cache[exchange]

    def _spot_info(self):
        return SPOT.get(self.market_cfg.underlying.upper(), SPOT["NIFTY"])

    def _sync_lot_size_from_instruments(self):
        try:
            info = self._spot_info(); df = self._instruments(info["opt_exch"])
            opts = df[(df["name"] == info["name"]) & (df["instrument_type"].isin(["CE", "PE"]))]
            if not opts.empty:
                lot = int(opts.sort_values("expiry").iloc[0]["lot_size"])
                if lot > 0 and lot != self.market_cfg.lot_size:
                    logger.info(f"[Kite] Lot size for {info['name']} from instrument master: {lot} (config had {self.market_cfg.lot_size})")
                    self.market_cfg.lot_size = lot
        except Exception as e:
            logger.debug(f"[Kite] lot size sync: {e}")

    def _nearest_expiry(self, info) -> Optional[date]:
        df = self._instruments(info["opt_exch"]); today = datetime.now(IST).date()
        opts = df[(df["name"] == info["name"]) & (df["instrument_type"].isin(["CE", "PE"])) & (df["expiry"] >= today)]
        return None if opts.empty else opts["expiry"].min()

    def _option_symbol(self, info, expiry, strike: float, opt_type: str) -> Optional[str]:
        df = self._instruments(info["opt_exch"])
        m = df[(df["name"] == info["name"]) & (df["expiry"] == expiry) & (df["instrument_type"] == opt_type) & (np.isclose(df["strike"], strike))]
        return None if m.empty else str(m.iloc[0]["tradingsymbol"])

    def _quote_key(self, symbol: str) -> str:
        s = symbol.strip().upper()
        if s in SPOT:
            return SPOT[s]["quote"]
        if ":" in s:
            return s
        return f"{'BFO' if s.startswith(('SENSEX', 'BANKEX')) else 'NFO'}:{s}"

    # ---------------- quotes ----------------
    def _ltp_many(self, keys: List[str]) -> Dict[str, float]:
        if not self.authenticated or not keys:
            return {}
        self._throttle_api_call(1.0)
        try:
            res = self.kite.ltp(keys)
        except Exception as e:
            logger.warning(f"[Kite] ltp failed: {e}")
            return {}
        out = {}
        now = time.time()
        for k, v in res.items():
            p = float(v.get("last_price", 0.0) or 0.0)
            if p > 0:
                out[k] = p; self._ltp_cache[k] = p; self._ltp_time_cache[k] = now
        return out

    def fetch_live_spot_price(self) -> float:
        key = self._spot_info()["quote"]
        if key in self._ltp_cache and time.time() - self._ltp_time_cache.get(key, 0) < 2.0:
            return self._ltp_cache[key]
        p = self._ltp_many([key]).get(key, 0.0)
        if p <= 0:
            p = self._ltp_cache.get(key, 0.0)
        if p <= 0:
            logger.critical("[Kite] No live spot price available. Returning 0 so callers retry.")
        return p

    def fetch_option_ltp(self, option_symbol: str, spot_price: float = 0.0) -> float:
        sym = option_symbol.strip().upper()
        if sym.endswith(("_CE", "_PE")) and "_ATM_" in sym:
            return 0.0
        key = self._quote_key(sym)
        if key in self._ltp_cache and time.time() - self._ltp_time_cache.get(key, 0) < 1.5:
            return self._ltp_cache[key]
        p = self._ltp_many([key]).get(key, 0.0)
        if p <= 0:
            p = self._ltp_cache.get(key, 0.0)
        if p <= 0:
            logger.warning(f"[Kite] No LTP for {sym}; returning 0 (never synthetic while a broker session exists).")
        return p

    def get_multiple_ltp(self, symbols: List[str]) -> Dict[str, float]:
        keys = {s: self._quote_key(s) for s in symbols if s}
        res = self._ltp_many(list(set(keys.values())))
        return {s: res.get(k, self._ltp_cache.get(k, 0.0)) for s, k in keys.items()}

    def get_ltp(self, symbol: str, spot_price: Optional[float] = None) -> float:
        s = symbol.strip().upper()
        if s in SPOT:
            return self.fetch_live_spot_price()
        return self.fetch_option_ltp(s, spot_price or 0.0)

    def fetch_fresh_ltp(self, symbol: str, spot_price: Optional[float] = None) -> float:
        key = self._quote_key(symbol)
        self._ltp_time_cache.pop(key, None)
        return self.get_ltp(symbol, spot_price)

    # ---------------- strike resolution (target delta via Black-Scholes on live premiums) ----------------
    def resolve_option_strike(self, action: str, current_price: Optional[float] = None) -> Optional[str]:
        info = self._spot_info(); opt_type = "CE" if action == "BUY" else "PE"
        if not self.authenticated:
            return f"{info['name']}_ATM_{opt_type}"
        try:
            spot = current_price if current_price and current_price > 0 else self.fetch_live_spot_price()
            if spot <= 0:
                return f"{info['name']}_ATM_{opt_type}"
            expiry = self._nearest_expiry(info)
            if expiry is None:
                return f"{info['name']}_ATM_{opt_type}"
            step = self.market_cfg.step_size
            atm = round(spot / step) * step
            mode = str(getattr(self.market_cfg, "option_strike_mode", "ITM")).upper()
            if mode == "ATM":
                return self._option_symbol(info, expiry, atm, opt_type) or f"{info['name']}_ATM_{opt_type}"
            # candidates: ATM and up to 3 strikes in the money
            itm_dir = -1 if opt_type == "CE" else +1
            strikes = [atm + itm_dir * k * step for k in range(0, 4)]
            syms = {s: self._option_symbol(info, expiry, s, opt_type) for s in strikes}
            syms = {s: v for s, v in syms.items() if v}
            if not syms:
                return f"{info['name']}_ATM_{opt_type}"
            ltps = self._ltp_many([f"{info['opt_exch']}:{v}" for v in syms.values()])
            prem = {s: ltps.get(f"{info['opt_exch']}:{v}", 0.0) for s, v in syms.items()}
            now = datetime.now(IST)
            exp_dt = IST.localize(datetime.combine(expiry, datetime.min.time()).replace(hour=15, minute=30))
            t_years = max((exp_dt - now).total_seconds(), 1800.0) / (365.0 * 24 * 3600)
            target = float(getattr(self.market_cfg, "target_delta", 0.72))
            best, best_gap = None, 9.9
            deltas = {}
            atm_p = prem.get(atm, 0.0)
            sigma = _implied_vol(atm_p, spot, atm, t_years, opt_type == "CE") if atm_p > 0 else 0.14
            for s, p in prem.items():
                _, _, nd1 = _bs_call_put(spot, s, t_years, sigma)
                delta = nd1 if opt_type == "CE" else (1.0 - nd1)
                deltas[s] = delta
                if abs(delta - target) < best_gap:
                    best, best_gap = s, abs(delta - target)
            chosen = best if best is not None else (atm + itm_dir * step)
            self.last_delta = deltas.get(chosen, target)
            sym = syms.get(chosen) or syms.get(atm)
            logger.info(f"[Kite] Strike {chosen:.0f} {opt_type} (expiry {expiry}, IV {sigma:.1%}, delta gap {best_gap:.2f}) -> {sym}")
            return sym or f"{info['name']}_ATM_{opt_type}"
        except Exception as e:
            logger.warning(f"[Kite] resolve_option_strike: {e}")
            return f"{info['name']}_ATM_{opt_type}"

    # ---------------- history ----------------
    def _hist(self, token: int, frm: datetime, to: datetime) -> pd.DataFrame:
        rows = []
        cur = frm
        while cur < to:                       # Kite serves at most 60 days of minute data per call
            end = min(cur + timedelta(days=59), to)
            self._throttle_api_call(0.4)
            rows += self.kite.historical_data(token, cur, end, "minute")
            cur = end + timedelta(minutes=1)
        if not rows:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume", "timestamp"])
        df = pd.DataFrame(rows).rename(columns={"date": "timestamp"})
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        if df["timestamp"].dt.tz is None:
            df["timestamp"] = df["timestamp"].dt.tz_localize(IST)
        else:
            df["timestamp"] = df["timestamp"].dt.tz_convert(IST)
        t = df["timestamp"].dt.time
        df = df[(t >= datetime.min.time().replace(hour=9, minute=15)) & (t < datetime.min.time().replace(hour=15, minute=30))]
        return df[["open", "high", "low", "close", "volume", "timestamp"]].reset_index(drop=True)

    def fetch_historical_bars(self, days: int = 5, symbol: Optional[str] = None) -> pd.DataFrame:
        if self.authenticated:
            try:
                info = SPOT.get((symbol or self.market_cfg.underlying).upper(), self._spot_info())
                now = datetime.now(IST)
                df = self._hist(info["token"], now - timedelta(days=days + 3), now)
                if len(df) >= 375:
                    logger.info(f"[Kite] Loaded {len(df)} historical 1-minute bars for {info['name']}.")
                    return df
            except Exception as e:
                logger.warning(f"[Kite] historical_data failed ({e}); using local history.")
        return super().fetch_historical_bars(days=days, symbol=symbol)

    def fetch_last_completed_minute_bar(self, now_ist) -> Optional[Tuple[float, float, float, float, float]]:
        if not self.authenticated:
            return None
        info = self._spot_info()
        want = pd.Timestamp(now_ist).floor("min") - pd.Timedelta(minutes=1)
        self._throttle_api_call(0.4)
        rows = self.kite.historical_data(info["token"], want.to_pydatetime() - timedelta(minutes=3), want.to_pydatetime() + timedelta(minutes=1), "minute")
        for r in reversed(rows):
            ts = pd.Timestamp(r["date"])
            ts = ts.tz_localize(IST) if ts.tzinfo is None else ts.tz_convert(IST)
            if ts == want:
                return float(r["open"]), float(r["high"]), float(r["low"]), float(r["close"]), float(r.get("volume", 0) or 0)
        return None
