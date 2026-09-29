"""
Zerodha Kite Connect adapter for the live runner: spot minute bars, option contract selection (nearest weekly
expiry, target-delta strike via Black-Scholes on live premiums), quotes, MIS market orders with verification.
Paper mode never touches place_order.
"""
from __future__ import annotations

import logging
import math
import time
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd
import pytz

from .env import ROOT, env

log = logging.getLogger("pa_engine.kite")
IST = pytz.timezone("Asia/Kolkata")
SPOT = {"NIFTY": {"token": 256265, "quote": "NSE:NIFTY 50", "name": "NIFTY", "opt_exch": "NFO", "step": 50},
        "BANKNIFTY": {"token": 260105, "quote": "NSE:NIFTY BANK", "name": "BANKNIFTY", "opt_exch": "NFO", "step": 100},
        "SENSEX": {"token": 265, "quote": "BSE:SENSEX", "name": "SENSEX", "opt_exch": "BFO", "step": 100}}


def load_token(explicit: str = "") -> Tuple[str, str]:
    if explicit:
        return explicit.strip(), "env"
    today = date.today().isoformat()
    f = ROOT / "Dependencies" / f"kite_token_{today}.txt"
    if f.exists():
        parts = f.read_text().strip().split("|")
        if len(parts) >= 2 and parts[0] == today and parts[1]:
            return parts[1], str(f)
    return "", "none"


def _bs(spot, strike, t, sigma, r=0.065):
    if t <= 0 or sigma <= 0:
        return max(0.0, spot - strike), max(0.0, strike - spot), 1.0 if spot > strike else 0.0
    d1 = (math.log(spot / strike) + (r + 0.5 * sigma ** 2) * t) / (sigma * math.sqrt(t)); d2 = d1 - sigma * math.sqrt(t)
    N = lambda x: 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
    return spot * N(d1) - strike * math.exp(-r * t) * N(d2), strike * math.exp(-r * t) * N(-d2) - spot * N(-d1), N(d1)


def _iv(price, spot, strike, t, is_call):
    lo, hi = 0.02, 2.0
    for _ in range(40):
        mid = 0.5 * (lo + hi); c, p, _ = _bs(spot, strike, t, mid)
        lo, hi = (lo, mid) if (c if is_call else p) > price else (mid, hi)
    return 0.5 * (lo + hi)


class Kite:
    def __init__(self, symbol: str):
        from kiteconnect import KiteConnect
        self.info = SPOT[symbol.upper()]
        key = env("KITE_API_KEY"); token, src = load_token(env("KITE_ACCESS_TOKEN"))
        if not key:
            raise RuntimeError("KITE_API_KEY is not set in .env")
        if not token:
            raise RuntimeError("no Kite access token for today: run `python -m pa_engine login`")
        self.kite = KiteConnect(api_key=key); self.kite.set_access_token(token)
        self.profile = self.kite.profile(); self.user = self.profile.get("user_name") or self.profile.get("user_id")
        log.info(f"Kite login OK as {self.user} (token from {src})")
        self._inst: Optional[pd.DataFrame] = None; self._last_call = 0.0

    def _throttle(self, gap: float):
        w = self._last_call + gap - time.time()
        if w > 0:
            time.sleep(w)
        self._last_call = time.time()

    # ---------------- market data
    def minute_bars(self, frm: datetime, to: datetime) -> pd.DataFrame:
        self._throttle(0.35)
        rows = self.kite.historical_data(self.info["token"], frm, to, "minute")
        if not rows:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
        df = pd.DataFrame(rows).rename(columns={"date": "timestamp"})
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["timestamp"] = df["timestamp"].dt.tz_localize(IST) if df["timestamp"].dt.tz is None else df["timestamp"].dt.tz_convert(IST)
        return df[["timestamp", "open", "high", "low", "close", "volume"]]

    def ltp(self, keys: List[str]) -> Dict[str, float]:
        self._throttle(0.35)
        try:
            return {k: float(v.get("last_price") or 0.0) for k, v in self.kite.ltp(keys).items()}
        except Exception as e:
            log.warning(f"ltp failed: {e}"); return {}

    def spot(self) -> float:
        return self.ltp([self.info["quote"]]).get(self.info["quote"], 0.0)

    # ---------------- contracts
    def instruments(self) -> pd.DataFrame:
        if self._inst is None:
            df = pd.DataFrame(self.kite.instruments(self.info["opt_exch"]))
            df["expiry"] = pd.to_datetime(df["expiry"]).dt.date
            self._inst = df[(df["name"] == self.info["name"]) & (df["instrument_type"].isin(["CE", "PE"]))]
        return self._inst

    def lot_size(self) -> int:
        df = self.instruments()
        return int(df.sort_values("expiry").iloc[0]["lot_size"]) if len(df) else 0

    def nearest_expiry(self) -> Optional[date]:
        df = self.instruments(); today = datetime.now(IST).date()
        e = df[df["expiry"] >= today]["expiry"]
        return None if e.empty else e.min()

    def pick_option(self, direction: int, spot: float, target_delta: float = 0.72) -> Optional[dict]:
        """Nearest expiry, strike whose BS delta (IV from the ATM premium) is closest to target_delta."""
        opt = "CE" if direction > 0 else "PE"; exp = self.nearest_expiry()
        if exp is None or spot <= 0:
            return None
        df = self.instruments(); step = self.info["step"]; atm = round(spot / step) * step
        itm = -1 if opt == "CE" else +1
        cands = df[(df["expiry"] == exp) & (df["instrument_type"] == opt) & (df["strike"].isin([atm + itm * k * step for k in range(0, 4)]))]
        if cands.empty:
            return None
        keys = [f"{self.info['opt_exch']}:{s}" for s in cands["tradingsymbol"]]
        prem = self.ltp(keys)
        now = datetime.now(IST); exp_dt = IST.localize(datetime.combine(exp, datetime.min.time()).replace(hour=15, minute=30))
        t = max((exp_dt - now).total_seconds(), 1800.0) / (365.0 * 86400)
        atm_row = cands[cands["strike"] == atm]
        atm_p = prem.get(f"{self.info['opt_exch']}:{atm_row.iloc[0]['tradingsymbol']}", 0.0) if len(atm_row) else 0.0
        sigma = _iv(atm_p, spot, atm, t, opt == "CE") if atm_p > 0 else 0.14
        best = None
        for _, r in cands.iterrows():
            _, _, nd1 = _bs(spot, float(r["strike"]), t, sigma); delta = nd1 if opt == "CE" else 1.0 - nd1
            p = prem.get(f"{self.info['opt_exch']}:{r['tradingsymbol']}", 0.0)
            gap = abs(delta - target_delta)
            if p > 0 and (best is None or gap < best["gap"]):
                best = {"tradingsymbol": r["tradingsymbol"], "exchange": self.info["opt_exch"], "strike": float(r["strike"]), "type": opt,
                        "expiry": str(exp), "lot_size": int(r["lot_size"]), "delta": round(delta, 3), "premium": p, "iv": round(sigma, 4), "gap": gap}
        return best

    # ---------------- orders
    def market_order(self, tradingsymbol: str, exchange: str, side: str, quantity: int, tag: str = "paengine") -> Tuple[Optional[str], str, float]:
        """Places a MIS market order and polls its status. Returns (order_id, status, average_price)."""
        try:
            oid = self.kite.place_order(variety=self.kite.VARIETY_REGULAR, exchange=exchange, tradingsymbol=tradingsymbol,
                                        transaction_type=side, quantity=int(quantity), product=self.kite.PRODUCT_MIS,
                                        order_type=self.kite.ORDER_TYPE_MARKET, validity=self.kite.VALIDITY_DAY, tag=tag)
        except Exception as e:
            log.critical(f"place_order failed {side} {quantity} {tradingsymbol}: {e}")
            return None, f"ERROR {e}", 0.0
        status, avg = "UNKNOWN", 0.0
        for _ in range(10):
            time.sleep(0.7)
            try:
                h = self.kite.order_history(str(oid))
                if h:
                    status = str(h[-1].get("status", "UNKNOWN")).upper(); avg = float(h[-1].get("average_price") or 0.0)
                    if status in ("COMPLETE", "REJECTED", "CANCELLED"):
                        break
            except Exception as e:
                log.warning(f"order_history: {e}")
        return str(oid), status, avg

    def balance(self) -> float:
        try:
            m = self.kite.margins("equity"); a = m.get("available", {})
            return float(a.get("live_balance") or a.get("cash") or 0.0)
        except Exception:
            return 0.0
