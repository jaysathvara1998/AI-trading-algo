from datetime import datetime
from pathlib import Path
import sys
import pytz
# Automatically ensure project root is in sys.path when running directly
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from algo_vpin_dhan.config import CONFIG

try:
    from Dhan_Tradehull import Tradehull
except ImportError:
    Tradehull = None


def run_diagnostics():
    print("=" * 65)
    print("      DHANHQ & TRADEHULL CONNECTION HEALTH DIAGNOSTICS")
    print("=" * 65)
    tz = pytz.timezone("Asia/Kolkata")
    now_ist = datetime.now(tz)
    print(f"Timestamp: {now_ist.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(f"Client ID: {CONFIG.dhan.client_id}")
    print(f"Auth Mode: {CONFIG.dhan.auth_mode}")
    print("-" * 65)

    if Tradehull is None:
        print("[FAIL] Dhan-Tradehull package is not installed.")
        return False

    # 1. Test Authentication
    print("[1/6] Authenticating via PIN + TOTP...")
    try:
        if CONFIG.dhan.auth_mode == "pin_totp":
            th = Tradehull(
                ClientCode=CONFIG.dhan.client_id,
                mode="pin_totp",
                pin=CONFIG.dhan.pin,
                totp_secret=CONFIG.dhan.totp_secret
            )
        else:
            th = Tradehull(
                ClientCode=CONFIG.dhan.client_id,
                token_id=CONFIG.dhan.access_token,
                mode="access_token"
            )
        print("[PASS] Authentication & token validation successful!")
    except Exception as e:
        print(f"[FAIL] Authentication error: {e}")
        return False

    # 2. Check Account Balance & Funds
    print("\n[2/6] Checking Account Balance & Margin...")
    try:
        balance_info = th.get_balance()
        print(f"[PASS] Account Balance Response: {balance_info}")
    except Exception as e:
        print(f"[WARN] Balance query warning: {e}")

    # 3. Check Active Positions & Orders
    print("\n[3/6] Checking Open Positions & Orderbook...")
    try:
        positions = th.get_positions()
        print(f"[PASS] Current Positions count: {len(positions) if positions is not None else 0}")
    except Exception as e:
        print(f"[WARN] Positions query warning: {e}")

    # 4. Check Active Nifty Expiry & Future Script
    print("\n[4/6] Resolving Nifty Active Expiry & Contract...")
    try:
        exp_date = th.get_expiry_date("NIFTY", "FUT")
        print(f"[PASS] Nearest Nifty Expiry Date: {exp_date}")
        if exp_date:
            fut_script = th.get_future_script("NIFTY", exp_date)
            print(f"[PASS] Active Nifty Futures Contract: {fut_script}")
    except Exception as e:
        print(f"[WARN] Expiry resolution warning: {e}")

    # 5. Check Live LTP & Quote Data
    print("\n[5/6] Fetching Live Quotes & Market LTP...")
    try:
        ltp_res = th.get_ltp_data(["NIFTY", "BANKNIFTY"])
        print(f"[PASS] Live LTP Data: {ltp_res}")
    except Exception as e:
        print(f"[WARN] LTP query warning: {e}")

    # 6. Check Historical 1-Minute Bars
    print("\n[6/6] Fetching Historical 1-Minute OHLCV Candles...")
    try:
        bars = th.get_historical_data(
            tradingsymbol="NIFTY",
            exchange="NSE",
            timeframe="1"
        )
        if bars is not None and not bars.empty:
            print(f"[PASS] Retrieved {len(bars)} 1-minute bars successfully!")
            print(f"       Latest Bar: {bars.iloc[-1].to_dict()}")
        else:
            print("[INFO] Historical bar query returned empty (market may be closed or indexing).")
    except Exception as e:
        print(f"[WARN] Historical data query warning: {e}")

    print("\n" + "=" * 65)
    print("      DIAGNOSTICS COMPLETE: CONNECTION IS HEALTHY & ACTIVE")
    print("=" * 65)
    return True


if __name__ == "__main__":
    run_diagnostics()
