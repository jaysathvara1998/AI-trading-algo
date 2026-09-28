#!/usr/bin/env python
"""
Daily Zerodha Kite login. Run once each morning before starting the bot:

    python kite_login.py                # prints the login URL, asks for the request_token
    python kite_login.py <request_token>

Steps: open the printed URL, log in to Kite, and copy the `request_token` from the redirect URL
(...?request_token=XXXX&action=login&status=success). The generated access token is written to
Dependencies/kite_token_<YYYY-MM-DD>.txt (gitignored) and picked up automatically by the engine.
"""
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from algo_vpin_v2.config import CONFIG   # loads .env
from kiteconnect import KiteConnect


def main():
    kc = CONFIG.kite
    if not kc.api_key or not kc.api_secret:
        sys.exit("KITE_API_KEY / KITE_API_SECRET are not set in .env")
    kite = KiteConnect(api_key=kc.api_key)
    if len(sys.argv) > 1:
        request_token = sys.argv[1].strip()
    else:
        print("\n1. Open this URL and log in to Kite:\n   " + kite.login_url())
        print("2. After login you are redirected to your app's redirect URL. Copy the request_token from it.\n")
        request_token = input("request_token: ").strip()
    if not request_token:
        sys.exit("No request_token given")
    data = kite.generate_session(request_token, api_secret=kc.api_secret)
    token = data["access_token"]
    kite.set_access_token(token)
    prof = kite.profile()
    out_dir = ROOT / "Dependencies"; out_dir.mkdir(exist_ok=True)
    out = out_dir / f"kite_token_{date.today().isoformat()}.txt"
    out.write_text(f"{date.today().isoformat()}|{token}|{prof.get('user_id', '')}\n")
    print(f"Logged in as {prof.get('user_name', prof.get('user_id'))}. Token saved to {out}. Valid until ~06:00 tomorrow.")


if __name__ == "__main__":
    main()
