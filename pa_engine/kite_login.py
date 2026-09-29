"""
Daily Zerodha Kite login:  python -m pa_engine login [--request-token XXXX]
Opens/prints the login URL, exchanges the request_token for an access token and writes
Dependencies/kite_token_<YYYY-MM-DD>.txt (gitignored), which the live runner picks up automatically.
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

from .env import ROOT, env, load_env


def main(request_token: str | None = None):
    load_env()
    from kiteconnect import KiteConnect
    key, secret = env("KITE_API_KEY"), env("KITE_API_SECRET")
    if not key or not secret:
        sys.exit("KITE_API_KEY / KITE_API_SECRET are not set in .env")
    kite = KiteConnect(api_key=key)
    if not request_token:
        print("\n1. Open this URL and log in to Kite:\n   " + kite.login_url())
        print("2. Copy the request_token from the redirect URL (...?request_token=XXXX&action=login&status=success)\n")
        request_token = input("request_token: ").strip()
    data = kite.generate_session(request_token.strip(), api_secret=secret)
    token = data["access_token"]; kite.set_access_token(token); prof = kite.profile()
    out_dir = ROOT / "Dependencies"; out_dir.mkdir(exist_ok=True)
    out = out_dir / f"kite_token_{date.today().isoformat()}.txt"
    out.write_text(f"{date.today().isoformat()}|{token}|{prof.get('user_id', '')}\n")
    print(f"Logged in as {prof.get('user_name', prof.get('user_id'))}. Token saved to {out}. Valid until ~06:00 tomorrow.")
