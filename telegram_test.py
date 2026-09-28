#!/usr/bin/env python
"""
Verify the Telegram credentials in .env by sending one test message through the same API the engine uses.

    python telegram_test.py

Setup (one time): create a bot with @BotFather (/newbot) -> TELEGRAM_BOT_TOKEN; message the bot once, then open
https://api.telegram.org/bot<TOKEN>/getUpdates and copy "chat":{"id":...} -> TELEGRAM_CHAT_ID. Put both in .env.
"""
import json, sys, urllib.parse, urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from algo_vpin_v2.config import CONFIG   # loads .env


def main():
    tok, chat = CONFIG.telegram.bot_token, CONFIG.telegram.chat_id
    if not tok or not chat:
        sys.exit("TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID are not set in .env")
    me = json.load(urllib.request.urlopen(f"https://api.telegram.org/bot{tok}/getMe", timeout=10))
    if not me.get("ok"):
        sys.exit(f"Token rejected: {me}")
    data = urllib.parse.urlencode({"chat_id": chat, "text": "Algo VPIN v2: Telegram test message. Alerts will arrive here.", "parse_mode": "HTML"}).encode()
    r = json.load(urllib.request.urlopen(f"https://api.telegram.org/bot{tok}/sendMessage", data=data, timeout=10))
    if not r.get("ok"):
        sys.exit(f"Send failed: {r.get('description')}")
    who = r["result"]["chat"].get("first_name") or r["result"]["chat"].get("title") or chat
    print(f"OK: bot @{me['result']['username']} delivered a test message to '{who}' (chat id {chat}). ENABLE_TELEGRAM={CONFIG.telegram.enabled}")


if __name__ == "__main__":
    main()
