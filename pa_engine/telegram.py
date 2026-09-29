"""Telegram alerts (plain urllib). Silent no-op when credentials are missing; never raises into the engine."""
from __future__ import annotations

import json
import logging
import urllib.parse
import urllib.request

from .env import env, env_bool

log = logging.getLogger("pa_engine.telegram")


class Telegram:
    def __init__(self):
        self.token = env("TELEGRAM_BOT_TOKEN"); self.chat = env("TELEGRAM_CHAT_ID")
        self.enabled = env_bool("ENABLE_TELEGRAM", True) and bool(self.token and self.chat)

    def send(self, text: str) -> bool:
        if not self.enabled:
            return False
        try:
            data = urllib.parse.urlencode({"chat_id": self.chat, "text": text, "parse_mode": "HTML"}).encode()
            r = json.load(urllib.request.urlopen(f"https://api.telegram.org/bot{self.token}/sendMessage", data=data, timeout=10))
            return bool(r.get("ok"))
        except Exception as e:
            log.warning(f"telegram send failed: {e}")
            return False

    def test(self) -> str:
        if not self.token or not self.chat:
            return "TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID are not set in .env"
        me = json.load(urllib.request.urlopen(f"https://api.telegram.org/bot{self.token}/getMe", timeout=10))
        if not me.get("ok"):
            return f"token rejected: {me}"
        ok = self.send("pa_engine v3: Telegram test message. Alerts will arrive here.")
        return f"OK: bot @{me['result']['username']} -> chat {self.chat}" if ok else "send failed"
