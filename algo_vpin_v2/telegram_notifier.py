"""
Real-Time Telegram Notification Service for Algo VPIN v2.0
Dispatches instant trade entries, trailing stop locks, and exit alerts directly to mobile.
"""

import threading
import urllib.request
import urllib.parse
import json
import logging
import html
from typing import Optional

from .config import CONFIG

logger = logging.getLogger("algo_vpin_v2.telegram")


def send_telegram_async(message_html: str, bot_token: Optional[str] = None, chat_id: Optional[str] = None):
    """Dispatches Telegram notification in a background thread with zero latency to trading loop."""
    token = bot_token or CONFIG.telegram.bot_token
    cid = chat_id or CONFIG.telegram.chat_id
    enabled = CONFIG.telegram.enabled

    if not enabled or not token or not cid:
        return

    def _send():
        try:
            url = f"https://api.telegram.org/bot{token}/sendMessage"
            payload = json.dumps({
                "chat_id": cid,
                "text": message_html,
                "parse_mode": "HTML"
            }).encode("utf-8")
            req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=4)
        except Exception as e:
            logger.warning(f"[Telegram Notification Failed]: {e}")

    threading.Thread(target=_send, daemon=True).start()


def notify_trade_entry(symbol: str, action: str, quantity: int, entry_price: float, sl: float, tp: float, vpin: float, regime: str, total_capital: float, track_name: str = ""):
    """Sends formatted trade entry alert with clear track labelling and timestamp"""
    from datetime import datetime
    import pytz
    tz = pytz.timezone("Asia/Kolkata")
    time_str = datetime.now(tz).strftime("%H:%M:%S IST")

    sym_safe = html.escape(str(symbol))
    act_safe = html.escape(str(action))
    track_safe = html.escape(str(track_name))
    icon = "🟢" if ("CALL" in symbol or "CE" in symbol or action.upper() == "BUY") else "🔴"
    side_str = "BUY CALL" if ("CALL" in symbol or "CE" in symbol) else ("BUY PUT" if ("PUT" in symbol or "PE" in symbol) else f"BUY {act_safe}")
    track_header = f" [{track_safe}]" if track_safe else ""
    msg = (
        f"{icon} <b>[TRADE ENTRY TRIGGERED{track_header}]</b>\n\n"
        f"<b>Time:</b> {time_str}\n"
        f"<b>Contract:</b> {sym_safe}\n"
        f"<b>Action:</b> {side_str} ({quantity} Qty)\n"
        f"<b>Entry Premium:</b> ₹{entry_price:.2f} (Capital: ₹{total_capital:,.2f})\n"
        f"<b>Stop-Loss:</b> ₹{sl:.2f} | <b>Target:</b> ₹{tp:.2f}\n"
        f"<b>VPIN Regime:</b> {regime} (Toxicity: {vpin:.3f})\n"
        f"<b>Engine:</b> Algo VPIN v2.0 Live"
    )
    send_telegram_async(msg)


def notify_trailing_sl(symbol: str, peak_price: float, gain_pct: float, new_sl: float, tier_name: str, track_name: str = ""):
    """Sends formatted trailing stop update with track label and timestamp"""
    from datetime import datetime
    import pytz
    tz = pytz.timezone("Asia/Kolkata")
    time_str = datetime.now(tz).strftime("%H:%M:%S IST")

    sym_safe = html.escape(str(symbol))
    tier_safe = html.escape(str(tier_name)).upper()
    track_safe = html.escape(str(track_name))
    track_header = f" [{track_safe}]" if track_safe else ""
    msg = (
        f"🔒 <b>[TRAILING STOP LOCKED - {tier_safe}{track_header}]</b>\n\n"
        f"<b>Time:</b> {time_str}\n"
        f"<b>Contract:</b> {sym_safe}\n"
        f"<b>Peak Price:</b> ₹{peak_price:.2f} (+{gain_pct:.1f}%)\n"
        f"<b>New SL Floor:</b> ₹{new_sl:.2f}\n"
        f"<b>Status:</b> Profit Protected (Zero Downside Risk)"
    )
    send_telegram_async(msg)


def notify_trade_exit(symbol: str, exit_price: float, pnl_inr: float, pnl_pct: float, reason: str, day_pnl: float, track_name: str = ""):
    """Sends formatted trade exit and PnL report with track name and timestamp"""
    from datetime import datetime
    import pytz
    tz = pytz.timezone("Asia/Kolkata")
    time_str = datetime.now(tz).strftime("%H:%M:%S IST")

    sym_safe = html.escape(str(symbol))
    reason_safe = html.escape(str(reason))
    track_safe = html.escape(str(track_name))
    icon = "💰" if pnl_inr >= 0 else "🛑"
    status_tag = "PROFIT" if pnl_inr >= 0 else "STOP-LOSS / EXIT"
    track_header = f" [{track_safe}]" if track_safe else ""
    msg = (
        f"{icon} <b>[TRADE CLOSED - {status_tag}{track_header}]</b>\n\n"
        f"<b>Time:</b> {time_str}\n"
        f"<b>Contract:</b> {sym_safe}\n"
        f"<b>Exit Price:</b> ₹{exit_price:.2f}\n"
        f"<b>Trade Realized P&L:</b> <b>{pnl_inr:+.2f} INR ({pnl_pct:+.1f}%)</b>\n"
        f"<b>Exit Reason:</b> <code>{reason_safe}</code>\n"
        f"<b>Total Day Realized P&L:</b> <b>{day_pnl:+.2f} INR</b>"
    )
    send_telegram_async(msg)
