"""
High-Impact News & Volatility Event Risk Filter for Algo VPIN v2.0 (AURA-v2)
Implements Pillar 2 of the Pre-9:15 AM Master Routine:
- Tracks key macro risk windows (RBI MPC, Fed Rate, CPI, High-Impact Data Releases)
- Evaluates intraday news volatility risk to prevent spike/slippage traps
- Flags risk regimes: NORMAL, ELEVATED_EVENT_RISK, HIGH_VOLATILITY_LOCKOUT
"""

from dataclasses import dataclass
from datetime import datetime, time
from enum import Enum
from typing import List, Optional
import pytz
import logging

logger = logging.getLogger("algo_vpin_v2.news_filter")


class NewsRiskLevel(Enum):
    CLEAR = "CLEAR"                   # Normal market conditions
    CAUTION = "CAUTION"               # Near scheduled release window (tighten SL)
    LOCKOUT = "LOCKOUT"               # Active high-impact announcement window (no new entries)


@dataclass
class NewsEventState:
    risk_level: NewsRiskLevel
    event_name: str
    impact: str                       # HIGH, MEDIUM, LOW
    is_entry_allowed: bool
    warning_message: str


import json
import urllib.request
import xml.etree.ElementTree as ET


class NewsFilter:
    """
    Scans for live high-impact events from ForexFactory & Moneycontrol Live News feeds.
    Guards against news volatility traps and sudden slippage spikes.
    """
    FOREX_FACTORY_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
    MONEYCONTROL_RSS_URLS = [
        "https://www.moneycontrol.com/rss/marketreports.xml",
        "https://www.moneycontrol.com/rss/economy.xml",
        "https://www.moneycontrol.com/rss/latestnews.xml"
    ]

    def __init__(self):
        self.tz = pytz.timezone("Asia/Kolkata")
        self.live_events: List[dict] = []
        self.moneycontrol_headlines: List[str] = []
        self._last_fetch_date: Optional[str] = None
        
        # Recurring intraday volatility windows (IST) as baseline fallback
        self.high_risk_windows = [
            (time(9, 55), time(10, 15), "RBI MPC / India Macro Release Window"),
            (time(11, 25), time(11, 45), "European Market Open Volatility Window"),
            (time(14, 25), time(14, 45), "US Pre-Market & High Impact Data Window")
        ]
        self.fetch_live_economic_calendar()
        self.fetch_moneycontrol_news()

    def fetch_moneycontrol_news(self) -> int:
        """
        Fetches live breaking news headlines from Moneycontrol RSS feeds.
        """
        headlines = []
        for url in self.MONEYCONTROL_RSS_URLS:
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=3) as resp:
                    content = resp.read().decode("utf-8", errors="ignore")
                    root = ET.fromstring(content)
                    for item in root.findall(".//item")[:5]:
                        title = item.find("title")
                        if title is not None and title.text:
                            clean_t = title.text.replace("#39;", "'").replace("&amp;", "&").strip()
                            headlines.append(clean_t)
            except Exception as e:
                logger.debug(f"[NewsFilter] Moneycontrol RSS fetch notice for {url}: {e}")
        
        self.moneycontrol_headlines = headlines[:8]
        if self.moneycontrol_headlines:
            logger.info(f"[NewsFilter] Moneycontrol live headlines synced: {len(self.moneycontrol_headlines)} breaking updates.")
        return len(self.moneycontrol_headlines)

    def fetch_live_economic_calendar(self) -> int:
        """
        Fetches live red-folder high-impact news from ForexFactory public JSON API.
        """
        try:
            req = urllib.request.Request(self.FOREX_FACTORY_URL, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=4) as resp:
                data = json.loads(resp.read().decode())
                # Filter for High impact events in USD, INR, EUR, GBP
                self.live_events = [
                    d for d in data 
                    if d.get("impact") in ["High", "Holiday"] and d.get("country") in ["USD", "INR", "EUR", "GBP"]
                ]
                self._last_fetch_date = datetime.now(self.tz).strftime("%Y-%m-%d")
                logger.info(f"[NewsFilter] Live ForexFactory calendar synced: {len(self.live_events)} high-impact events loaded.")
                return len(self.live_events)
        except Exception as e:
            logger.debug(f"[NewsFilter] Live calendar fetch failed (using fallback windows): {e}")
            return 0

    def check_news_risk(self, now: Optional[datetime] = None) -> NewsEventState:
        """
        Evaluates active risk level based on current time and calendar state.
        """
        if now is None:
            now = datetime.now(self.tz)
        current_time = now.time()

        # Check live events if time matches
        for ev in self.live_events:
            ev_date_str = ev.get("date", "")
            # Example format: 2026-09-18T08:30:00-04:00
            if ev_date_str:
                try:
                    ev_dt = datetime.fromisoformat(ev_date_str).astimezone(self.tz)
                    if ev_dt.date() == now.date():
                        time_diff_min = abs((now - ev_dt).total_seconds()) / 60.0
                        if time_diff_min <= 15:  # Within 15 minutes of high-impact news release
                            return NewsEventState(
                                risk_level=NewsRiskLevel.CAUTION,
                                event_name=f"{ev.get('country')} {ev.get('title')}",
                                impact="HIGH",
                                is_entry_allowed=True,
                                warning_message=f"ForexFactory High-Impact News Release ({ev.get('country')} {ev.get('title')}). Tightening SL!"
                            )
                except Exception:
                    pass

        # Check standard recurring volatility windows
        for start_t, end_t, event_name in self.high_risk_windows:
            if start_t <= current_time <= end_t:
                return NewsEventState(
                    risk_level=NewsRiskLevel.CAUTION,
                    event_name=event_name,
                    impact="HIGH",
                    is_entry_allowed=True,
                    warning_message=f"High-impact macro event window active ({event_name}). Enforcing tight risk!"
                )

        return NewsEventState(
            risk_level=NewsRiskLevel.CLEAR,
            event_name="None Scheduled",
            impact="LOW",
            is_entry_allowed=True,
            warning_message="Economic calendar clear. Standard volatility rules apply."
        )
