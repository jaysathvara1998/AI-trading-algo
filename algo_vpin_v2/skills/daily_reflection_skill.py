"""
Daily Reflection & Continuous Learning Skill for Algo VPIN v2.0 (AURA-v2)
Integrates:
1. Quantitative Post-Market Audit (Win/Loss Clustering, Chop & Bleed Detection)
2. Gemini AI Agent Reflection (Deep qualitative institutional reasoning & tactical playbook synthesis)
3. Adaptive Hyperparameter Tuning (Dynamic ML threshold calibration)
4. Knowledge Persistence in `data/trade_learnings.json`
"""

import os
import json
import logging
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
import pytz

from .base_skill import BaseTradingSkill, SkillResult
from ..config import CONFIG

logger = logging.getLogger("algo_vpin_v2.skills.daily_reflection")


class DailyReflectionSkill(BaseTradingSkill):
    """
    Continuous Self-Learning & Experience Replay Skill.
    Audits daily trades at 15:30 IST and prepares pre-market briefings for 09:00 AM.
    """
    def __init__(self, data_dir: Optional[Path] = None, gemini_api_key: Optional[str] = None):
        super().__init__(name="DailyReflectionSkill")
        self.tz = pytz.timezone("Asia/Kolkata")
        self.data_dir = data_dir or (Path(__file__).resolve().parent.parent / "data")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.learnings_file = self.data_dir / "trade_learnings.json"
        self.gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY", "") or getattr(getattr(CONFIG, "gemini", None), "api_key", "")

    def evaluate(self, *args, **kwargs) -> SkillResult:
        """Standard skill evaluation returning current adapted market rules"""
        learnings = self.load_latest_learnings()
        return SkillResult(
            is_favorable=True,
            confidence=0.90,
            signal="ADAPTIVE_RULES_ACTIVE",
            reason=learnings.get("summary_takeaway", "Standard rules active"),
            metadata=learnings.get("adapted_rules", {})
        )

    def load_latest_learnings(self) -> Dict[str, Any]:
        """Loads persistent learnings from JSON disk ledger"""
        if self.learnings_file.exists():
            try:
                with open(self.learnings_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.debug(f"[DailyReflection] Could not load learnings file: {e}")
        return {
            "last_audit_date": "",
            "summary_takeaway": "Baseline risk rules active.",
            "adapted_rules": {
                "max_daily_trades": 6,
                "xgb_min_prob_threshold": 0.72,
                "instant_sl_cut": True,
                "avoid_chop_zone": False
            },
            "ai_reflection": ""
        }

    def run_end_of_day_audit(self, target_date: Optional[str] = None) -> Dict[str, Any]:
        """
        Runs comprehensive post-market quantitative audit + Gemini AI reflection.
        """
        today_str = target_date or datetime.now(self.tz).strftime("%Y-%m-%d")
        trades_path = self.data_dir / "trades_history.csv"
        dfs = []
        if trades_path.exists():
            try:
                dfs.append(pd.read_csv(trades_path))
            except Exception as e:
                logger.debug(f"[DailyReflection] Error reading main trades file: {e}")

        # Also search for any archived trade files for today
        for f in self.data_dir.glob("trades_history_*.csv"):
            try:
                dfs.append(pd.read_csv(f))
            except Exception as e:
                logger.debug(f"[DailyReflection] Error reading archive file {f}: {e}")

        if not dfs:
            logger.warning("[DailyReflection] No trades history files found for audit.")
            return {}

        try:
            df = pd.concat(dfs, ignore_index=True)
            if "trade_id" in df.columns:
                df = df.drop_duplicates(subset=["trade_id"])
            elif "entry_timestamp" in df.columns and "exit_timestamp" in df.columns:
                df = df.drop_duplicates(subset=["entry_timestamp", "exit_timestamp"])
            
            df["exit_dt"] = pd.to_datetime(df["exit_timestamp"])
            day_trades = df[df["exit_dt"].dt.strftime("%Y-%m-%d") == today_str].copy()

            if day_trades.empty:
                logger.info(f"[DailyReflection] No trades found for {today_str}. Audit skipped.")
                return {}

            total_trades = len(day_trades)
            wins = day_trades[day_trades["pnl_inr"] > 0]
            losses = day_trades[day_trades["pnl_inr"] <= 0]
            win_count = len(wins)
            loss_count = len(losses)
            win_rate = (win_count / total_trades) if total_trades > 0 else 0.0
            net_pnl = float(day_trades["pnl_inr"].sum())

            # Diagnosis: Categorize mistakes
            chop_losses = 0
            bleed_losses = 0
            reversal_losses = 0

            for _, r in losses.iterrows():
                pnl_pct = float(r.get("pnl_pct", 0.0))
                reason = str(r.get("exit_reason", ""))
                if pnl_pct < -20.0 or "DISASTER" in reason.upper():
                    bleed_losses += 1
                elif "REVERSAL" in reason.upper():
                    reversal_losses += 1
                else:
                    chop_losses += 1

            # Adaptive rule adjustments for tomorrow
            adapted_max_trades = 12 if win_rate >= 0.50 else 10
            adapted_xgb_threshold = 0.72 if win_rate >= 0.70 else 0.75

            quant_summary = (
                f"Date: {today_str} | Trades: {total_trades} | Wins: {win_count} | Losses: {loss_count} | "
                f"Win Rate: {win_rate*100:.1f}% | Net P&L: INR {net_pnl:+,.2f} | "
                f"Mistakes Breakdown: [Chop: {chop_losses}, Bleed: {bleed_losses}, Reversal: {reversal_losses}]"
            )
            logger.info(f"[DailyReflection] Quant Audit Result -> {quant_summary}")

            # AI Agent (Gemini) Deep Reflection
            ai_commentary = self._generate_gemini_reflection(
                date_str=today_str,
                total_trades=total_trades,
                win_count=win_count,
                loss_count=loss_count,
                win_rate=win_rate,
                net_pnl=net_pnl,
                day_trades_df=day_trades
            )

            audit_data = {
                "last_audit_date": today_str,
                "timestamp": datetime.now(self.tz).isoformat(),
                "total_trades": total_trades,
                "win_rate": round(win_rate, 4),
                "net_pnl_inr": round(net_pnl, 2),
                "quant_metrics": {
                    "wins": win_count,
                    "losses": loss_count,
                    "chop_losses": chop_losses,
                    "bleed_losses": bleed_losses,
                    "reversal_losses": reversal_losses,
                    "best_trade_pnl": float(day_trades["pnl_inr"].max()),
                    "worst_trade_pnl": float(day_trades["pnl_inr"].min())
                },
                "adapted_rules": {
                    "max_daily_trades": adapted_max_trades,
                    "xgb_min_prob_threshold": adapted_xgb_threshold,
                    "instant_sl_cut": True,
                    "avoid_chop_zone": (chop_losses >= 2)
                },
                "summary_takeaway": f"Win Rate: {win_rate*100:.0f}% | Net: INR {net_pnl:+,.0f}. Max Trades: {adapted_max_trades}, XGB Min Conf: {adapted_xgb_threshold*100:.0f}%.",
                "ai_reflection": ai_commentary
            }

            with open(self.learnings_file, "w", encoding="utf-8") as f:
                json.dump(audit_data, f, indent=2)

            logger.info(f"[DailyReflection] Trade learnings successfully persisted to {self.learnings_file}")
            return audit_data

        except Exception as e:
            logger.error(f"[DailyReflection] Error in end-of-day audit: {e}")
            return {}

    def _generate_gemini_reflection(
        self,
        date_str: str,
        total_trades: int,
        win_count: int,
        loss_count: int,
        win_rate: float,
        net_pnl: float,
        day_trades_df: pd.DataFrame
    ) -> str:
        """
        Sends structured trade telemetry to Gemini AI for qualitative institutional reflection.
        Falls back cleanly to algorithmic heuristic summary if offline or no API key.
        """
        # Format trade rows for context
        trades_table = []
        for _, r in day_trades_df.iterrows():
            entry_t = str(r.get("entry_time", ""))[-14:-6] if r.get("entry_time") else ""
            exit_t = str(r.get("exit_timestamp", ""))[-14:-6] if r.get("exit_timestamp") else ""
            sym = str(r.get("symbol", ""))
            pnl = float(r.get("pnl_inr", 0.0))
            reason = str(r.get("exit_reason", ""))[:40]
            trades_table.append(f"[{entry_t}->{exit_t}] {sym} | PnL: INR {pnl:+,.1f} | Reason: {reason}")

        trades_str = "\n".join(trades_table)

        if not self.gemini_api_key:
            # Algorithmic fallback
            return (
                f"[QUANT REFLECTION] Session {date_str} completed with {win_count}/{total_trades} wins ({win_rate*100:.1f}%). "
                f"Net P&L: INR {net_pnl:+,.2f}. Runners generated strong upside while choppy entries were disciplined."
            )

        prompt = (
            f"You are an Elite Institutional Quant & Head of Risk for an Indian Options Algorithmic Trading Desk.\n"
            f"Review today's ({date_str}) live intraday options trading performance:\n\n"
            f"Session Metrics:\n"
            f"- Total Closed Trades: {total_trades}\n"
            f"- Win Count: {win_count} | Loss Count: {loss_count}\n"
            f"- Win Rate: {win_rate*100:.1f}%\n"
            f"- Net Realized P&L: INR {net_pnl:,.2f}\n\n"
            f"Detailed Trades:\n{trades_str}\n\n"
            f"Provide a concise, razor-sharp 3-bullet point institutional critique:\n"
            f"1. Key Strength (What generated the highest profit)\n"
            f"2. Key Vulnerability / Mistake (What caused unnecessary drawdown)\n"
            f"3. Concrete Actionable Rule for Tomorrow's Pre-Market Plan"
        )

        try:
            model_name = getattr(getattr(CONFIG, "gemini", None), "model", "gemini-2.5-flash")
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={self.gemini_api_key}"
            payload = json.dumps({
                "contents": [{"parts": [{"text": prompt}]}]
            }).encode("utf-8")
            req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                candidates = data.get("candidates", [])
                if candidates:
                    text = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                    return text.strip()
        except Exception as e:
            logger.debug(f"[DailyReflection] Gemini API call notice: {e}")

        return (
            f"[QUANT REFLECTION] Session {date_str} completed with {win_count}/{total_trades} wins ({win_rate*100:.1f}%). "
            f"Net P&L: INR {net_pnl:+,.2f}. Focus tomorrow on strictly waiting for 5-min HTF confirmation."
        )

    def get_morning_briefing(self) -> str:
        """
        Produces concise briefing string loaded into the Pre-Market Journal & HUD at 09:00 AM.
        """
        learnings = self.load_latest_learnings()
        last_date = learnings.get("last_audit_date", "None")
        win_rate = learnings.get("win_rate", 0.5) * 100
        net_pnl = learnings.get("net_pnl_inr", 0.0)
        rules = learnings.get("adapted_rules", {})
        max_trades = rules.get("max_daily_trades", 6)
        min_conf = rules.get("xgb_min_prob_threshold", 0.72) * 100

        return (
            f"[EXPERIENCE MEMORY | Prior Session {last_date}]\n"
            f"  * Performance: Win Rate {win_rate:.0f}% | Net P&L: INR {net_pnl:+,.2f}\n"
            f"  * Adaptive Rules: Max Trades={max_trades} | Min ML Conf={min_conf:.0f}% | Instant SL Cut=ON"
        )

    def generate_premarket_strategic_plan(
        self,
        symbol: str,
        current_spot: float,
        pdh: float,
        pdl: float,
        pdc: float,
        global_sentiment: str,
        heavyweight_bias: str
    ) -> str:
        """
        Synthesizes today's key levels with past learned lessons into a sharp pre-market trade posture.
        """
        learnings = self.load_latest_learnings()
        past_critique = learnings.get("ai_reflection", "")

        if not self.gemini_api_key:
            return (
                f"Opening Levels: PDH={pdh:.1f}, PDL={pdl:.1f}, PDC={pdc:.1f}. "
                f"Global: {global_sentiment} | Breadth: {heavyweight_bias}. "
                f"Tactical Plan: Wait for 5-min candle confirmation near structural levels before entry."
            )

        prompt = (
            f"You are the Head Trader for an Indian Index Options Desk trading {symbol}.\n"
            f"Generate a concise 2-sentence Pre-Market Trading Plan for today's market open based on:\n"
            f"- Spot / PDC: {current_spot:.1f} / {pdc:.1f}\n"
            f"- Key Levels: PDH = {pdh:.1f}, PDL = {pdl:.1f}\n"
            f"- Market Cues: Global Sentiment = {global_sentiment}, Heavyweights Breadth = {heavyweight_bias}\n"
            f"- Past Lessons to enforce: {past_critique[:250]}\n\n"
            f"Output only 2 sharp, actionable sentences on entry focus and risk guardrails."
        )

        try:
            model_name = getattr(getattr(CONFIG, "gemini", None), "model", "gemini-2.5-flash")
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={self.gemini_api_key}"
            payload = json.dumps({"contents": [{"parts": [{"text": prompt}]}]}).encode("utf-8")
            req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=12) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                candidates = data.get("candidates", [])
                if candidates:
                    return candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "").strip()
        except Exception as e:
            logger.debug(f"[DailyReflection] Pre-market plan generation notice: {e}")

        return (
            f"Opening Levels: PDH={pdh:.1f}, PDL={pdl:.1f}, PDC={pdc:.1f}. "
            f"Global: {global_sentiment} | Breadth: {heavyweight_bias}. "
            f"Tactical Plan: Trade only when 5-min HTF aligns with momentum. Enforce strict 10pt stop loss."
        )
