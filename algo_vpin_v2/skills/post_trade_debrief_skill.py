"""
Post-Trade Debrief & Thesis Validation Skill
Inspired by marian2js/trading-skills (post-trade-debrief, thesis-validation)
Classifies trades into Process vs Outcome categories to eliminate hindsight bias and refine strategy execution.
"""

import logging
from typing import Any, Dict, List, Optional
import pandas as pd
from .base_skill import BaseTradingSkill, SkillResult

logger = logging.getLogger("algo_vpin_v2.skills.post_trade_debrief")


class PostTradeDebriefSkill(BaseTradingSkill):
    """
    Institutional Post-Mortem & Trade Quality Auditor.
    Separates 'Process Quality' from 'Market Luck' using a 4-Quadrant Matrix:
    1. [PROCESS WIN] High-conviction entry with disciplined trailing exit (True Success)
    2. [ACCEPTABLE LOSS] Plan followed, invalidated cleanly at SL (Cost of doing business)
    3. [PROCESS MISTAKE] Entry in low-momentum chop or held past planned stop (Target for elimination)
    4. [DANGEROUS WIN] Ill-disciplined entry that got bailed out by luck (False positive)
    """
    def __init__(self):
        super().__init__(name="PostTradeDebriefSkill")

    def evaluate(
        self,
        trade_record: Optional[Dict[str, Any]] = None,
        all_trades_df: Optional[pd.DataFrame] = None,
        **kwargs
    ) -> SkillResult:
        if trade_record is None and (all_trades_df is None or all_trades_df.empty):
            return SkillResult(
                is_favorable=True,
                confidence=0.50,
                signal="NO_TRADES_TO_AUDIT",
                reason="PostTradeDebrief: No closed trades provided for audit."
            )

        if trade_record is not None:
            category, explanation = self._audit_single_trade(trade_record)
            return SkillResult(
                is_favorable=(category in ("PROCESS_WIN", "ACCEPTABLE_LOSS")),
                confidence=0.85,
                signal=category,
                reason=explanation,
                metadata={"category": category}
            )

        # Audit full batch
        summary_counts = {"PROCESS_WIN": 0, "ACCEPTABLE_LOSS": 0, "PROCESS_MISTAKE": 0, "DANGEROUS_WIN": 0}
        for _, row in all_trades_df.iterrows():
            cat, _ = self._audit_single_trade(row.to_dict())
            summary_counts[cat] = summary_counts.get(cat, 0) + 1

        total = len(all_trades_df)
        process_score = (summary_counts["PROCESS_WIN"] + summary_counts["ACCEPTABLE_LOSS"]) / total if total > 0 else 1.0

        return SkillResult(
            is_favorable=(process_score >= 0.70),
            confidence=round(process_score, 2),
            signal="BATCH_AUDIT_COMPLETE",
            reason=f"Process Quality Score: {process_score*100:.0f}% | Wins: {summary_counts['PROCESS_WIN']}, Acceptable Losses: {summary_counts['ACCEPTABLE_LOSS']}, Process Mistakes: {summary_counts['PROCESS_MISTAKE']}",
            metadata=summary_counts
        )

    def _audit_single_trade(self, t: Dict[str, Any]) -> tuple:
        pnl = float(t.get("pnl_inr", 0.0))
        pnl_pct = float(t.get("pnl_pct", 0.0))
        reason = str(t.get("exit_reason", "")).upper()

        if pnl > 0:
            if "TRAILING" in reason or "TARGET" in reason or "RUNNER" in reason:
                return "PROCESS_WIN", f"Disciplined winner: captured profit via {reason} (+INR {pnl:.1f})."
            else:
                return "DANGEROUS_WIN", f"Discretionary/premature exit on winner (+INR {pnl:.1f})."
        else:
            if "EMERGENCY_REVERSAL" in reason or ("TRAILING" in reason and pnl_pct > -10.0):
                return "ACCEPTABLE_LOSS", f"Disciplined defense: cut cleanly on reversal/trail ({reason}, loss {pnl_pct:.1f}%)."
            elif "STOP_LOSS_HIT" in reason and pnl_pct > -25.0:
                return "ACCEPTABLE_LOSS", f"Standard invalidation: stopped out within initial risk floor ({pnl_pct:.1f}%)."
            else:
                return "PROCESS_MISTAKE", f"Process mistake: severe loss ({pnl_pct:.1f}% | {reason}) due to chop or missed cut."
