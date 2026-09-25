"""
Humanoid Cognitive AI Trading Agent for Algo VPIN v2.0
Emulates a seasoned human quantitative trader with cognitive reasoning,
visual perception, time-of-day tactical adaptability, and human-like thought streams.
"""

from dataclasses import dataclass
from datetime import datetime, time
from enum import Enum
from typing import Optional, Dict, Any, List, Tuple
import logging
import pytz

from .config import ExecutionMode, TradeStrategyMode

logger = logging.getLogger("algo_vpin_v2.humanoid")



class MarketPhase(Enum):
    PRE_OPEN = "PRE_OPEN"
    MORNING_EXPANSION = "MORNING_EXPANSION"        # 09:15 - 10:30 (High Momentum & Trend Follow)
    MIDDAY_TACTICAL_SCALP = "MIDDAY_TACTICAL_SCALP" # 10:30 - 13:30 (Mean-Reversion, Range Bounces & Quick Locks)
    AFTERNOON_SURGE = "AFTERNOON_SURGE"            # 13:30 - 15:15 (Breakouts, Institutional Expansion)
    POST_CLOSE = "POST_CLOSE"                      # 15:15 - 15:30 (Square-off & Session Reflection)


@dataclass
class TacticalPlaybook:
    phase: MarketPhase
    name: str
    target_profit_pct: float
    breakeven_trigger_pct: float
    trailing_step_pct: float
    max_hold_bars: int
    focus_mode: str
    tactical_rationale: str


class HumanoidTraderAgent:
    """
    Humanoid Autonomous Trading Agent.
    Synthesizes multi-brain neural signals into human-like consciousness,
    market phase awareness, and tactical agility.
    """

    def __init__(self, agent_name: str = "AURA-v2 (Humanoid Quant)"):
        self.agent_name = agent_name
        self.tz = pytz.timezone("Asia/Kolkata")
        self.last_thought: str = "Warming up sensory cortex and scanning institutional order flow..."
        self.total_cognitive_cycles: int = 0
        self.insights_history: List[str] = []

    def get_current_market_phase(self, current_time: Optional[datetime] = None) -> MarketPhase:
        now_ist = current_time or datetime.now(self.tz)
        current_t = now_ist.time()

        if current_t < time(9, 15):
            return MarketPhase.PRE_OPEN
        elif time(9, 15) <= current_t < time(10, 30):
            return MarketPhase.MORNING_EXPANSION
        elif time(10, 30) <= current_t < time(13, 30):
            return MarketPhase.MIDDAY_TACTICAL_SCALP
        elif time(13, 30) <= current_t < time(15, 15):
            return MarketPhase.AFTERNOON_SURGE
        else:
            return MarketPhase.POST_CLOSE

    def get_tactical_playbook(self, current_time: Optional[datetime] = None) -> TacticalPlaybook:
        phase = self.get_current_market_phase(current_time)

        if phase == MarketPhase.MORNING_EXPANSION:
            return TacticalPlaybook(
                phase=phase,
                name="MORNING MOMENTUM RUNNER",
                target_profit_pct=0.30,       # Aim for +30% expansion
                breakeven_trigger_pct=0.12,   # Move SL to breakeven at +12%
                trailing_step_pct=0.08,       # Trail every +8%
                max_hold_bars=35,
                focus_mode="Trend Following & Breakout Expansion",
                tactical_rationale="Morning volatility is high. Give trades room to breathe and capture the macro trend."
            )
        elif phase == MarketPhase.MIDDAY_TACTICAL_SCALP:
            return TacticalPlaybook(
                phase=phase,
                name="MIDDAY TACTICAL SCALPER",
                target_profit_pct=0.15,       # Fast +15% profit capture
                breakeven_trigger_pct=0.06,   # Lock breakeven early at +6% to beat Theta decay
                trailing_step_pct=0.04,       # Tight trailing stop
                max_hold_bars=15,
                focus_mode="Mean-Reversion & Quick Profit Lock",
                tactical_rationale="Midday range-bound consolidation. Scalp quick pops and lock gains before Theta decay bites."
            )
        elif phase == MarketPhase.AFTERNOON_SURGE:
            return TacticalPlaybook(
                phase=phase,
                name="AFTERNOON SURGE RUNNER",
                target_profit_pct=0.35,       # Aim for +35% expansion
                breakeven_trigger_pct=0.10,   # Breakeven at +10%
                trailing_step_pct=0.06,       # Dynamic trail
                max_hold_bars=25,
                focus_mode="Short-Covering & Breakout Continuation",
                tactical_rationale="Institutional afternoon volume is active. Ride aggressive momentum breakouts."
            )
        elif phase == MarketPhase.PRE_OPEN:
            return TacticalPlaybook(
                phase=phase,
                name="PRE-MARKET LEVEL SCANNER",
                target_profit_pct=0.0,
                breakeven_trigger_pct=0.0,
                trailing_step_pct=0.0,
                max_hold_bars=0,
                focus_mode="Levels Journaling & Global Bias Mapping",
                tactical_rationale="Analyzing GIFT Nifty delta, Asian markets, economic calendar, and PDH/PDL equilibrium."
            )
        else:
            return TacticalPlaybook(
                phase=phase,
                name="INTRADAY SQUARE-OFF / OBSERVATION",
                target_profit_pct=0.05,
                breakeven_trigger_pct=0.02,
                trailing_step_pct=0.02,
                max_hold_bars=5,
                focus_mode="Capital Preservation",
                tactical_rationale="Market close approaching. Guard open profits and exit cleanly."
            )

    def evaluate_execution_mode(
        self,
        ensemble_dec: Any,
        heavyweight_st: Any,
        vpin_res: Any,
        smc_st: Any = None,
        configured_mode: ExecutionMode = ExecutionMode.AUTO_BRAIN_SELECT
    ) -> Tuple[TradeStrategyMode, str]:
        """
        AI Dynamic Brain Mode Selector:
        Evaluates whether an incoming setup qualifies for INSTITUTIONAL_SWING or LIGHTNING_SCALPER.
        """
        if configured_mode == ExecutionMode.SCALPER_ONLY:
            return TradeStrategyMode.SCALPER, "Enforced: Pure Scalper Mode (1:1.5 RR + 0.5R Trail)"
        if configured_mode == ExecutionMode.SWING_ONLY:
            return TradeStrategyMode.INSTITUTIONAL_SWING, "Enforced: Institutional Swing Runner"

        # AI Dynamic Assessment
        is_unanimous = getattr(ensemble_dec, 'is_unanimous', False) or (getattr(ensemble_dec, 'xgb_confidence', 0.5) >= 0.80)
        hw_aligned = False
        if heavyweight_st:
            adv = getattr(heavyweight_st, 'advances', 0)
            dec = getattr(heavyweight_st, 'declines', 0)
            total = max(1, adv + dec)
            ratio = max(adv, dec) / total
            div_sig = getattr(heavyweight_st, 'divergence_signal', "") or ""
            has_trap = "DIVERGENCE_TRAP" in div_sig
            hw_aligned = (ratio >= 0.70) and not has_trap

        vpin_high = (getattr(vpin_res, 'vpin', 0.0) >= 0.25)

        if is_unanimous and hw_aligned and vpin_high:
            return TradeStrategyMode.INSTITUTIONAL_SWING, "Brain Consensus: High Institutional Conviction -> INSTITUTIONAL_SWING (Multi-Target Runner)"
        else:
            return TradeStrategyMode.SCALPER, "Brain Consensus: Standard Momentum Breakout -> LIGHTNING_SCALPER (1:1.5 RR + 0.5R Step Trail)"

    def synthesize_thought(
        self,
        curr_price: float,
        vpin_res: Any,
        garch_res: Any,
        ensemble_dec: Any,
        macro_st: Any = None,
        smc_st: Any = None,
        global_st: Any = None,
        news_st: Any = None,
        heavyweight_st: Any = None,
        pattern_st: Any = None,
        vision_prediction: int = 0,
        vision_confidence: float = 0.0,
        position_side_str: str = "FLAT",
        execution_mode_desc: str = ""
    ) -> str:
        """
        Synthesizes all multi-brain inputs, SMC concepts, and Classical Chart Patterns (Fidelity/CFI/SRCC).
        """
        self.total_cognitive_cycles += 1
        playbook = self.get_tactical_playbook()
        
        # Sensory Breakdown
        vision_desc = "Neutral/Scanning"
        if vision_prediction == 1:
            vision_desc = f"Bullish Pattern ({vision_confidence*100:.0f}% confidence)"
        elif vision_prediction == -1:
            vision_desc = f"Bearish Pattern ({vision_confidence*100:.0f}% confidence)"

        svm_str = "BULLISH" if ensemble_dec.svm_prediction == 1 else ("BEARISH" if ensemble_dec.svm_prediction == -1 else "NEUTRAL")
        xgb_str = f"BULLISH ({ensemble_dec.xgb_confidence*100:.0f}%)" if ensemble_dec.xgb_prediction == 1 else (f"BEARISH ({ensemble_dec.xgb_confidence*100:.0f}%)" if ensemble_dec.xgb_prediction == -1 else "NEUTRAL")
        
        thoughts = []
        
        # Classical Chart Pattern Highlight (Fidelity / CFI / SRCC)
        if pattern_st:
            thoughts.append(f"Chart Pattern: [{pattern_st.description}]. Target: {pattern_st.target_price:.1f}, SL: {pattern_st.stop_loss_level:.1f}.")

        # Pillar 3: Heavyweight Divergence Trap Check
        if heavyweight_st and getattr(heavyweight_st, 'divergence_signal', None):
            thoughts.append(f"Heavyweight Warning: {heavyweight_st.divergence_signal}")
            
        # Pillar 2: News Risk Check
        if news_st and getattr(news_st, 'risk_level', None) and news_st.risk_level.value == "CAUTION":
            thoughts.append(f"Event Alert: {news_st.warning_message}")

        # Committee Decision Reasoning
        if ensemble_dec.final_action.value != "HOLD" and not ensemble_dec.is_vetoed:
            action_word = "BUYING CALL" if ensemble_dec.final_action.value == "BUY" else "BUYING PUT"
            thoughts.append(f"High-conviction alignment observed. Committee consensus triggers {action_word}.")
            if execution_mode_desc:
                thoughts.append(f"[{execution_mode_desc}]")
            if smc_st and getattr(smc_st, 'liquidity_sweep', None):
                thoughts.append(f"SMC Liquidity Trap Confirmed: [{smc_st.liquidity_sweep}].")
            thoughts.append(f"Eyes: {vision_desc} | GARCH: {garch_res.signal.value} ({garch_res.annualized_vol*100:.1f}%) | Toxicity: {vpin_res.regime.value}.")
            thoughts.append(f"Playbook: [{playbook.name}] -> {playbook.focus_mode}.")
        elif ensemble_dec.is_vetoed:
            thoughts.append(f"Committee spotted divergence or microstructure resistance. Vetoing entry to protect capital.")
            thoughts.append(f"Context: SVM={svm_str}, XGB={xgb_str}, Toxicity={vpin_res.vpin:.3f}.")
        elif position_side_str != "FLAT":
            thoughts.append(f"Active in trade ({position_side_str}). Step Trailing Guardian stops active under {playbook.name} rules.")
        else:
            thoughts.append(f"Scanning market flow under [{playbook.name}]. Spot: {curr_price:,.2f} | Awaiting institutional setup.")

        full_thought = " ".join(thoughts)
        self.last_thought = full_thought
        return full_thought

    def render_humanoid_hud(
        self,
        timestamp: datetime,
        curr_price: float,
        vpin_res: Any,
        garch_res: Any,
        ensemble_dec: Any,
        pos_str: str,
        day_pnl: float,
        vision_str: str = "Neutral",
        global_st: Any = None,
        news_st: Any = None,
        heavyweight_st: Any = None,
        smc_st: Any = None,
        journal: Any = None,
        execution_mode_tag: str = "AUTO (Scalp/Swing)"
    ):
        """
        Renders a comprehensive Humanoid Cognitive HUD displaying all 4 Pre-Market Pillars and Multi-Brain State.
        """
        playbook = self.get_tactical_playbook(timestamp)
        thought = self.synthesize_thought(
            curr_price=curr_price,
            vpin_res=vpin_res,
            garch_res=garch_res,
            ensemble_dec=ensemble_dec,
            macro_st=None,
            smc_st=smc_st,
            global_st=global_st,
            news_st=news_st,
            heavyweight_st=heavyweight_st,
            position_side_str=pos_str
        )

        pnl_color = "+" if day_pnl >= 0 else ""
        global_desc = global_st.description if global_st else "Global Cues: Initializing"
        news_desc = news_st.warning_message if news_st else "News: Clear"
        hw_desc = heavyweight_st.summary if heavyweight_st else "Heavyweights: Monitoring"
        
        levels_str = ""
        if journal:
            levels_str = f"PDH: {journal.pdh:.1f} | PDL: {journal.pdl:.1f} | 50% Eq: {journal.equilibrium_50:.1f} | Max Trades: {journal.max_trades_limit}"
        elif smc_st:
            levels_str = f"PDH: {smc_st.pdh:.1f} | PDL: {smc_st.pdl:.1f} | Range Pos: {smc_st.dealing_range_pct*100:.0f}% ({smc_st.zone})"

        hud = (
            f"\n+================================================================================+\n"
            f"| [HUMANOID AI TRADER: {self.agent_name}] -> {timestamp.strftime('%H:%M:%S IST')} | Phase: {playbook.name}\n"
            f"+--------------------------------------------------------------------------------+\n"
            f"| [1. Global Cues  ] : {global_desc}\n"
            f"| [2. News/Calendar] : {news_desc}\n"
            f"| [3. Heavyweights ] : {hw_desc}\n"
            f"| [4. Pre-Market   ] : {levels_str}\n"
            f"+--------------------------------------------------------------------------------+\n"
            f"| * Spot Index     : {curr_price:,.2f} (Vol: {garch_res.annualized_vol*100:.1f}%, Toxicity: {vpin_res.vpin:.3f})\n"
            f"| * 4-Brain Models : SVM={ensemble_dec.svm_prediction:+d} | XGB={ensemble_dec.xgb_prediction:+d} ({ensemble_dec.xgb_confidence*100:.0f}%) | ANN={getattr(ensemble_dec, 'ann_prediction', 0):+d} | Eyes={vision_str}\n"
            f"| * Execution Mode : {execution_mode_tag}\n"
            f"| * Active Strategy: {playbook.focus_mode} (Target: +{playbook.target_profit_pct*100:.0f}%, Trail: +{playbook.trailing_step_pct*100:.0f}%)\n"
            f"| * Position State : {pos_str} | Cumulative Day P&L: {pnl_color}{day_pnl:,.2f} INR\n"
            f"+--------------------------------------------------------------------------------+\n"
            f"| >>> COGNITIVE THOUGHT STREAM:\n"
            f"| \"{thought}\"\n"
            f"+================================================================================+"
        )
        logger.info(hud)
        import sys
        sys.stdout.flush()
