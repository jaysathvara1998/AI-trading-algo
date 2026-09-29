"""
Engine configuration. Every threshold in the specification is a hypothesis; all of them live here,
are serialisable to JSON (versioned with each backtest), and can be overridden per experiment.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class Timeframes:
    context: int = 5       # minutes: regime, major structure
    setup: int = 3         # minutes: liquidity interaction, BOS/CHOCH, retest
    execution: int = 1     # minutes: entry trigger


@dataclass
class SwingConfig:
    left: int = 2                   # bars on each side for a local extremum
    right: int = 2                  # confirmation latency (bars); recorded on every swing
    min_dist_atr: float = 0.5       # consecutive swings must be at least this far apart (setup-tf ATR)
    atr_period: int = 14


@dataclass
class RegimeConfig:
    highvol_atr_ratio: float = 1.8      # context ATR(fast) / ATR(slow) above this = HIGH_VOLATILITY
    fast_atr: int = 5
    slow_atr: int = 30
    range_overlap_atr: float = 0.5      # last two swing highs and lows within this = RANGE


@dataclass
class LevelsConfig:
    opening_range_min: int = 15
    equal_tol_atr: float = 0.15         # two swings within this distance = equal highs/lows
    recent_swings: int = 3              # how many recent setup-tf swing highs/lows count as levels
    max_touches: int = 2                # level no longer fresh after this many prior touches today


@dataclass
class SweepConfig:
    min_penetration_atr: float = 0.10   # must trade beyond the level by at least this (setup-tf ATR)
    max_penetration_atr: float = 1.50   # beyond this it is a breakout, not a sweep
    reclaim_bars: int = 2               # must close back inside within this many setup-tf candles (1 = same candle)
    require_wick: bool = False          # require rejection wick >= 40% of candle range on the sweep candle


@dataclass
class StructureConfig:
    bos_min_atr: float = 0.15           # close must exceed the swing by this (noise band)
    max_bars_after_sweep: int = 6       # setup-tf candles allowed between the sweep and the structure shift
    require_shift: bool = True          # BOS/CHOCH required (research matrix: sweep-only entries when False)


@dataclass
class RetestConfig:
    enabled: bool = True
    zone_atr: float = 0.25              # retest zone = broken level +/- this
    expire_bars: int = 8                # setup-tf candles to wait for the retest
    invalidate_atr: float = 0.35        # a close this far back through the broken level kills the setup


@dataclass
class ConfirmationConfig:
    body_frac: float = 0.5              # execution candle body >= this fraction of its range
    micro_bos_lookback: int = 5         # execution-tf swing lookback for a micro-BOS
    max_bars: int = 10                  # execution candles to wait for confirmation after retest / shift
    require_momentum: bool = True


@dataclass
class ScoringConfig:
    context_structure: int = 2
    setup_structure: int = 2
    sweep: int = 2
    shift: int = 2
    retest: int = 2
    micro_bos: int = 1
    momentum: int = 1
    volume: int = 1
    location: int = 1
    entry_threshold: int = 8


@dataclass
class RiskConfig:
    buffer_atr: float = 0.25            # stop buffer beyond the invalidation level (setup-tf ATR)
    min_stop_atr: float = 0.5
    max_stop_atr: float = 3.0
    min_rr: float = 2.0                 # realistic structural target must be at least this many R away
    min_target_atr: float = 1.0         # ignore opposing levels closer than this when choosing the target
    max_risk_inr: float = 1500.0        # per-trade monetary risk cap (used for sizing and for skipping)
    sizing: str = "fixed"               # "fixed" (lots = base_lots, skip if risk > cap) or "risk" (lots from cap)
    base_lots: int = 1
    max_lots: int = 3
    lot_size: int = 65
    daily_loss_limit_inr: float = 3000.0
    max_consecutive_losses: int = 2
    max_trades_per_day: int = 3
    cooldown_bars: int = 5              # setup-tf candles after an exit before a new setup can enter


@dataclass
class ExitConfig:
    tp1_r: float = 1.0
    tp1_fraction: float = 0.5           # fraction of the position closed at TP1 (0 = none)
    tp2_r: float = 2.0
    tp2_fraction: float = 0.0           # additional fraction closed at TP2 (remainder runs)
    breakeven_after_r: float = 1.0      # move stop to entry (+buffer) once this R is reached (0 = never)
    breakeven_buffer_atr: float = 0.05
    trail: str = "hybrid"               # "structure" | "atr" | "hybrid" | "none"
    trail_atr_factor: float = 1.5
    trail_after_r: float = 1.0          # trailing starts once this R is reached
    time_exit_bars: int = 20            # setup-tf candles without reaching progress_r -> exit
    time_exit_progress_r: float = 0.5
    invalidation_exit: bool = True      # exit on an opposing CHOCH on the setup timeframe


@dataclass
class SessionConfig:
    entry_start: str = "09:30"
    entry_end: str = "14:30"
    flatten: str = "15:15"
    skip_expiry_day: bool = True        # NIFTY Tue / SENSEX Thu (post Sep 2025 calendar)


@dataclass
class CostConfig:
    instrument: str = "option"          # "option" | "futures" | "points" (raw underlying, no costs)
    premium_pct: float = 0.007          # option premium as fraction of spot (ITM weekly, ~0.72 delta)
    delta: float = 0.72
    theta_per_min: float = 0.00025      # fraction of premium lost per minute held
    spread_pct: float = 0.002           # per side
    brokerage_per_order: float = 20.0
    stt_sell_pct: float = 0.0015        # options; futures use 0.0002 on notional
    slippage_points: float = 0.5        # underlying slippage per side (futures / points modes)


@dataclass
class EngineConfig:
    symbol: str = "NIFTY"
    tf: Timeframes = field(default_factory=Timeframes)
    swing: SwingConfig = field(default_factory=SwingConfig)
    regime: RegimeConfig = field(default_factory=RegimeConfig)
    levels: LevelsConfig = field(default_factory=LevelsConfig)
    sweep: SweepConfig = field(default_factory=SweepConfig)
    structure: StructureConfig = field(default_factory=StructureConfig)
    retest: RetestConfig = field(default_factory=RetestConfig)
    confirm: ConfirmationConfig = field(default_factory=ConfirmationConfig)
    score: ScoringConfig = field(default_factory=ScoringConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    exit: ExitConfig = field(default_factory=ExitConfig)
    session: SessionConfig = field(default_factory=SessionConfig)
    cost: CostConfig = field(default_factory=CostConfig)
    version: str = "v3.0"

    # ---- (de)serialisation and dotted overrides ----
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EngineConfig":
        cfg = cls()
        for f in fields(cls):
            if f.name in d:
                cur = getattr(cfg, f.name)
                if is_dataclass(cur) and isinstance(d[f.name], dict):
                    for k, v in d[f.name].items():
                        if hasattr(cur, k):
                            setattr(cur, k, v)
                else:
                    setattr(cfg, f.name, d[f.name])
        return cfg

    @classmethod
    def load(cls, path: str | Path) -> "EngineConfig":
        return cls.from_dict(json.loads(Path(path).read_text()))

    def override(self, **kv: Any) -> "EngineConfig":
        """cfg.override(**{"retest.enabled": False, "risk.min_rr": 1.5}) -> new config."""
        d = self.to_dict()
        for key, val in kv.items():
            parts = key.split(".")
            node = d
            for p in parts[:-1]:
                node = node[p]
            node[parts[-1]] = _coerce(val, node.get(parts[-1]))
        return EngineConfig.from_dict(d)

    def for_symbol(self, symbol: str) -> "EngineConfig":
        s = symbol.upper()
        lot = {"NIFTY": 65, "SENSEX": 20, "BANKNIFTY": 30}.get(s, 65)
        return self.override(**{"symbol": s, "risk.lot_size": lot})


def _coerce(val: Any, current: Any) -> Any:
    if isinstance(current, bool):
        return val if isinstance(val, bool) else str(val).lower() in ("1", "true", "yes", "on")
    if isinstance(current, int) and not isinstance(current, bool):
        return int(val)
    if isinstance(current, float):
        return float(val)
    return val
