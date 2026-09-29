"""Unit tests for the deterministic building blocks of pa_engine (run: .venv/bin/python -m pytest -q)."""
import pandas as pd
import pytest

from pa_engine.candles import Candle, aggregate, atr, parse_hhmm, minute_index
from pa_engine.config import EngineConfig
from pa_engine.costs import evaluate, risk_inr
from pa_engine.levels import Level
from pa_engine.liquidity import detect_sweeps
from pa_engine.position import Position
from pa_engine.risk import RiskPlan, plan, structural_stop
from pa_engine.structure import compute_structure, label_swings, trend_from_labels
from pa_engine.swings import detect_swings


def mk(vals, start="2026-01-05 09:15"):
    t0 = pd.Timestamp(start, tz="Asia/Kolkata")
    out = []
    for i, v in enumerate(vals):
        if isinstance(v, tuple):
            o, h, l, c = v
        else:
            o = c = v; h = v + 1; l = v - 1
        out.append(Candle(t0 + pd.Timedelta(minutes=i), o, h, l, c))
    return out


def test_minute_index_and_parse():
    assert minute_index(pd.Timestamp("2026-01-05 09:15", tz="Asia/Kolkata")) == 0
    assert parse_hhmm("15:15") == 360 and parse_hhmm("09:30") == 15


def test_aggregate_uses_only_completed_candles():
    c = mk(range(10))
    assert len(aggregate(c, 3)) == 3
    assert len(aggregate(c, 3, upto=7)) == 2          # minutes 0..7 -> candles [0,1,2],[3,4,5] complete; [6,7] not
    g = aggregate(c, 3)[1]
    assert g.open == 3 and g.close == 5 and g.high == 6 and g.low == 2


def test_swings_confirmed_with_latency_and_alternate():
    c = mk([10, 11, 15, 11, 10, 9, 5, 9, 10, 14, 10, 9])
    sw = detect_swings(c, atr_value=1.0, left=2, right=2, min_dist_atr=0.0)
    kinds = [s.kind for s in sw]
    assert kinds == ["H", "L", "H"]
    assert sw[0].idx == 2 and sw[0].confirmed_idx == 4
    assert sw[1].price == 4 and sw[1].idx == 6
    # nothing is known before confirmation
    assert detect_swings(c, 1.0, 2, 2, 0.0, upto=3) == []


def test_structure_bos_choch_needs_close_beyond_band():
    labels = ["HH", "HL", "HH", "HL"]
    assert trend_from_labels(labels) == "BULL"
    assert trend_from_labels(["LL", "LH", "LL", "LH"]) == "BEAR"
    c = mk([10, 11, 15, 11, 10, 9, 5, 9, 10, 14, 10, 9, 8, 15.05, 15.5])
    sw = detect_swings(c, 1.0, 2, 2, 0.0)
    st = compute_structure(c, sw, atr_value=1.0, bos_min_atr=0.15)
    ev = [e for e in st.events if e.direction > 0]
    # latest known swing high is candle 9 (high 15). Close 15.05 at idx 13 is inside the 0.15 band (no break);
    # close 15.5 at idx 14 breaks it. Prior structure is not BULL, so the break is a CHOCH.
    assert len(ev) == 1 and ev[0].idx == 14 and ev[0].level == 15.0 and ev[0].kind == "CHOCH"


def test_sweep_detection_penetration_and_reclaim():
    lv = [Level("PDH", 100.0, "H", "PDH")]
    # candle 1 pokes above 100 by 0.5 (0.5 ATR) and closes back below -> sweep
    c = mk([(99, 99.5, 98.5, 99), (99, 100.5, 98.8, 99.2), (99.2, 99.6, 98.9, 99.0)])
    cfg = EngineConfig().sweep
    seen = set()
    ev = detect_sweeps(c, lv, 1.0, 1, cfg, seen)
    assert len(ev) == 1 and ev[0].direction == -1 and abs(ev[0].penetration_atr - 0.5) < 1e-9
    assert detect_sweeps(c, lv, 1.0, 1, cfg, seen) == []           # de-duplicated
    # acceptance above the level (close beyond) is not a sweep
    c2 = mk([(99, 99.5, 98.5, 99), (99, 100.5, 98.8, 100.3)])
    assert detect_sweeps(c2, lv, 1.0, 1, cfg, set()) == []
    # too-large penetration is a breakout, not a sweep
    c3 = mk([(99, 99.5, 98.5, 99), (99, 102.0, 98.8, 99.2)])
    assert detect_sweeps(c3, lv, 1.0, 1, cfg, set()) == []


def test_risk_plan_rules():
    cfg = EngineConfig()
    assert structural_stop(+1, 100.0, 2.0, cfg.risk) == 99.5
    levels = [Level("PDH", 110.0, "H", "PDH"), Level("X", 101.0, "H", "SWING_H")]
    p, why = plan(+1, 102.0, 100.0, levels, 2.0, cfg.risk, cfg.cost, cfg.exit, "test")
    assert p is not None and p.stop == 99.5 and p.target_name == "PDH" and p.rr == pytest.approx(8 / 2.5)
    assert p.quantity == cfg.risk.lot_size and p.risk_inr == pytest.approx(2.5 * 0.72 * 65)
    # too-wide stop rejected
    p, why = plan(+1, 102.0, 90.0, levels, 2.0, cfg.risk, cfg.cost, cfg.exit, "test")
    assert p is None and "max" in why
    # insufficient R:R rejected
    p, why = plan(+1, 102.0, 100.0, [Level("near", 104.0, "H", "SWING_H")], 1.0, cfg.risk, cfg.cost, cfg.exit, "t")
    assert p is None and "R:R" in why
    # monetary cap rejected
    r2 = cfg.override(**{"risk.max_risk_inr": 50})
    p, why = plan(+1, 102.0, 100.0, levels, 2.0, r2.risk, r2.cost, r2.exit, "t")
    assert p is None and "cap" in why


def _pos(direction=+1):
    cfg = EngineConfig()
    rp = RiskPlan(direction, 100.0, 98.0 if direction > 0 else 102.0, 2.0, 100.0 + 2 * direction, 100.0 + 4 * direction, "T", 2.0, 65, 1, 93.6, "t")
    p = Position(rp, 0, 100.0, 65, "s"); p._exit_cfg = cfg.exit; p._atr = 1.0
    return p


def test_position_adverse_first_then_partials_and_never_loosens():
    p = _pos()
    # candle touches both the stop and TP1: stop wins
    f = p.on_candle(Candle(pd.Timestamp.now(), 100, 103, 97, 101), 1, 999)
    assert len(f) == 1 and f[0].reason == "STOP" and f[0].quantity == 65 and not p.open
    p = _pos()
    f = p.on_candle(Candle(pd.Timestamp.now(), 100, 102.5, 99.5, 102), 1, 999)
    assert f and f[0].reason == "TP1" and f[0].quantity == 32 and p.remaining == 33 and p.state == "PROFIT_1R"
    assert p.stop >= 100.0                         # breakeven after 1R
    old = p.stop
    p._tighten(99.0, 2, "x")
    assert p.stop == old                            # never loosened
    f = p.on_candle(Candle(pd.Timestamp.now(), 102, 104.5, 101.5, 104), 3, 999)
    assert p.state == "RUNNER" and p.remaining == 33 and f == []
    f = p.on_candle(Candle(pd.Timestamp.now(), 104, 104.2, 103.5, 104), 4, 4)
    assert f and f[0].reason == "SESSION_FLATTEN" and not p.open


def test_position_time_exit_and_invalidation():
    p = _pos()
    for i in range(19):
        assert p.on_setup_close(i, 100.2, 1.0, [], choch_against=False) == []
    f = p.on_setup_close(19, 100.2, 1.0, [], choch_against=False)      # 20th setup candle without 0.5R progress
    assert f and f[0].reason == "TIME_EXIT"
    p = _pos()
    f = p.on_setup_close(1, 99.5, 1.0, [], choch_against=True)
    assert f and f[0].reason == "STRUCTURE_INVALIDATED"


def test_costs_option_round_trip_is_negative_on_flat_exit():
    cfg = EngineConfig().cost
    e = evaluate(cfg, +1, 25000.0, 25000.0, 10, 65)
    assert e.net < 0 and e.costs > 0 and abs(e.gross) < 200
    assert risk_inr(cfg, 20.0, 65) == pytest.approx(20 * 0.72 * 65)


def test_config_override_and_roundtrip(tmp_path):
    cfg = EngineConfig().override(**{"risk.min_rr": "1.5", "retest.enabled": "false", "score.entry_threshold": 6})
    assert cfg.risk.min_rr == 1.5 and cfg.retest.enabled is False and cfg.score.entry_threshold == 6
    cfg.save(tmp_path / "c.json"); back = EngineConfig.load(tmp_path / "c.json")
    assert back.to_dict() == cfg.to_dict()
    assert EngineConfig().for_symbol("sensex").risk.lot_size == 20


def test_backtest_is_deterministic():
    from pa_engine.backtest import run
    from pa_engine.candles import load_sessions
    ss = load_sessions("data/nifty_12m_1min.csv.gz")[:12]
    cfg = EngineConfig()
    a = run(cfg, ss); b = run(cfg, ss)
    assert a.equals(b)
    if len(a):
        assert (a.entry_minute > a.index.map(lambda i: -1)).all()
        assert (a.exit_minute >= a.entry_minute).all()
        assert (a.exit_minute <= parse_hhmm("15:15")).all()
