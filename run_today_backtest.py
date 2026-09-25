"""
High-Precision Historical Replay & Backtest Engine for Today's Session (2026-09-21)
Evaluates Track 1 (Dual-Brain SVM + XGB) vs Track 2 (Tri-Brain ANN + XGB + SVM)
using real Dhan 1-minute exchange bars and live option contract tick data.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import pytz
import pandas as pd
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from algo_vpin_v2.config import CONFIG, InstrumentMode, TradeStrategyMode
from algo_vpin_v2.data_feed import DhanDataFeed, BarOHLCV
from algo_vpin_v2.vpin import VPINCalculator, ToxicityRegime
from algo_vpin_v2.garch_engine import GARCHEngine, GARCHForecastResult, DirectionalSignal
from algo_vpin_v2.macro_features import MacroFeatureEngine
from algo_vpin_v2.smc_engine import SMCEngine
from algo_vpin_v2.ensemble_brain import EnsembleBrain
from algo_vpin_v2.risk_manager import RiskManager, PositionSide, ActivePosition


def run_today_backtest():
    tz = pytz.timezone("Asia/Kolkata")
    feed = DhanDataFeed(CONFIG)
    dhan = feed.tradehull_client.Dhan if feed.tradehull_client else None
    if dhan is None:
        print("[ERROR] Dhan client not authenticated. Please check credentials.")
        return

    today_str = "2026-09-21"
    print("=" * 80)
    print(f"      RUNNING BACKTEST REPLAY FOR TODAY'S SESSION ({today_str})")
    print("=" * 80)

    # 1. Fetch Today's 1-Minute Exchange Bars for NIFTY Spot
    print(f"[1/4] Fetching today's 1-minute exchange bars for NIFTY from Dhan API...")
    r_today = dhan.intraday_minute_data(
        security_id="13",
        exchange_segment="IDX_I",
        instrument_type="INDEX",
        from_date=today_str,
        to_date=today_str
    )
    if not (isinstance(r_today, dict) and r_today.get("status") == "success" and "data" in r_today):
        print(f"[ERROR] Failed to fetch today's bars: {r_today}")
        return

    today_df = pd.DataFrame(r_today["data"])
    today_df["timestamp"] = pd.to_datetime(today_df["timestamp"], unit="s", utc=True).dt.tz_convert(tz)
    today_df = today_df.sort_values("timestamp").reset_index(drop=True)
    print(f"Loaded {len(today_df)} exchange bars for today (from {today_df.iloc[0]['timestamp'].strftime('%H:%M')} to {today_df.iloc[-1]['timestamp'].strftime('%H:%M')} IST).")

    # 2. Fetch Warm-Up Bars from previous sessions
    print("[2/4] Ingesting warm-up history for VPIN, GARCH, and SMC Macro state...")
    hist_prev = feed.fetch_historical_bars(days=5)
    hist_prev = hist_prev[hist_prev["timestamp"].dt.strftime("%Y-%m-%d") < today_str]
    warmup_df = hist_prev.tail(150).reset_index(drop=True)

    # 3. Cache Option Data for today's active strikes
    print("[3/4] Caching real option strike candle data from Dhan...")
    option_cache = {}
    inst_df = feed.tradehull_client.instrument_df

    def get_option_bars(sec_id_str: str):
        if sec_id_str not in option_cache:
            r = dhan.intraday_minute_data(
                security_id=sec_id_str,
                exchange_segment="NSE_FNO",
                instrument_type="OPTIDX",
                from_date=today_str,
                to_date=today_str
            )
            if isinstance(r, dict) and r.get("status") == "success" and "data" in r:
                odf = pd.DataFrame(r["data"])
                odf["timestamp"] = pd.to_datetime(odf["timestamp"], unit="s", utc=True).dt.tz_convert(tz)
                odf = odf.set_index(odf["timestamp"].dt.strftime("%H:%M"))
                option_cache[sec_id_str] = odf
            else:
                option_cache[sec_id_str] = None
        return option_cache[sec_id_str]

    # Pre-fetch key strikes traded today: 23350 CE, 23400 CE, 23450 CE, 23500 PE, 23450 PE
    key_sec_ids = {
        "23500 PE": "57022",
        "23350 CE": "56995",
        "23400 CE": "57002",
        "23450 PE": "57011",
    }
    for k, sid in key_sec_ids.items():
        get_option_bars(sid)

    # 4. Initialize Engines
    macro_eng = MacroFeatureEngine()
    vpin_calc = VPINCalculator(CONFIG.vpin)
    garch_eng = GARCHEngine(CONFIG.garch)
    smc_eng = SMCEngine()
    ensemble = EnsembleBrain(CONFIG.ensemble)
    s_ok, x_ok, a_ok = ensemble.load_all_models("NIFTY")
    print(f"Loaded Models -> SVM={s_ok}, XGBoost={x_ok}, ANN={a_ok}")

    # Seed warm-up bars
    for _, row in warmup_df.iterrows():
        p = float(row["close"])
        h = float(row.get("high", p))
        l = float(row.get("low", p))
        v = float(row["volume"])
        ts = row["timestamp"]
        macro_eng.update_1min_bar(p, h, l)
        vpin_calc.process_bar(p, v, ts)
        garch_eng.add_bar(p, ts)
        smc_eng.update_bar(p, h, l, p, v)

    # Setup Simulation Tracks
    track1_pos = None
    track2_pos = None
    track1_trades = []
    track2_trades = []

    recent_highs = []
    recent_lows = []

    print("[4/4] Executing Replay Simulation across today's session with Chinmay Trading Skills...")

    for i, row in today_df.iterrows():
        p = float(row["close"])
        o = float(row.get("open", p))
        h = float(row.get("high", p))
        l = float(row.get("low", p))
        v = float(row["volume"])
        ts = row["timestamp"]
        time_str = ts.strftime("%H:%M")

        recent_highs.append(h)
        recent_lows.append(l)

        # Engine updates
        mst = macro_eng.update_1min_bar(p, h, l)
        vres = vpin_calc.process_bar(p, v, ts)
        gres = garch_eng.add_bar(p, ts)
        smc_st = smc_eng.update_bar(o, h, l, p, v)

        if gres is None:
            continue

        prev_o = float(today_df.iloc[i - 1].get("open", p)) if i > 0 else o
        prev_c = float(today_df.iloc[i - 1]["close"]) if i > 0 else p
        p_delta = p - prev_c
        sigma = gres.sigma_next
        mu = gres.mu_next

        key_levels = [mst.pdl, mst.pdh, round((mst.pdh + mst.pdl) / 2.0, 2)]

        # Evaluate 4-Brain Consensus empowered by Chinmay Trading Skills
        dec = ensemble.evaluate(
            garch_signal=gres.signal,
            price_delta=p_delta,
            rolling_vol=sigma,
            garch_forecast=mu,
            vpin=vres.vpin,
            macro_state=mst,
            smc_state=smc_st,
            open_p=o,
            high_p=h,
            low_p=l,
            close_p=p,
            prev_open=prev_o,
            prev_close=prev_c,
            recent_highs=recent_highs[-10:],
            recent_lows=recent_lows[-10:],
            key_levels=key_levels,
            underlying="NIFTY"
        )

        def get_live_opt_price(opt_sym: str, spot_p: float):
            sec_row = inst_df[inst_df["SEM_CUSTOM_SYMBOL"] == opt_sym]
            if not sec_row.empty:
                sid = str(sec_row.iloc[-1]["SEM_SMST_SECURITY_ID"])
                odf = get_option_bars(sid)
                if odf is not None and time_str in odf.index:
                    return float(odf.loc[time_str]["close"])
            strike = 23400.0
            import re
            m = re.search(r"(\d{5})", opt_sym)
            if m:
                strike = float(m.group(1))
            is_c = "CALL" in opt_sym or "CE" in opt_sym
            intrinsic = max(0.0, spot_p - strike) if is_c else max(0.0, strike - spot_p)
            return round(intrinsic + 26.0, 2)

        # -------------------------------------------------------------
        # TRACK 1 SIMULATION (Dual-Brain + Position Guardian Skill)
        # -------------------------------------------------------------
        if track1_pos is not None:
            opt_p = get_live_opt_price(track1_pos["symbol"], p)
            guardian_res = ensemble.position_guardian_skill.evaluate(
                symbol=track1_pos["symbol"],
                is_call=("CALL" in track1_pos["symbol"]),
                entry_price=track1_pos["entry_p"],
                current_option_price=opt_p,
                current_spot_price=p,
                spot_sl=track1_pos["spot_sl"],
                current_sl=track1_pos["sl"],
                target_tp=track1_pos["tp"],
                initial_risk_pts=track1_pos["initial_risk_pts"],
                highest_price_seen=track1_pos["highest_p"],
                current_time_hour=ts.hour,
                current_time_minute=ts.minute
            )
            g_act = guardian_res.metadata["action"]
            track1_pos["sl"] = g_act.updated_sl
            track1_pos["highest_p"] = guardian_res.metadata["peak_price"]

            if g_act.should_exit:
                pnl = (opt_p - track1_pos["entry_p"]) * 65
                track1_trades.append({
                    "entry_time": track1_pos["entry_time"],
                    "exit_time": time_str,
                    "symbol": track1_pos["symbol"],
                    "action": track1_pos["action"],
                    "entry_p": track1_pos["entry_p"],
                    "exit_p": opt_p,
                    "pnl": pnl,
                    "pnl_pct": ((opt_p - track1_pos["entry_p"]) / track1_pos["entry_p"]) * 100,
                    "reason": g_act.exit_reason,
                    "pattern": track1_pos.get("pattern", "Micro-Momentum"),
                    "rr_ratio": track1_pos.get("rr_ratio", 1.5)
                })
                track1_pos = None

        if track1_pos is None and (ts.hour < 15 or (ts.hour == 15 and ts.minute <= 15)):
            if not dec.is_vetoed and dec.final_action in (DirectionalSignal.BUY, DirectionalSignal.SELL):
                act = "BUY" if dec.final_action == DirectionalSignal.BUY else "SELL"
                sym = feed.resolve_option_strike(act, p)
                opt_p = get_live_opt_price(sym, p)

                rr_plan = dec.risk_reward_result.metadata.get("plan") if dec.risk_reward_result else None
                if rr_plan is not None:
                    sl_pts = rr_plan.option_risk_pts
                    tp_pts = rr_plan.option_reward_pts
                    spot_sl = rr_plan.spot_sl
                    rr_ratio = rr_plan.risk_reward_ratio
                else:
                    sl_pts = min(10.0, max(6.0, opt_p * 0.12))
                    tp_pts = sl_pts * 1.5
                    spot_sl = (p - 18.0) if act == "BUY" else (p + 18.0)
                    rr_ratio = 1.5

                pa_meta = dec.price_action_result.metadata if dec.price_action_result else {}
                pats = pa_meta.get("patterns", [])
                pat_str = ", ".join(pats) if pats else "Micro-Momentum"

                track1_pos = {
                    "entry_time": time_str,
                    "symbol": sym,
                    "action": act,
                    "entry_p": opt_p,
                    "sl": round(opt_p - sl_pts, 2),
                    "tp": round(opt_p + tp_pts, 2),
                    "initial_risk_pts": sl_pts,
                    "highest_p": opt_p,
                    "entry_spot": p,
                    "spot_sl": spot_sl,
                    "pattern": pat_str,
                    "rr_ratio": rr_ratio
                }

        # -------------------------------------------------------------
        # TRACK 2 SIMULATION (Tri-Brain + Position Guardian Skill)
        # -------------------------------------------------------------
        if track2_pos is not None:
            opt_p = get_live_opt_price(track2_pos["symbol"], p)
            guardian_res = ensemble.position_guardian_skill.evaluate(
                symbol=track2_pos["symbol"],
                is_call=("CALL" in track2_pos["symbol"]),
                entry_price=track2_pos["entry_p"],
                current_option_price=opt_p,
                current_spot_price=p,
                spot_sl=track2_pos["spot_sl"],
                current_sl=track2_pos["sl"],
                target_tp=track2_pos["tp"],
                initial_risk_pts=track2_pos["initial_risk_pts"],
                highest_price_seen=track2_pos["highest_p"],
                current_time_hour=ts.hour,
                current_time_minute=ts.minute
            )
            g_act = guardian_res.metadata["action"]
            track2_pos["sl"] = g_act.updated_sl
            track2_pos["highest_p"] = guardian_res.metadata["peak_price"]

            if g_act.should_exit:
                pnl = (opt_p - track2_pos["entry_p"]) * 65
                track2_trades.append({
                    "entry_time": track2_pos["entry_time"],
                    "exit_time": time_str,
                    "symbol": track2_pos["symbol"],
                    "action": track2_pos["action"],
                    "entry_p": track2_pos["entry_p"],
                    "exit_p": opt_p,
                    "pnl": pnl,
                    "pnl_pct": ((opt_p - track2_pos["entry_p"]) / track2_pos["entry_p"]) * 100,
                    "reason": g_act.exit_reason,
                    "pattern": track2_pos.get("pattern", "Micro-Momentum"),
                    "rr_ratio": track2_pos.get("rr_ratio", 1.5)
                })
                track2_pos = None

        if track2_pos is None and (ts.hour < 15 or (ts.hour == 15 and ts.minute <= 15)):
            if not dec.tri_is_vetoed and dec.tri_final_action in (DirectionalSignal.BUY, DirectionalSignal.SELL):
                act = "BUY" if dec.tri_final_action == DirectionalSignal.BUY else "SELL"
                sym = feed.resolve_option_strike(act, p)
                opt_p = get_live_opt_price(sym, p)

                rr_plan = dec.risk_reward_result.metadata.get("plan") if dec.risk_reward_result else None
                if rr_plan is not None:
                    sl_pts = rr_plan.option_risk_pts
                    tp_pts = rr_plan.option_reward_pts
                    spot_sl = rr_plan.spot_sl
                    rr_ratio = rr_plan.risk_reward_ratio
                else:
                    sl_pts = min(10.0, max(6.0, opt_p * 0.12))
                    tp_pts = sl_pts * 1.5
                    spot_sl = (p - 18.0) if act == "BUY" else (p + 18.0)
                    rr_ratio = 1.5

                pa_meta = dec.price_action_result.metadata if dec.price_action_result else {}
                pats = pa_meta.get("patterns", [])
                pat_str = ", ".join(pats) if pats else "Micro-Momentum"

                track2_pos = {
                    "entry_time": time_str,
                    "symbol": sym,
                    "action": act,
                    "entry_p": opt_p,
                    "sl": round(opt_p - sl_pts, 2),
                    "tp": round(opt_p + tp_pts, 2),
                    "initial_risk_pts": sl_pts,
                    "highest_p": opt_p,
                    "entry_spot": p,
                    "spot_sl": spot_sl,
                    "pattern": pat_str,
                    "rr_ratio": rr_ratio
                }

    # -------------------------------------------------------------
    # PRINT PERFORMANCE SCOREBOARD
    # -------------------------------------------------------------
    df1 = pd.DataFrame(track1_trades)
    df2 = pd.DataFrame(track2_trades)

    print("\n" + "=" * 90)
    print("      TRACK 1 (DUAL-BRAIN + CHINMAY SKILLS) TODAY'S TRADE JOURNAL")
    print("=" * 90)
    if df1.empty:
        print("  No trades triggered for Track 1.")
    else:
        for idx, t in df1.iterrows():
            pnl_str = f"+INR {t['pnl']:,.2f}" if t['pnl'] >= 0 else f"-INR {abs(t['pnl']):,.2f}"
            print(f"  Trade #{idx+1} | {t['entry_time']} -> {t['exit_time']} IST | {t['symbol']} ({t['action']})")
            print(f"     Setup: {t['pattern']} | Target R:R: 1:{t['rr_ratio']:.2f}")
            print(f"     Entry: INR {t['entry_p']:.2f} | Exit: INR {t['exit_p']:.2f} | PnL: {pnl_str} ({t['pnl_pct']:+.1f}%)")
            print(f"     Reason: {t['reason']}\n")

    print("=" * 90)
    print("      TRACK 2 (TRI-BRAIN ANN + CHINMAY SKILLS) TODAY'S TRADE JOURNAL")
    print("=" * 90)
    if df2.empty:
        print("  No trades triggered for Track 2.")
    else:
        for idx, t in df2.iterrows():
            pnl_str = f"+INR {t['pnl']:,.2f}" if t['pnl'] >= 0 else f"-INR {abs(t['pnl']):,.2f}"
            print(f"  Trade #{idx+1} | {t['entry_time']} -> {t['exit_time']} IST | {t['symbol']} ({t['action']})")
            print(f"     Setup: {t['pattern']} | Target R:R: 1:{t['rr_ratio']:.2f}")
            print(f"     Entry: INR {t['entry_p']:.2f} | Exit: INR {t['exit_p']:.2f} | PnL: {pnl_str} ({t['pnl_pct']:+.1f}%)")
            print(f"     Reason: {t['reason']}\n")
    if df2.empty:
        print("  No trades triggered for Track 2.")
    else:
        for idx, t in df2.iterrows():
            pnl_str = f"+INR {t['pnl']:,.2f}" if t['pnl'] >= 0 else f"-INR {abs(t['pnl']):,.2f}"
            print(f"  Trade #{idx+1} | {t['entry_time']} -> {t['exit_time']} IST | {t['symbol']} ({t['action']})")
            print(f"     Entry: INR {t['entry_p']:.2f} | Exit: INR {t['exit_p']:.2f} | PnL: {pnl_str} ({t['pnl_pct']:+.1f}%)")
            print(f"     Reason: {t['reason']}\n")

    pnl1 = df1["pnl"].sum() if not df1.empty else 0.0
    wins1 = len(df1[df1["pnl"] > 0]) if not df1.empty else 0
    total1 = len(df1)
    wr1 = (wins1 / total1 * 100) if total1 > 0 else 0.0

    pnl2 = df2["pnl"].sum() if not df2.empty else 0.0
    wins2 = len(df2[df2["pnl"] > 0]) if not df2.empty else 0
    total2 = len(df2)
    wr2 = (wins2 / total2 * 100) if total2 > 0 else 0.0

    print("=" * 85)
    print("                   HEAD-TO-HEAD PERFORMANCE COMPARISON")
    print("=" * 85)
    print(f"  {'Metric':<30} | {'Track 1 (Dual-Brain)':<22} | {'Track 2 (Tri-Brain ANN)':<22}")
    print("-" * 85)
    print(f"  {'Total Trades Executed':<30} | {total1:<22} | {total2:<22}")
    print(f"  {'Winning Trades':<30} | {wins1:<22} | {wins2:<22}")
    print(f"  {'Win Rate (%)':<30} | {wr1:<21.1f}% | {wr2:<21.1f}%")
    print(f"  {'Net Realized P&L (INR)':<30} | INR {pnl1:<17.2f} | INR {pnl2:<17.2f}")
    print("=" * 85 + "\n")


if __name__ == "__main__":
    run_today_backtest()
