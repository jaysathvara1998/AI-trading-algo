"""
NATGASMINI Live & Replay Quantitative Testing Engine
Enables live real-time and historical replay testing of Algo VPIN v2.0
with Chinmay Trading Skills on MCX Natural Gas Mini (NATGASMINI).
"""

import sys
import time
import argparse
from pathlib import Path
from datetime import datetime, timedelta
import pytz
import pandas as pd
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

from algo_vpin_v2.config import CONFIG, AppConfig
from algo_vpin_v2.data_feed import DhanDataFeed
from algo_vpin_v2.macro_features import MacroFeatureEngine
from algo_vpin_v2.vpin import VPINCalculator
from algo_vpin_v2.garch_engine import GARCHEngine, DirectionalSignal
from algo_vpin_v2.smc_engine import SMCEngine
from algo_vpin_v2.ensemble_brain import EnsembleBrain


NATGAS_FUT_SEC_ID = "568246"  # NATGASMINI-25Sep2026-FUT
NATGAS_EXCH_SEG = "MCX_COMM"
NATGAS_INST_TYPE = "FUTCOM"
NATGAS_LOT_SIZE = 250  # 250 mmBtu per mini contract


def run_natgasmini_backtest(is_live: bool = False):
    CONFIG.market.set_symbol("NATGASMINI")
    feed = DhanDataFeed(CONFIG)
    dhan = feed.tradehull_client.Dhan if feed.tradehull_client else None
    if dhan is None:
        print("[ERROR] Dhan client not authenticated. Please check credentials.")
        return

    tz = pytz.timezone("Asia/Kolkata")
    today_str = datetime.now(tz).strftime("%Y-%m-%d")

    print("=" * 90)
    print(f"      TESTING ALGO & AI PREDICTIONS ON MCX NATGASMINI ({today_str})")
    print(f"      Instrument: NATGASMINI-25Sep2026-FUT (SecID: {NATGAS_FUT_SEC_ID}) | Lot: {NATGAS_LOT_SIZE}")
    print("=" * 90)

    # 1. Fetch Today's 1-Minute Exchange Bars
    print("[1/4] Fetching today's 1-minute exchange bars for NATGASMINI from Dhan API...")
    r_today = dhan.intraday_minute_data(
        security_id=NATGAS_FUT_SEC_ID,
        exchange_segment=NATGAS_EXCH_SEG,
        instrument_type=NATGAS_INST_TYPE,
        from_date=today_str,
        to_date=today_str
    )
    if not (isinstance(r_today, dict) and r_today.get("status") == "success" and "data" in r_today):
        print(f"[ERROR] Failed to fetch NATGASMINI bars: {r_today}")
        return

    today_df = pd.DataFrame(r_today["data"])
    today_df["timestamp"] = pd.to_datetime(today_df["timestamp"], unit="s", utc=True).dt.tz_convert(tz)
    today_df = today_df.sort_values("timestamp").reset_index(drop=True)
    t_start = today_df.iloc[0]["timestamp"].strftime("%H:%M")
    t_end = today_df.iloc[-1]["timestamp"].strftime("%H:%M")
    print(f"Loaded {len(today_df)} exchange bars for NATGASMINI (from {t_start} to {t_end} IST).")
    print(f"Day Range: Low = {today_df['low'].min():.1f} | High = {today_df['high'].max():.1f} | Last = {today_df.iloc[-1]['close']:.1f}")

    # 2. Initialize Macro, VPIN, GARCH, SMC, and AI Ensemble Brain
    print("[2/4] Initializing Quant Pipeline with Chinmay Trading Skills...")
    macro_eng = MacroFeatureEngine()
    vpin_calc = VPINCalculator()
    garch_eng = GARCHEngine()
    smc_eng = SMCEngine()

    ensemble = EnsembleBrain()
    ensemble.load_all_models()

    # 3. Cache Option Candles for NATGASMINI strikes
    print("[3/4] Caching real option strike candles from Dhan MCX...")
    option_cache = {}
    inst_df = feed.tradehull_client.instrument_df

    def get_option_bars(sec_id_str: str):
        if sec_id_str not in option_cache:
            r = dhan.intraday_minute_data(
                security_id=sec_id_str,
                exchange_segment="MCX_COMM",
                instrument_type="OPTFUT",
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

    # Pre-cache key active strikes (270 CE/PE, 275 CE/PE, 280 CE/PE)
    key_opts = {
        "NATURALGASM 23 SEP 270 CALL": "578430",
        "NATURALGASM 23 SEP 270 PUT": "578510",
        "NATURALGASM 23 SEP 275 CALL": "578429",
        "NATURALGASM 23 SEP 275 PUT": "578509",
        "NATURALGASM 23 SEP 280 CALL": "578428",
        "NATURALGASM 23 SEP 280 PUT": "578508"
    }
    for opt_name, sid in key_opts.items():
        odf = get_option_bars(sid)
        if odf is not None:
            print(f"   Cached {opt_name}: {len(odf)} bars")

    def get_live_opt_price(opt_sym: str, spot_p: float, time_str: str) -> float:
        sec_row = inst_df[(inst_df["SEM_CUSTOM_SYMBOL"] == opt_sym) | (inst_df["SEM_TRADING_SYMBOL"] == opt_sym)]
        if not sec_row.empty:
            sid = str(sec_row.iloc[-1]["SEM_SMST_SECURITY_ID"])
            odf = get_option_bars(sid)
            if odf is not None and time_str in odf.index:
                return float(odf.loc[time_str]["close"])
            elif odf is not None and len(odf) > 0:
                return float(odf.iloc[-1]["close"])
        import re
        m = re.search(r"(\d{3})", opt_sym)
        strike = float(m.group(1)) if m else 275.0
        is_c = "CALL" in opt_sym or "CE" in opt_sym
        intrinsic = max(0.0, spot_p - strike) if is_c else max(0.0, strike - spot_p)
        return round(max(0.50, intrinsic + 2.80), 2)

    # 4. Simulation Engine
    print("[4/4] Executing Replay Simulation across NATGASMINI with Chinmay Trading Skills...")
    track1_pos = None
    track2_pos = None
    track1_trades = []
    track2_trades = []

    recent_highs = []
    recent_lows = []

    for i in range(len(today_df)):
        row = today_df.iloc[i]
        p = float(row["close"])
        o = float(row.get("open", p))
        h = float(row.get("high", p))
        l = float(row.get("low", p))
        v = float(row["volume"])
        ts = row["timestamp"]
        time_str = ts.strftime("%H:%M")

        recent_highs.append(h)
        recent_lows.append(l)

        mst = macro_eng.update_1min_bar(p, h, l)
        vres = vpin_calc.process_bar(p, v, ts)
        gres = garch_eng.add_bar(p, ts)
        smc_st = smc_eng.update_bar(o, h, l, p, v)

        if gres is None or i < 30:
            continue

        prev_o = float(today_df.iloc[i - 1].get("open", p))
        prev_c = float(today_df.iloc[i - 1]["close"])
        p_delta = p - prev_c
        sigma = gres.sigma_next
        mu = gres.mu_next

        key_levels = [today_df["low"].iloc[:i+1].min(), today_df["high"].iloc[:i+1].max()]

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
            underlying="NATGASMINI"
        )

        # -------------------------------------------------------------
        # TRACK 1 (Dual-Brain + Chinmay Skills)
        # -------------------------------------------------------------
        if track1_pos is not None:
            opt_p = get_live_opt_price(track1_pos["symbol"], p, time_str)
            guardian_res = ensemble.position_guardian_skill.evaluate(
                symbol=track1_pos["symbol"],
                is_call=("CALL" in track1_pos["symbol"] or "CE" in track1_pos["symbol"]),
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
                pnl = (opt_p - track1_pos["entry_p"]) * NATGAS_LOT_SIZE
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

        if track1_pos is None and (ts.hour < 23 or (ts.hour == 23 and ts.minute <= 15)):
            if not dec.is_vetoed and dec.final_action in (DirectionalSignal.BUY, DirectionalSignal.SELL):
                act = "BUY" if dec.final_action == DirectionalSignal.BUY else "SELL"
                sym = feed.resolve_option_strike(act, p)
                opt_p = get_live_opt_price(sym, p, time_str)

                rr_plan = dec.risk_reward_result.metadata.get("plan") if dec.risk_reward_result else None
                if rr_plan is not None:
                    sl_pts = max(0.50, rr_plan.option_risk_pts)
                    tp_pts = max(0.75, rr_plan.option_reward_pts)
                    spot_sl = rr_plan.spot_sl
                    rr_ratio = rr_plan.risk_reward_ratio
                else:
                    sl_pts = max(0.50, opt_p * 0.15)
                    tp_pts = sl_pts * 1.5
                    spot_sl = (p - 1.5) if act == "BUY" else (p + 1.5)
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
        # TRACK 2 (Tri-Brain ANN + Chinmay Skills)
        # -------------------------------------------------------------
        if track2_pos is not None:
            opt_p = get_live_opt_price(track2_pos["symbol"], p, time_str)
            guardian_res = ensemble.position_guardian_skill.evaluate(
                symbol=track2_pos["symbol"],
                is_call=("CALL" in track2_pos["symbol"] or "CE" in track2_pos["symbol"]),
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
                pnl = (opt_p - track2_pos["entry_p"]) * NATGAS_LOT_SIZE
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

        if track2_pos is None and (ts.hour < 23 or (ts.hour == 23 and ts.minute <= 15)):
            if not dec.tri_is_vetoed and dec.tri_final_action in (DirectionalSignal.BUY, DirectionalSignal.SELL):
                act = "BUY" if dec.tri_final_action == DirectionalSignal.BUY else "SELL"
                sym = feed.resolve_option_strike(act, p)
                opt_p = get_live_opt_price(sym, p, time_str)

                rr_plan = dec.risk_reward_result.metadata.get("plan") if dec.risk_reward_result else None
                if rr_plan is not None:
                    sl_pts = max(0.50, rr_plan.option_risk_pts)
                    tp_pts = max(0.75, rr_plan.option_reward_pts)
                    spot_sl = rr_plan.spot_sl
                    rr_ratio = rr_plan.risk_reward_ratio
                else:
                    sl_pts = max(0.50, opt_p * 0.15)
                    tp_pts = sl_pts * 1.5
                    spot_sl = (p - 1.5) if act == "BUY" else (p + 1.5)
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

    # Account for any active position currently still open at last bar
    for trk_name, pos, trk_trades in [("Track 1", track1_pos, track1_trades), ("Track 2", track2_pos, track2_trades)]:
        if pos is not None:
            last_p = float(today_df.iloc[-1]["close"])
            last_t = today_df.iloc[-1]["timestamp"].strftime("%H:%M")
            cur_opt = get_live_opt_price(pos["symbol"], last_p, last_t)
            pnl = (cur_opt - pos["entry_p"]) * NATGAS_LOT_SIZE
            trk_trades.append({
                "entry_time": pos["entry_time"],
                "exit_time": f"{last_t} (OPEN)",
                "symbol": pos["symbol"],
                "action": pos["action"],
                "entry_p": pos["entry_p"],
                "exit_p": cur_opt,
                "pnl": pnl,
                "pnl_pct": ((cur_opt - pos["entry_p"]) / pos["entry_p"]) * 100,
                "reason": f"CURRENTLY_ACTIVE (Mark-to-Market @ {last_t} IST | Spot {last_p:.1f})",
                "pattern": pos.get("pattern", "Micro-Momentum"),
                "rr_ratio": pos.get("rr_ratio", 1.5)
            })

    # 5. Print Results
    def print_journal(trades, track_title):
        print("\n" + "=" * 90)
        print(f"      {track_title} NATGASMINI TRADE JOURNAL")
        print("=" * 90)
        if not trades:
            print("  No trades executed.")
            return
        total_pnl = 0.0
        wins = 0
        for idx, t in enumerate(trades, 1):
            total_pnl += t["pnl"]
            is_win = t["pnl"] > 0
            if is_win:
                wins += 1
            sign = "+" if t["pnl"] >= 0 else "-"
            print(f"  Trade #{idx} | {t['entry_time']} -> {t['exit_time']} IST | {t['symbol']} ({t['action']})")
            print(f"     Setup: {t['pattern']} | Target R:R: 1:{t['rr_ratio']:.2f}")
            print(f"     Entry: INR {t['entry_p']:.2f} | Exit: INR {t['exit_p']:.2f} | PnL: {sign}INR {abs(t['pnl']):.2f} ({t['pnl_pct']:+.1f}%)")
            print(f"     Reason: {t['reason']}\n")

    print_journal(track1_trades, "TRACK 1 (DUAL-BRAIN + CHINMAY SKILLS)")
    print_journal(track2_trades, "TRACK 2 (TRI-BRAIN ANN + CHINMAY SKILLS)")

    def calc_stats(trades):
        total = len(trades)
        wins = sum(1 for t in trades if t["pnl"] > 0)
        wr = (wins / total * 100) if total > 0 else 0.0
        pnl = sum(t["pnl"] for t in trades)
        return total, wins, wr, pnl

    t1_tot, t1_w, t1_wr, t1_pnl = calc_stats(track1_trades)
    t2_tot, t2_w, t2_wr, t2_pnl = calc_stats(track2_trades)

    print("=" * 90)
    print("                   HEAD-TO-HEAD PERFORMANCE COMPARISON (NATGASMINI)")
    print("=" * 90)
    print(f"  {'Metric':<32} | {'Track 1 (Dual-Brain)':<24} | {'Track 2 (Tri-Brain ANN)':<24}")
    print("-" * 90)
    print(f"  {'Total Trades Executed':<32} | {t1_tot:<24} | {t2_tot:<24}")
    print(f"  {'Winning Trades':<32} | {t1_w:<24} | {t2_w:<24}")
    print(f"  {'Win Rate (%)':<32} | {t1_wr:<22.1f} % | {t2_wr:<22.1f} %")
    sign1 = "+" if t1_pnl >= 0 else ""
    sign2 = "+" if t2_pnl >= 0 else ""
    print(f"  {'Net Realized P&L (INR)':<32} | INR {sign1}{t1_pnl:<20.2f} | INR {sign2}{t2_pnl:<20.2f}")
    print("=" * 90)

    # 6. Live Mode (if requested)
    if is_live:
        print("\n" + "=" * 90)
        print("      ENTERING LIVE MONITORING MODE FOR NATGASMINI (MCX OPEN UNTIL 23:30 IST)")
        print("      Listening for new 1-minute exchange bars & evaluating real-time AI predictions...")
        print("=" * 90)
        last_seen_ts = today_df.iloc[-1]["timestamp"]
        while True:
            try:
                time.sleep(15)
                r_now = dhan.intraday_minute_data(
                    security_id=NATGAS_FUT_SEC_ID,
                    exchange_segment=NATGAS_EXCH_SEG,
                    instrument_type=NATGAS_INST_TYPE,
                    from_date=today_str,
                    to_date=today_str
                )
                if isinstance(r_now, dict) and "data" in r_now and len(r_now["data"]) > 0:
                    latest_df = pd.DataFrame(r_now["data"])
                    latest_df["timestamp"] = pd.to_datetime(latest_df["timestamp"], unit="s", utc=True).dt.tz_convert(tz)
                    new_bars = latest_df[latest_df["timestamp"] > last_seen_ts]
                    for _, nbar in new_bars.iterrows():
                        cur_p = float(nbar["close"])
                        cur_ts = nbar["timestamp"]
                        last_seen_ts = cur_ts
                        cur_gres = garch_eng.add_bar(cur_p, cur_ts)
                        cur_mst = macro_eng.update_1min_bar(cur_p, float(nbar["high"]), float(nbar["low"]))
                        cur_vres = vpin_calc.process_bar(cur_p, float(nbar["volume"]), cur_ts)
                        cur_smc = smc_eng.update_bar(float(nbar["open"]), float(nbar["high"]), float(nbar["low"]), cur_p, float(nbar["volume"]))
                        cur_dec = ensemble.evaluate(
                            garch_signal=cur_gres.signal if cur_gres else DirectionalSignal.HOLD,
                            price_delta=cur_p - float(latest_df.iloc[-2]["close"]),
                            rolling_vol=cur_gres.sigma_next if cur_gres else 0.001,
                            garch_forecast=cur_gres.mu_next if cur_gres else 0.0,
                            vpin=cur_vres.vpin,
                            macro_state=cur_mst,
                            smc_state=cur_smc,
                            open_p=float(nbar["open"]),
                            high_p=float(nbar["high"]),
                            low_p=float(nbar["low"]),
                            close_p=cur_p,
                            prev_open=float(latest_df.iloc[-2]["open"]),
                            prev_close=float(latest_df.iloc[-2]["close"]),
                            recent_highs=latest_df["high"].tail(10).tolist(),
                            recent_lows=latest_df["low"].tail(10).tolist(),
                            key_levels=[latest_df["low"].min(), latest_df["high"].max()],
                            underlying="NATGASMINI"
                        )
                        t_now = cur_ts.strftime("%H:%M:%S")
                        print(f"[{t_now} IST] NATGASMINI: INR {cur_p:.2f} | Brain Action: {cur_dec.final_action.name} (Conf: {cur_dec.xgb_confidence:.2f}) | Reason: {cur_dec.reason}")
            except KeyboardInterrupt:
                print("\n[INFO] Stopped live monitoring mode.")
                break
            except Exception as e:
                print(f"[WARNING] Live loop tick notice: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Algo & AI Predictions on NATGASMINI")
    parser.add_argument("--live", action="store_true", help="Keep running in live monitoring mode")
    args = parser.parse_args()
    run_natgasmini_backtest(is_live=args.live)
