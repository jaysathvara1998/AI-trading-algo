"""
Multi-Day Quantitative Stress Test & Cross-Validation for Artificial Neural Network (ANN) Deep Brain
Ingests all real exchange intraday sessions (Sep 01 to Sep 15), trains deep weights,
and benchmarks out-of-sample performance day-by-day.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent))

from algo_vpin_v2.ann_brain import ANNBrain
from algo_vpin_v2.xgboost_brain import XGBoostBrain
from algo_vpin_v2.svm_filter import SVMTradeFilter


def run_multi_day_stress_test():
    print("=" * 80, flush=True)
    print("      MULTI-DAY DEEP NEURAL NETWORK (ANN) BENCHMARK & TRAINING SUITE     ", flush=True)
    print("=" * 80, flush=True)

    data_dir = Path("algo_vpin_v2/data")
    day_files = sorted([
        f for f in data_dir.glob("nifty_v2_1min_*.csv")
    ])

    if not day_files:
        print("No historical session CSVs found in algo_vpin_v2/data", flush=True)
        return

    print(f"Discovered {len(day_files)} historical market sessions:", flush=True)
    for f in day_files:
        print(f"  - {f.name} ({f.stat().st_size / 1024:.1f} KB)", flush=True)

    # Initialize AI Brain Models
    ann = ANNBrain(hidden_layer_sizes=(64, 32))
    ann.model_path = Path("algo_vpin_v2/models/ann_model.joblib")
    xgb = XGBoostBrain()
    svm = SVMTradeFilter()

    cumulative_ann_wins = 0
    cumulative_ann_signals = 0

    cumulative_xgb_wins = 0
    cumulative_xgb_signals = 0

    cumulative_svm_wins = 0
    cumulative_svm_signals = 0

    cumulative_tri_wins = 0
    cumulative_tri_signals = 0

    session_reports = []

    for day_idx, day_file in enumerate(day_files, 1):
        df = pd.read_csv(day_file)
        if len(df) < 10:
            continue

        prices = df["close"].values
        n_bars = len(prices)
        date_str = day_file.stem.replace("nifty_v2_1min_", "")

        print(f"\nProcessing {date_str} ({n_bars} bars)...", flush=True)

        day_ann_wins = 0
        day_ann_signals = 0
        day_tri_wins = 0
        day_tri_signals = 0

        # Initial day warm up
        start_eval_idx = 30 if day_idx == 1 else 1

        for i in range(1, start_eval_idx):
            curr_p = prices[i]
            prev_p = prices[i - 1]
            delta = curr_p - prev_p
            vpin = float(df.get("vpin", pd.Series([0.20]*n_bars)).iloc[i])
            vol = float(df.get("garch_vol", pd.Series([0.35]*n_bars)).iloc[i])
            forecast = float(df.get("garch_mu", pd.Series([0.0]*n_bars)).iloc[i])
            week_pos = float(df.get("week_pos", pd.Series([0.5]*n_bars)).iloc[i])
            trend_15m = int(df.get("trend_15m", pd.Series([0]*n_bars)).iloc[i])

            feat = ann.build_feature_vector(delta, vol, forecast, vpin, 15.0, 25.0, week_pos, trend_15m)
            ann.update_bar(curr_p, delta, vol, forecast, vpin, 15.0, 25.0, week_pos, trend_15m)
            xgb.update_bar(curr_p, feat)
            svm.update_bar(curr_p, delta, vol, forecast)

        if day_idx == 1 and not ann.is_trained:
            ann.train()
            xgb.train_model()
            svm.train_model()

        # Step through evaluation bars
        for i in range(start_eval_idx, n_bars - 1):
            curr_p = prices[i]
            next_p = prices[i + 1]
            actual_move = next_p - curr_p

            delta = curr_p - prices[i - 1]
            vpin = float(df.get("vpin", pd.Series([0.20]*n_bars)).iloc[i])
            vol = float(df.get("garch_vol", pd.Series([0.35]*n_bars)).iloc[i])
            forecast = float(df.get("garch_mu", pd.Series([0.0]*n_bars)).iloc[i])
            week_pos = float(df.get("week_pos", pd.Series([0.5]*n_bars)).iloc[i])
            trend_15m = int(df.get("trend_15m", pd.Series([0]*n_bars)).iloc[i])

            feat = ann.build_feature_vector(delta, vol, forecast, vpin, 15.0, 25.0, week_pos, trend_15m)

            ann_sig, ann_conf, _ = ann.predict(feat)
            xgb_sig, xgb_conf = xgb.predict(feat)
            svm_feat = svm.build_feature_vector(delta, vol, forecast)
            svm_sig, _ = svm.predict(svm_feat)

            # ANN
            if ann_sig != 0:
                day_ann_signals += 1
                cumulative_ann_signals += 1
                if (ann_sig == 1 and actual_move > 0) or (ann_sig == -1 and actual_move < 0):
                    day_ann_wins += 1
                    cumulative_ann_wins += 1

            # XGB
            if xgb_sig != 0:
                cumulative_xgb_signals += 1
                if (xgb_sig == 1 and actual_move > 0) or (xgb_sig == -1 and actual_move < 0):
                    cumulative_xgb_wins += 1

            # SVM
            if svm_sig != 0:
                cumulative_svm_signals += 1
                if (svm_sig == 1 and actual_move > 0) or (svm_sig == -1 and actual_move < 0):
                    cumulative_svm_wins += 1

            # Tri-Brain Consensus
            if ann_sig != 0 and ann_sig == xgb_sig == svm_sig:
                day_tri_signals += 1
                cumulative_tri_signals += 1
                if (ann_sig == 1 and actual_move > 0) or (ann_sig == -1 and actual_move < 0):
                    day_tri_wins += 1
                    cumulative_tri_wins += 1

            # Ingest bar memory
            if ann._pending_feature is not None and ann._pending_price is not None:
                future_ret = curr_p - ann._pending_price
                lbl = 2 if future_ret > 2.5 else (0 if future_ret < -2.5 else 1)
                ann.feature_history.append(ann._pending_feature)
                ann.target_history.append(lbl)
            ann._pending_feature = feat
            ann._pending_price = curr_p

            if xgb._pending_feature is not None and xgb._pending_price is not None:
                f_ret = curr_p - xgb._pending_price
                xgb.feature_history.append(xgb._pending_feature)
                xgb.target_history.append(1 if f_ret >= 0 else 0)
            xgb._pending_feature = feat
            xgb._pending_price = curr_p

            if svm._pending_feature is not None and svm._pending_price is not None:
                f_ret = curr_p - svm._pending_price
                svm.feature_history.append(svm._pending_feature)
                svm.target_history.append(1 if f_ret >= 0 else -1)
            svm._pending_feature = svm_feat
            svm._pending_price = curr_p

        # End of day model update
        ann.train()
        xgb.train_model()
        svm.train_model()

        ann_day_wr = (day_ann_wins / day_ann_signals * 100) if day_ann_signals > 0 else 0.0
        tri_day_wr = (day_tri_wins / day_tri_signals * 100) if day_tri_signals > 0 else 0.0

        session_reports.append({
            "Date": date_str,
            "Bars": n_bars,
            "ANN Signals": day_ann_signals,
            "ANN Win Rate": f"{ann_day_wr:.1f}%",
            "Tri-Brain Signals": day_tri_signals,
            "Tri-Brain Win Rate": f"{tri_day_wr:.1f}%"
        })

    # Save trained model
    ann.save_model()
    print("\n" + "=" * 80, flush=True)
    print("                     DAY-BY-DAY PERFORMANCE BREAKDOWN                     ", flush=True)
    print("=" * 80, flush=True)
    rep_df = pd.DataFrame(session_reports)
    print(rep_df.to_string(index=False), flush=True)

    print("\n" + "=" * 80, flush=True)
    print("                 CUMULATIVE MULTI-DAY BENCHMARK TOTALS                    ", flush=True)
    print("=" * 80, flush=True)
    tot_ann_wr = (cumulative_ann_wins / cumulative_ann_signals * 100) if cumulative_ann_signals > 0 else 0.0
    tot_xgb_wr = (cumulative_xgb_wins / cumulative_xgb_signals * 100) if cumulative_xgb_signals > 0 else 0.0
    tot_svm_wr = (cumulative_svm_wins / cumulative_svm_signals * 100) if cumulative_svm_signals > 0 else 0.0
    tot_tri_wr = (cumulative_tri_wins / cumulative_tri_signals * 100) if cumulative_tri_signals > 0 else 0.0

    print(f"Total Combined Bars Processed         : {sum(r['Bars'] for r in session_reports):,}", flush=True)
    print(f"1. ANN Deep Neural Network Standalone : {tot_ann_wr:.2f}% Win Rate ({cumulative_ann_wins}/{cumulative_ann_signals} Trades)", flush=True)
    print(f"2. XGBoost Decision Tree              : {tot_xgb_wr:.2f}% Win Rate ({cumulative_xgb_wins}/{cumulative_xgb_signals} Trades)", flush=True)
    print(f"3. SVM Support Vector Machine         : {tot_svm_wr:.2f}% Win Rate ({cumulative_svm_wins}/{cumulative_svm_signals} Trades)", flush=True)
    print("-" * 80, flush=True)
    print(f"[TRI-BRAIN HIGH-CONVICTION CONSENSUS] : {tot_tri_wr:.2f}% WIN RATE ({cumulative_tri_wins}/{cumulative_tri_signals} Setups)", flush=True)
    print("=" * 80, flush=True)
    print(f"Trained Deep Neural Network saved to: {ann.model_path}", flush=True)


if __name__ == "__main__":
    run_multi_day_stress_test()
