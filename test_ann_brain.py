"""
Test & Benchmark Suite for Artificial Neural Network (ANN) Deep Brain vs XGBoost vs SVM.
Loads real intraday exchange dataset and evaluates deep pattern extraction accuracy, win rates, and consensus.
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


def run_benchmark():
    print("=" * 70)
    print("ARTIFICIAL NEURAL NETWORK (ANN) DEEP BRAIN BENCHMARK SUITE")
    print("=" * 70)

    # Load real 1-min data
    csv_paths = [
        Path("algo_vpin_v2/data/nifty_v2_1min_20260915.csv"),
        Path("algo_vpin_v2/data/nifty_v2_1min_20260904.csv"),
        Path("algo_vpin_v2/data/nifty_v2_1min_20260902.csv"),
    ]
    
    valid_csv = None
    for p in csv_paths:
        if p.exists():
            valid_csv = p
            break

    if not valid_csv:
        print("No CSV found. Creating synthetic market sequence for verification.")
        df = pd.DataFrame({
            "close": 23400 + np.cumsum(np.random.randn(500) * 8),
            "vpin": 0.22 + np.random.rand(500) * 0.15,
            "garch_vol": 0.35 + np.random.rand(500) * 0.20,
            "garch_mu": np.random.randn(500) * 0.0003,
            "week_pos": np.random.rand(500),
            "trend_15m": np.random.choice([-1, 0, 1], 500)
        })
    else:
        print(f"Loading real exchange dataset from: {valid_csv}")
        df = pd.read_csv(valid_csv)

    print(f"Total historical bars loaded: {len(df)}")

    # Initialize models
    ann = ANNBrain(hidden_layer_sizes=(64, 32))
    xgb = XGBoostBrain()
    svm = SVMTradeFilter()

    # Feed bars into models
    prices = df["close"].values
    n_bars = len(prices)

    warm_up_bars = min(150, n_bars // 2)
    print(f"\n[Phase 1] Ingesting {warm_up_bars} bars for Deep Architecture Warm-Up...")

    for i in range(1, warm_up_bars):
        curr_p = prices[i]
        prev_p = prices[i - 1]
        delta = curr_p - prev_p
        vpin = float(df.get("vpin", pd.Series([0.22]*n_bars)).iloc[i])
        vol = float(df.get("garch_vol", pd.Series([0.35]*n_bars)).iloc[i])
        forecast = float(df.get("garch_mu", pd.Series([0.0]*n_bars)).iloc[i])
        week_pos = float(df.get("week_pos", pd.Series([0.5]*n_bars)).iloc[i])
        trend_15m = int(df.get("trend_15m", pd.Series([0]*n_bars)).iloc[i])

        feat = ann.build_feature_vector(delta, vol, forecast, vpin, 15.0, 25.0, week_pos, trend_15m)

        ann.update_bar(
            current_price=curr_p, price_delta=delta, rolling_vol=vol,
            garch_forecast=forecast, vpin=vpin, dist_pdh=15.0, dist_pdl=25.0,
            week_pos=week_pos, trend_15m=trend_15m
        )
        xgb.update_bar(
            current_price=curr_p,
            feature_vector=feat
        )
        svm.update_bar(
            current_price=curr_p, price_delta=delta, rolling_vol=vol,
            garch_forecast=forecast
        )

    # Force Train
    ann.train()
    xgb.train_model()
    svm.train_model()

    print(f"-> ANN Neural Network Status: Trained={ann.is_trained} (Hidden Layers: {ann.hidden_layer_sizes})")
    print(f"-> XGBoost Status: Trained={xgb.is_trained}")
    print(f"-> SVM Status: Trained={svm.is_trained}")

    print("\n" + "=" * 70)
    print(f"[Phase 2] Forward Inference Across Next {n_bars - warm_up_bars} Out-of-Sample Bars")
    print("=" * 70)

    # Metrics Tracking: (Wins, Losses, Total Signals Triggered)
    ann_wins = 0
    ann_signals = 0

    xgb_wins = 0
    xgb_signals = 0

    svm_wins = 0
    svm_signals = 0

    tri_wins = 0
    tri_signals = 0

    for i in range(warm_up_bars, n_bars - 1):
        curr_p = prices[i]
        next_p = prices[i + 1]
        actual_move = next_p - curr_p

        delta = curr_p - prices[i - 1]
        vpin = float(df.get("vpin", pd.Series([0.22]*n_bars)).iloc[i])
        vol = float(df.get("garch_vol", pd.Series([0.35]*n_bars)).iloc[i])
        forecast = float(df.get("garch_mu", pd.Series([0.0]*n_bars)).iloc[i])
        week_pos = float(df.get("week_pos", pd.Series([0.5]*n_bars)).iloc[i])
        trend_15m = int(df.get("trend_15m", pd.Series([0]*n_bars)).iloc[i])

        feat = ann.build_feature_vector(delta, vol, forecast, vpin, 15.0, 25.0, week_pos, trend_15m)

        ann_sig, ann_conf, ann_probs = ann.predict(feat)
        xgb_sig, xgb_conf = xgb.predict(feat)
        svm_feat = svm.build_feature_vector(delta, vol, forecast)
        svm_sig, svm_dist = svm.predict(svm_feat)

        # 1. ANN Evaluation
        if ann_sig != 0:
            ann_signals += 1
            if (ann_sig == 1 and actual_move > 0) or (ann_sig == -1 and actual_move < 0):
                ann_wins += 1

        # 2. XGBoost Evaluation
        if xgb_sig != 0:
            xgb_signals += 1
            if (xgb_sig == 1 and actual_move > 0) or (xgb_sig == -1 and actual_move < 0):
                xgb_wins += 1

        # 3. SVM Evaluation
        if svm_sig != 0:
            svm_signals += 1
            if (svm_sig == 1 and actual_move > 0) or (svm_sig == -1 and actual_move < 0):
                svm_wins += 1

        # 4. Tri-Brain Consensus (ANN + XGBoost + SVM Agreement)
        if ann_sig != 0 and ann_sig == xgb_sig == svm_sig:
            tri_signals += 1
            if (ann_sig == 1 and actual_move > 0) or (ann_sig == -1 and actual_move < 0):
                tri_wins += 1

        # Online continuous updates
        ann.update_bar(curr_p, delta, vol, forecast, vpin, 15.0, 25.0, week_pos, trend_15m)
        xgb.update_bar(curr_p, feat)
        svm.update_bar(curr_p, delta, vol, forecast)

    print("\n======================================================================")
    print("                     BENCHMARK RESULTS & METRICS                      ")
    print("======================================================================")
    ann_wr = (ann_wins / ann_signals * 100) if ann_signals > 0 else 0.0
    xgb_wr = (xgb_wins / xgb_signals * 100) if xgb_signals > 0 else 0.0
    svm_wr = (svm_wins / svm_signals * 100) if svm_signals > 0 else 0.0
    tri_wr = (tri_wins / tri_signals * 100) if tri_signals > 0 else 0.0

    print(f"1. ANN Deep Neural Network : {ann_wr:.2f}% Win Rate ({ann_wins}/{ann_signals} Active Signals)")
    print(f"2. XGBoost Decision Tree   : {xgb_wr:.2f}% Win Rate ({xgb_wins}/{xgb_signals} Active Signals)")
    print(f"3. SVM Support Vector      : {svm_wr:.2f}% Win Rate ({svm_wins}/{svm_signals} Active Signals)")
    print("-" * 70)
    print(f"[TRI-BRAIN CONSENSUS] (ANN + XGB + SVM Agree) : {tri_wr:.2f}% WIN RATE ({tri_wins}/{tri_signals} High-Conviction Setups)")
    print("======================================================================")


if __name__ == "__main__":
    run_benchmark()
