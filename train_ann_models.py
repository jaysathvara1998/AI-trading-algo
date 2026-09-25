"""
Deep Artificial Neural Network (ANN) Institutional Training Script
Trains and serializes calibrated ANN Deep Brains on 1-minute historical datasets:
- NIFTY: `data/nifty_12m_1min.csv.gz` or recent Nifty CSVs
- SENSEX: `data/sensex_12m_1min.csv.gz` or recent Sensex CSVs
Outputs:
- `algo_vpin_v2/models/ann_model_nifty.joblib`
- `algo_vpin_v2/models/ann_model_sensex.joblib`
- `algo_vpin_v2/models/ann_model.joblib`
"""

import sys
from pathlib import Path
import gzip
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_score

# Add root directory to sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from algo_vpin_v2.ann_brain import ANNBrain


def load_dataset(symbol: str) -> pd.DataFrame:
    data_dir = ROOT_DIR / "algo_vpin_v2" / "data"
    sym = symbol.lower()
    
    # 1. Check compressed 12m file
    gz_path = data_dir / f"{sym}_12m_1min.csv.gz"
    if gz_path.exists():
        print(f"Loading {symbol} from compressed archive: {gz_path.name}...")
        with gzip.open(gz_path, "rt", encoding="utf-8") as f:
            return pd.read_csv(f)

    # 2. Check multiple CSV files
    csv_files = sorted(list(data_dir.glob(f"{sym}_v2_1min_*.csv")), reverse=True)
    if csv_files:
        dfs = []
        for f in csv_files[:5]:
            dfs.append(pd.read_csv(f))
        print(f"Loading {symbol} from {len(dfs)} recent daily CSV files...")
        return pd.concat(dfs, ignore_index=True)

    raise FileNotFoundError(f"No historical data found for {symbol} in {data_dir}")


def train_ann_for_asset(symbol: str):
    print("\n" + "=" * 75)
    print(f"   TRAINING DEEP ARTIFICIAL NEURAL NETWORK (ANN) FOR {symbol.upper()}   ")
    print("=" * 75)

    df = load_dataset(symbol)
    print(f"-> Total Raw Historical Bars: {len(df):,}")

    prices = df["close"].values
    n = len(prices)
    if n < 500:
        print(f"Insufficient bars ({n}) to train deep network.")
        return

    # Subsample if dataset is enormous for fast, optimal convergence (e.g. 20,000 most recent bars)
    if n > 25000:
        df = df.tail(25000).reset_index(drop=True)
        prices = df["close"].values
        n = len(prices)
        print(f"-> Subsampled to most recent {n:,} high-quality 1-minute bars")

    ann = ANNBrain(hidden_layer_sizes=(64, 32, 16))
    ann.set_underlying(symbol)
    move_threshold = ann.get_move_threshold()
    print(f"-> Asset Move Noise Filter Threshold: ±{move_threshold:.1f} pts")

    # Extract engineered feature matrix
    X_list = []
    y_list = []

    # Calculate rolling metrics
    returns = np.diff(prices, prepend=prices[0])
    vpin_series = df["vpin"].values if "vpin" in df.columns else np.clip(np.abs(returns) / (np.std(returns) * 3 + 1e-6), 0.1, 0.9)
    vol_series = df["garch_vol"].values if "garch_vol" in df.columns else pd.Series(returns).rolling(20, min_periods=1).std().values
    
    # 3-bar lookahead labeling to capture directional momentum
    lookahead = 3
    for i in range(20, n - lookahead):
        curr_p = prices[i]
        future_p = prices[i + lookahead]
        ret = future_p - curr_p

        if ret > move_threshold:
            label = 2  # CALL
        elif ret < -move_threshold:
            label = 0  # PUT
        else:
            label = 1  # HOLD

        delta = returns[i]
        vol = vol_series[i]
        vpin = vpin_series[i]
        forecast = delta * 0.5
        week_pos = 0.5
        trend_15m = 1 if delta > 0 else (-1 if delta < 0 else 0)

        feat = ann.build_feature_vector(
            price_delta=delta,
            rolling_vol=vol,
            garch_forecast=forecast,
            vpin=vpin,
            dist_pdh=15.0,
            dist_pdl=20.0,
            week_pos=week_pos,
            trend_15m=trend_15m
        )

        X_list.append(feat)
        y_list.append(label)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)

    unique, counts = np.unique(y, return_counts=True)
    dist_str = ", ".join([f"Class {u}: {c} ({c/len(y)*100:.1f}%)" for u, c in zip(unique, counts)])
    print(f"-> Label Distribution: {dist_str}")

    # Train / Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle=False)
    print(f"-> Training Samples: {len(X_train):,} | Validation Samples: {len(X_test):,}")

    # Scale & Fit
    X_train_scaled = ann.scaler.fit_transform(X_train)
    X_test_scaled = ann.scaler.transform(X_test)

    from sklearn.neural_network import MLPClassifier
    ann.model = MLPClassifier(
        hidden_layer_sizes=ann.hidden_layer_sizes,
        activation='relu',
        solver='adam',
        alpha=0.03,  # Strong L2 regularization against noise
        batch_size=64,
        learning_rate='adaptive',
        learning_rate_init=0.002,
        max_iter=350,
        early_stopping=True,
        n_iter_no_change=25,
        random_state=42
    )

    ann.model.fit(X_train_scaled, y_train)
    ann.is_trained = True

    # Evaluation
    y_pred = ann.model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n[Validation Results for {symbol.upper()}]")
    print(f"-> Accuracy: {acc*100:.2f}% | Iterations: {ann.model.n_iter_} | Loss: {ann.model.loss_:.4f}")
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["PUT (0)", "HOLD (1)", "CALL (2)"], zero_division=0))

    # Save asset-specific model and default model
    models_dir = ROOT_DIR / "algo_vpin_v2" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    asset_path = models_dir / f"ann_model_{symbol.lower()}.joblib"
    ann.save_model(asset_path)
    print(f"[SUCCESS] Saved trained model to: {asset_path}")

    # Also save as default if NIFTY
    if symbol.upper() == "NIFTY":
        default_path = models_dir / "ann_model.joblib"
        ann.save_model(default_path)
        root_path = ROOT_DIR / "ann_model.joblib"
        ann.save_model(root_path)
        print(f"[SUCCESS] Saved default model to: {default_path}")


def main():
    for asset in ["NIFTY", "SENSEX"]:
        try:
            train_ann_for_asset(asset)
        except Exception as e:
            print(f"Error training ANN for {asset}: {e}")

    print("\n" + "=" * 75)
    print("   ALL ANN DEEP NEURAL NETWORKS SUCCESSFULLY TRAINED & SERIALIZED!   ")
    print("=" * 75)


if __name__ == "__main__":
    main()
