"""
Validation & Visual Benchmark Suite for 2D CNN Chart-Vision AI Brain
Evaluates accuracy, precision, recall, and pattern recognition capability on out-of-sample chart snapshots.
"""

import sys
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
import logging
from sklearn.metrics import classification_report, confusion_matrix

from vision_chart_brain.visual_predictor import VisualPredictor
from vision_chart_brain.chart_dataset_builder import ChartDatasetBuilder
from vision_chart_brain.train_vision_brain import load_training_bars

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")
logger = logging.getLogger("vision_chart_brain.benchmark")


def run_visual_benchmark():
    logger.info("=" * 75)
    logger.info("  2D CNN CANDLESTICK CHART-VISION AI BRAIN - BENCHMARK & EVALUATION  ")
    logger.info("=" * 75)

    predictor = VisualPredictor()
    if predictor.model is None:
        logger.error("No trained vision model found! Run train_vision_brain.py first.")
        return

    df_bars = load_training_bars()
    builder = ChartDatasetBuilder(window_bars=20, forward_horizon=5, impulse_threshold_pts=10.0)
    X, y_true = builder.build_dataset_from_df(df_bars)

    # Use out-of-sample last 25% for test benchmark
    test_split = int(len(X) * 0.75)
    X_test = X[test_split:]
    y_test = y_true[test_split:]

    logger.info(f"Evaluating model on {len(X_test)} out-of-sample visual chart windows...")

    import torch
    import torch.nn.functional as F

    predictor.model.eval()
    with torch.no_grad():
        tensor_test = torch.from_numpy(X_test).to(predictor.device)
        logits = predictor.model(tensor_test)
        probs = F.softmax(logits, dim=-1).cpu().numpy()
        y_pred = np.argmax(probs, axis=1)

    classes = ["CHOP / HOLD (0)", "BUY CALL (1)", "BUY PUT (2)"]
    print("\n--- CLASSIFICATION REPORT ---")
    print(classification_report(y_test, y_pred, target_names=classes, zero_division=0))

    print("--- CONFUSION MATRIX ---")
    cm = confusion_matrix(y_test, y_pred)
    print(pd.DataFrame(cm, index=[f"True {c}" for c in classes], columns=[f"Pred {c}" for c in classes]))

    acc = np.mean(y_test == y_pred) * 100
    print(f"\n>> Out-Of-Sample Overall Accuracy: {acc:.2f}%\n")
    logger.info("=" * 75)


if __name__ == "__main__":
    run_visual_benchmark()
