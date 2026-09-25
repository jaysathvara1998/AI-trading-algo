"""
Training Pipeline for 2D CNN Candlestick Chart-Vision AI Brain
Ingests historical exchange bars, renders visual datasets, trains deep ConvNet,
and saves validated PyTorch model weights to disk.
"""

import sys
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import logging

from vision_chart_brain.chart_renderer import FastChartRenderer
from vision_chart_brain.chart_dataset_builder import ChartDatasetBuilder
from vision_chart_brain.cnn_vision_model import CandlestickCNN
from algo_vpin_v2.data_feed import DhanDataFeed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [VisionBrain]: %(message)s"
)
logger = logging.getLogger("vision_chart_brain.train")


def load_training_bars() -> pd.DataFrame:
    """Loads historical 1-minute exchange bars from Dhan data feed or local data directory"""
    logger.info("Fetching real historical 1-minute exchange bars for training...")
    try:
        feed = DhanDataFeed()
        df = feed.fetch_historical_bars(days=10)
        if df is not None and len(df) >= 200:
            logger.info(f"Loaded {len(df)} historical bars from Dhan.")
            return df
    except Exception as e:
        logger.warning(f"Error fetching from Dhan live feed: {e}")

    # Fallback to local csv if feed offline
    data_dir = ROOT_DIR / "algo_vpin_v2" / "data"
    bar_csv = data_dir / "historical_bars.csv"
    if bar_csv.exists():
        logger.info(f"Loading bars from {bar_csv}...")
        df = pd.read_csv(bar_csv)
        return df

    # If no local data, synthesize realistic Brownian motion bars with trend spikes
    logger.info("Synthesizing realistic 2,500-bar multi-regime training dataset...")
    np.random.seed(42)
    n_bars = 2500
    prices = [23200.0]
    for _ in range(n_bars - 1):
        ret = np.random.normal(0.00005, 0.0012)
        # Add random impulse trends
        if np.random.rand() < 0.05:
            ret += np.random.choice([0.004, -0.004])
        prices.append(prices[-1] * (1.0 + ret))

    opens = np.array(prices)
    highs = opens * (1.0 + np.abs(np.random.normal(0.0, 0.0008, n_bars)))
    lows = opens * (1.0 - np.abs(np.random.normal(0.0, 0.0008, n_bars)))
    closes = opens + np.random.uniform(-0.5, 0.5, n_bars) * (highs - lows)
    volumes = np.random.uniform(20000, 150000, n_bars)

    df_synth = pd.DataFrame({
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes
    })
    return df_synth


def train_vision_model(epochs: int = 15, batch_size: int = 32, lr: float = 0.001):
    df_bars = load_training_bars()
    builder = ChartDatasetBuilder(window_bars=20, forward_horizon=5, impulse_threshold_pts=10.0)

    X, y = builder.build_dataset_from_df(df_bars)
    
    # Train/Validation Split (80% / 20%)
    n_samples = len(X)
    split_idx = int(n_samples * 0.80)

    X_train, y_train = X[:split_idx], y[:split_idx]
    X_val, y_val = X[split_idx:], y[split_idx:]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device} | Train samples: {len(X_train)} | Val samples: {len(X_val)}")

    # Class weights to balance CHOP vs Breakouts
    class_counts = np.bincount(y_train, minlength=3)
    total_train = len(y_train)
    class_weights = total_train / (3.0 * np.maximum(class_counts, 1))
    weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = CandlestickCNN(num_classes=3).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0
    save_path = Path(__file__).resolve().parent / "vision_cnn_model.pt"

    logger.info("=" * 70)
    logger.info("STARTING 2D CNN VISION BRAIN TRAINING (Candlestick Pattern Recognition)")
    logger.info("=" * 70)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train_items = 0

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_X.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct_train += (preds == batch_y).sum().item()
            total_train_items += batch_y.size(0)

        scheduler.step()
        train_loss = running_loss / total_train_items
        train_acc = correct_train / total_train_items

        # Validation Pass
        model.eval()
        val_loss_run = 0.0
        correct_val = 0
        total_val_items = 0

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                v_loss = criterion(outputs, batch_y)
                val_loss_run += v_loss.item() * batch_X.size(0)
                preds = torch.argmax(outputs, dim=1)
                correct_val += (preds == batch_y).sum().item()
                total_val_items += batch_y.size(0)

        val_loss = val_loss_run / total_val_items
        val_acc = correct_val / total_val_items

        logger.info(
            f"Epoch [{epoch:02d}/{epochs:02d}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.1f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.1f}%"
        )

        if val_acc >= best_val_acc or epoch == epochs:
            best_val_acc = val_acc
            torch.save(model.state_dict(), str(save_path))

    logger.info("=" * 70)
    logger.info(f"TRAINING COMPLETE! Best Validation Accuracy: {best_val_acc*100:.1f}%")
    logger.info(f"Model saved to: {save_path}")
    logger.info("=" * 70)


if __name__ == "__main__":
    train_vision_model(epochs=15, batch_size=32, lr=0.001)
