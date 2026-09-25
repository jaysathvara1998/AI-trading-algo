"""
Multi-Asset 2D CNN Computer Vision Chart Brain Trainer
Ingests historical 1-minute bars for NIFTY & SENSEX,
renders thousands of 128x128 RGB candlestick chart images in memory,
and trains deep PyTorch Convolutional Neural Networks for visual pattern confirmation:
  - Class 0: Consolidation Trap / Indecision (HOLD)
  - Class 1: Bullish Momentum Breakout (BUY CALL)
  - Class 2: Bearish Rejection Breakdown (BUY PUT)
"""

import os
import sys
from pathlib import Path
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from vision_chart_brain.chart_renderer import FastChartRenderer
from vision_chart_brain.cnn_vision_model import CandlestickCNN

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("vision_chart_brain.multi_asset_trainer")


INDEX_VISION_THRESHOLDS = {
    "NIFTY": 7.0,
    "SENSEX": 22.0,
    "BANKNIFTY": 16.0,
}


def build_vision_dataset(csv_path: Path, window_bars: int = 20, step_bars: int = 6, forward_window: int = 8, symbol: str = "NIFTY"):
    logger.info(f"Building 2D CNN vision dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    renderer = FastChartRenderer(image_size=128, window_bars=window_bars)

    images = []
    labels = []

    closes = df["close"].values
    highs = df["high"].values if "high" in df.columns else closes
    lows = df["low"].values if "low" in df.columns else closes
    opens = df["open"].values if "open" in df.columns else closes
    volumes = df["volume"].values if "volume" in df.columns else np.ones(len(closes))

    n_bars = len(df)
    sym_clean = symbol.strip().upper()
    threshold = INDEX_VISION_THRESHOLDS.get(sym_clean, 7.0)

    logger.info(f"Rendering {sym_clean} candlestick snapshots across {n_bars:,} bars (Horizon: {forward_window}m, Threshold: ±{threshold} pts)...")

    for i in range(window_bars, n_bars - forward_window, step_bars):
        w_open = opens[i - window_bars:i]
        w_high = highs[i - window_bars:i]
        w_low = lows[i - window_bars:i]
        w_close = closes[i - window_bars:i]
        w_vol = volumes[i - window_bars:i]

        sub_df = pd.DataFrame({
            "open": w_open, "high": w_high, "low": w_low, "close": w_close, "volume": w_vol
        })

        img_arr = renderer.render_ohlc(sub_df)
        img_tensor = img_arr.transpose(2, 0, 1).astype(np.float32) / 255.0

        future_ret = closes[i + forward_window] - closes[i]
        if future_ret >= threshold:
            label = 1  # BUY CALL
        elif future_ret <= -threshold:
            label = 2  # BUY PUT
        else:
            label = 0  # HOLD

        images.append(img_tensor)
        labels.append(label)

    X = np.stack(images)
    y = np.array(labels, dtype=np.int64)

    logger.info(f"Vision Dataset Built for {sym_clean}: {len(X):,} snapshots. Class distribution: HOLD={np.sum(y==0)}, CALL={np.sum(y==1)}, PUT={np.sum(y==2)}")
    return X, y


def train_vision_model(X: np.ndarray, y: np.ndarray, output_path: Path, epochs: int = 5, batch_size: int = 128):
    try:
        torch.set_num_threads(max(1, (os.cpu_count() or 4) - 1))
    except Exception:
        pass
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training PyTorch 2D CNN Vision Brain on {device} ({len(X):,} samples across {torch.get_num_threads()} CPU threads)...")

    indices = np.random.permutation(len(X))
    split = int(0.8 * len(X))
    train_idx, val_idx = indices[:split], indices[split:]

    X_train, y_train = torch.tensor(X[train_idx]), torch.tensor(y[train_idx])
    X_val, y_val = torch.tensor(X[val_idx]), torch.tensor(y[val_idx])

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)

    model = CandlestickCNN(num_classes=3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_X.size(0)
            preds = outputs.argmax(dim=-1)
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)

        scheduler.step()
        train_acc = correct / max(1, total)

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * batch_X.size(0)
                preds = outputs.argmax(dim=-1)
                val_correct += (preds == batch_y).sum().item()
                val_total += batch_y.size(0)

        val_acc = val_correct / max(1, val_total)
        logger.info(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {total_loss/total:.4f}, Acc: {train_acc*100:.1f}% | Val Loss: {val_loss/val_total:.4f}, Acc: {val_acc*100:.1f}%")

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), output_path)

    logger.info(f"Saved Best Vision Model to {output_path} (Best Val Accuracy: {best_val_acc*100:.2f}%)")


def main():
    data_dir = ROOT_DIR / "algo_vpin_v2" / "data"
    vision_dir = ROOT_DIR / "vision_chart_brain"

    # 1. Train on NIFTY
    nifty_csv = data_dir / "nifty_12m_1min.csv.gz"
    if nifty_csv.exists():
        X_n, y_n = build_vision_dataset(nifty_csv, symbol="NIFTY", step_bars=8)
        train_vision_model(X_n, y_n, output_path=vision_dir / "vision_cnn_nifty.pt", epochs=8)
        torch.save(torch.load(vision_dir / "vision_cnn_nifty.pt"), vision_dir / "vision_cnn_model.pt")

    # 2. Train on SENSEX
    sensex_csv = data_dir / "sensex_12m_1min.csv.gz"
    if sensex_csv.exists():
        X_s, y_s = build_vision_dataset(sensex_csv, symbol="SENSEX", step_bars=8)
        train_vision_model(X_s, y_s, output_path=vision_dir / "vision_cnn_sensex.pt", epochs=8)

    # 3. Train on BANKNIFTY
    banknifty_csv = data_dir / "banknifty_12m_1min.csv.gz"
    if banknifty_csv.exists():
        X_b, y_b = build_vision_dataset(banknifty_csv, symbol="BANKNIFTY", step_bars=8)
        train_vision_model(X_b, y_b, output_path=vision_dir / "vision_cnn_banknifty.pt", epochs=8)


if __name__ == "__main__":
    main()
