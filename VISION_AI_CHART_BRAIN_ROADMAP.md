# 🖼️ MASTER BLUEPRINT: 2D CNN & COMPUTER VISION CHART-READING BRAIN

This document serves as the complete technical architecture and ready-to-run implementation blueprint for training a **Computer Vision AI Brain** to visually analyze candlestick charts for institutional options trading.

---

## 🎯 1. Objective & Purpose

Traditional numerical models (XGBoost, SVM, ANN) only ingest scalar vectors `[price_delta, rolling_vol, vpin, garch_forecast]`. They cannot visually "see" 2D geometric patterns, multi-candle structures, or long rejection wicks.

The **Chart-Vision AI Brain** operates like a seasoned discretionary trader with computer vision:
- **Rejection Wicks & Liquidity Sweeps**: Recognizes long upper/lower shadow rejections at Previous Day High (PDH) and Previous Day Low (PDL) to veto bull/bear traps.
- **2D Structural Geometry**: Identifies Double Tops (M-pattern), Double Bottoms (W-pattern), and Compression Flags.
- **Multi-Brain Consensus**: Acts as an orthogonal 4th visual brain:
  $$\text{Signal} = \text{VPIN (Order Flow)} \ \cap \ \text{GARCH (Volatility)} \ \cap \ \text{Deep ANN (Tabular ML)} \ \cap \ \mathbf{\text{2D CNN (Chart Vision)}}$$

---

## 🏗️ 2. Architectural Components (`vision_chart_brain/`)

All files will be kept strictly isolated in `vision_chart_brain/` without touching `algo_vpin_v2/`:

```
vision_chart_brain/
├── __init__.py
├── chart_renderer.py           # Renders 128x128 high-contrast dark-mode candlestick images in memory (<20ms)
├── chart_dataset_builder.py    # Extracts 10,000+ historical chart snapshots & auto-labels them (BUY/SELL/HOLD)
├── cnn_vision_model.py         # PyTorch/TensorFlow 2D CNN Architecture (Conv2D -> MaxPool -> Dropout -> Dense)
├── train_vision_brain.py       # Training pipeline with AdamW, Cosine Annealing, Cross-Entropy Loss & Metrics
├── visual_predictor.py         # Real-time inference bridge (OHLCV -> Image -> Softmax Probabilities)
└── test_vision_benchmark.py    # Backtesting & out-of-sample visual validation suite
```

---

## 🧠 3. Deep Learning CNN Architecture

```
Input: [Batch, 3, 128, 128] (RGB Candlestick Image Array)
  │
  ├── Conv2D(32 filters, 3x3 kernel, ReLU) + BatchNorm2D + MaxPool2D(2x2)
  ├── Conv2D(64 filters, 3x3 kernel, ReLU) + BatchNorm2D + MaxPool2D(2x2)
  ├── Conv2D(128 filters, 3x3 kernel, ReLU) + BatchNorm2D + MaxPool2D(2x2)
  ├── GlobalAveragePooling2D
  ├── Dense(128, ReLU) + Dropout(0.4)
  └── Dense(3, Softmax) ──> [P(BUY CALL), P(HOLD / CHOP), P(BUY PUT)]
```

---

## 📋 4. Ready-to-Run Prompt (Copy & Paste to Execute Later)

Whenever you are ready to build and train the Vision Brain, send this exact prompt:

```markdown
Build and train the Standalone 2D CNN Chart-Vision AI Brain in the `vision_chart_brain/` directory as specified in `VISION_AI_CHART_BRAIN_ROADMAP.md`. 
1. Create `chart_renderer.py` to render 128x128 in-memory candlestick images.
2. Create `chart_dataset_builder.py` to generate 5,000+ labeled historical chart samples from Dhan historical data.
3. Build `cnn_vision_model.py` and train it via `train_vision_brain.py`.
4. Run `test_vision_benchmark.py` and display the accuracy, loss curve, and visual confusion matrix.
Do not modify `algo_vpin_v2/`.
```
