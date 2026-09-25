# Quantitative Trading Bot - Command Reference

All commands run from the project root. Credentials are read from `.env` (copy `.env.example`).

---

## 1. Live / Simulation Engine (`algo_vpin_v2`)
* **Strategy Stack**:
  - **Macro Engine**: Previous Day High/Low (PDH/PDL), 1-Week Range Position (0–100%), 15-min Trend
  - **Layer 1**: BVC & VPIN Toxicity (Volume Clock)
  - **Layer 2**: Rolling GARCH(1,1) Econometric Forecaster
  - **Layer 3**: Ensemble Brain (SVM RBF + XGBoost; Track 2 adds Deep ANN) + Skills veto chain
* **Risk Engine**: ₹50,000 Capital, Volatility Stops, 15:15 IST last entry, 15:24 IST Square-Off

### Live Market Session (09:15 - 15:30 IST):
```powershell
python algo_vpin_v2/main.py --mode live --symbol NIFTY
python algo_vpin_v2/main.py --mode live --symbol SENSEX
```
Optional flags: `--strategy-mode auto|scalper|swing`, `--no-compare` (disable Track 2 Tri-Brain).

### Offline Simulation (300 Bars):
```powershell
python algo_vpin_v2/main.py --mode simulation --symbol NIFTY --bars 300
```

---

## 2. Training & Benchmarks
```powershell
python algo_vpin_v2/train_all_brains.py                 # SVM / XGB / ANN from 12-month history
python algo_vpin_v2/train_multi_asset_brains.py         # per-asset models (NIFTY / SENSEX / BANKNIFTY)
python train_ann_models.py                              # root-level ANN used by the live ensemble
python vision_chart_brain/train_multi_asset_vision.py   # per-asset Vision CNN (.pt)
python run_backtest_stats.py                            # 5-day signal/backtest stats
python run_today_backtest.py                            # Track 1 vs Track 2 replay (needs Dhan auth)
python test_ann_brain.py
python multi_day_ann_benchmark.py
```

---

## 3. Model Inspection
```powershell
python -c "import joblib; d=joblib.load('algo_vpin_v2/models/xgb_model.joblib'); print('XGBoost Samples:', len(d['feature_history']), '| Trees:', d['model'].n_estimators)"
python -c "import pandas as pd; df=pd.read_csv('algo_vpin_v2/data/nifty_v2_1min_20260923.csv'); print(f'Total bars: {len(df)}'); print(df.tail(5))"
```

---

## 4. Unit Test Suite
```powershell
python -m pytest tests -v
```

---

## 5. Key Parameters

| Parameter | Value |
| :--- | :--- |
| **Microstructure** | VPIN (50 Volume Buckets) |
| **Econometrics** | Rolling GARCH(1,1) |
| **Machine Learning** | SVM + XGBoost (Track 1), + Deep ANN (Track 2) |
| **Macro Awareness** | PDH, PDL, Weekly Range, 15m Trend |
| **Execution Mode** | Options Buying (CE/PE), target delta 0.72 |
| **Nifty / Sensex Lot** | 65 qty / 20 qty (1 Lot) |
| **Capital** | ₹50,000 (₹2,500 daily-loss kill switch, off by default) |
| **Intraday Cutoff** | 15:15 last entry, 15:24 square-off |
