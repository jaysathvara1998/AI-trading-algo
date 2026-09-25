# Quantitative Trading Bot - Complete Command Reference

All commands can be executed directly from PowerShell/Terminal in `d:\Project\VPIN`, or by double-clicking the corresponding `.bat` launcher file.

---

## 1. Connection Health Check & Diagnostics
Verifies DhanHQ authentication, token validity, account funds balance, live quotes (LTP), and historical data feeds:

```powershell
python algo_vpin_dhan/check_connection.py
```
* **1-Click Batch Launcher**: `check_connection.bat`

---

## 2. Version 1: Baseline Engine (`algo_vpin_dhan`)
* **Strategy Stack**: Layer 1 VPIN (50 Buckets) + Layer 2 GARCH(1,1) Volatility + Layer 3 SVM RBF Brain
* **Risk Engine**: ₹50,000 Capital (1 Lot Max, ₹2,500 Daily Loss Kill-Switch, 15:24 IST Square-Off)

### Live Market Session (09:15 - 15:30 IST):
* **NIFTY Options**:
  ```powershell
  python algo_vpin_dhan/main.py --mode live --symbol NIFTY
  ```
  *1-Click Launcher*: `run_v1_nifty.bat`

* **SENSEX Options**:
  ```powershell
  python algo_vpin_dhan/main.py --mode live --symbol SENSEX
  ```
  *1-Click Launcher*: `run_v1_sensex.bat`

### Offline Simulation / Backtest (300 Bars):
```powershell
python algo_vpin_dhan/main.py --mode simulation --symbol NIFTY --bars 300
```
*1-Click Launcher*: `run_simulation_v1.bat`

---

## 3. Version 2: Multi-Timeframe Ensemble Engine (`algo_vpin_v2`)
* **Strategy Stack**:
  - **Macro Engine**: Previous Day High/Low (PDH/PDL), 1-Week Range Position (0–100%), 15-min Trend
  - **Layer 1**: BVC & VPIN Toxicity (Volume Clock)
  - **Layer 2**: Rolling GARCH(1,1) Econometric Forecaster
  - **Layer 3**: **Dual Ensemble Brain (SVM RBF + XGBoost Decision Trees Consensus)**
* **Risk Engine**: ₹50,000 Capital, Volatility Stops, 15:24 IST Square-Off

### Live Market Session (09:15 - 15:30 IST):
* **NIFTY Options**:
  ```powershell
  python algo_vpin_v2/main.py --mode live --symbol NIFTY
  ```
  *1-Click Launcher*: `run_v2_nifty.bat`

* **SENSEX Options**:
  ```powershell
  python algo_vpin_v2/main.py --mode live --symbol SENSEX
  ```
  *1-Click Launcher*: `run_v2_sensex.bat`

### Offline Simulation / Backtest (300 Bars):
```powershell
python algo_vpin_v2/main.py --mode simulation --symbol NIFTY --bars 300
```
*1-Click Launcher*: `run_simulation_v2.bat`

---

## 4. Model Inspection & Diagnostics

### Check Saved SVM Brain (.joblib):
```powershell
python -c "import joblib; d=joblib.load('algo_vpin_dhan/models/svm_model.joblib'); print('SVM Samples:', len(d['feature_history']), '| Vectors:', len(d['model'].support_vectors_))"
```

### Check Saved XGBoost Brain (.joblib):
```powershell
python -c "import joblib; d=joblib.load('algo_vpin_v2/models/xgb_model.joblib'); print('XGBoost Samples:', len(d['feature_history']), '| Trees:', d['model'].n_estimators)"
```

### View Recorded 1-Minute CSV Telemetry:
```powershell
python -c "import pandas as pd; df=pd.read_csv('algo_vpin_dhan/data/nifty_1min_20260901.csv'); print(f'Total bars: {len(df)}'); print(df.tail(5))"
```

---

## 5. Run Automated Unit Test Suite
```powershell
python -m pytest algo_vpin_dhan/tests -v
```

---

## 6. Architecture & Parameter Comparison

| Parameter | Version 1 (`algo_vpin_dhan`) | Version 2 (`algo_vpin_v2`) |
| :--- | :--- | :--- |
| **Microstructure** | VPIN (50 Volume Buckets) | VPIN (50 Volume Buckets) |
| **Econometrics** | Rolling GARCH(1,1) | Rolling GARCH(1,1) |
| **Machine Learning** | Single: **SVM (RBF Kernel)** | **Dual Ensemble: SVM + XGBoost Trees** |
| **Macro Awareness** | 1-Minute Focus | **PDH, PDL, Weekly Range, 15m Trend** |
| **Key Level Filter** | Standard SL/TP | **Rejects Buying at Weekly High / Selling at Weekly Low** |
| **Execution Mode** | ATM Options Buying (CE/PE) | ATM Options Buying (CE/PE) |
| **Nifty Lot Size** | 65 qty (1 Lot) | 65 qty (1 Lot) |
| **Sensex Lot Size**| 20 qty (1 Lot) | 20 qty (1 Lot) |
| **Capital Allocation** | ₹50,000 (₹2,500 Kill-Switch) | ₹50,000 (₹2,500 Kill-Switch) |
| **Intraday Cutoff**| 15:24 IST | 15:24 IST |
