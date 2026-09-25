# Future Roadmap: Self-Learning AI Strike Selector (ML & Deep Learning)

## Objective
Build a dynamic, self-learning Machine Learning / Deep Learning Strike Selection engine that autonomously selects the optimal strike tier:
$$\text{Strike Tier} \in \{\text{DEEP\_OTM}, \text{OTM}, \text{ATM}, \text{ITM-1}, \text{ITM-2}\}$$
based on real-time market microstructure, VPIN order flow toxicity, GARCH volatility forecasts, Option Greeks, and time-to-expiry (DTE).

---

## 1. Feature Engineering Vector
* **Order Flow Dynamics**: VPIN Toxicity Score, Volume Clock bucket completion velocity.
* **Volatility Dynamics**: GARCH(1,1) forecasted conditional volatility $\sigma_{t+1}$, return drift $\mu_{t+1}$, volatility term structure.
* **Options & Time Decay**: Days to Expiry (DTE), Intraday minute fraction (09:15 to 15:30), IV percentile & skew.
* **Macro Guardrails**: Distance to Previous Day High/Low (PDH/PDL), Week High/Low.

---

## 2. Model Architecture Candidates
1. **Option A: Multi-Class XGBoost / LightGBM Payoff Predictor**:
   * Evaluates Expected Value $\mathbb{E}[\text{Return}]$ across discrete strike buckets.
   * Low latency (<0.5ms inference per bar).
2. **Option B: PyTorch Deep Neural Network (MLP + Transformer)**:
   * Multi-layer neural network with Softmax probability outputs for tail-event explosion vs. steady drift.
3. **Option C: Reinforcement Learning (Q-Learning Policy Network)**:
   * Agent receives reward based on realized Sharpe ratio & point capture across various market regimes.

---

## 3. Training & Validation Pipeline
* Ingest 1-minute historical candles from Dhan `/charts/rollingoption` across OTM, ATM, and ITM strikes.
* Compute 5-min, 15-min, and 30-min forward returns.
* Train model to maximize Profit Factor while strictly avoiding theta decay traps in low-volatility chop.
