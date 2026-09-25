# Price-action research scripts

Empirical study of intraday price behaviour on the bundled 1-minute NIFTY / SENSEX data
(`algo_vpin_v2/data/*_12m_1min.csv.gz`), used for the September 2026 price-action research report.

- `pa_study.py` - time-of-day profile, serial correlation, intraday momentum, opening-range breakout,
  previous-day high/low, gaps, candlestick patterns, 5-min structure, trend-day and fade tests.
- `pa_strategy.py` - candidate price-action strategies vs the current momentum trigger, priced with the
  same delta/theta option-premium model and Dhan cost schedule as `backtest_12m.py`.

Run from the project root with the project venv:

    .venv/bin/python research/pa_study.py
    .venv/bin/python research/pa_strategy.py
