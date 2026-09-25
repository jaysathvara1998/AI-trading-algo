# Trend-day / range-day strategy: backtest results

Specification: `trend_range_strategy.py` (module docstring). Pricing: 0.72-delta weekly option at 0.7% of spot,
~9%/session time decay, 0.1% slippage each way, Dhan costs (STT modelled at 0.10%; actual since Apr 2026 is 0.15%).

## 1. In-sample (bundled data: Nov 2025 + Mar/Apr-Sep 2026, ~137 sessions per index) - 25 Sep 2026

| | NIFTY | SENSEX | Pooled |
|---|---|---|---|
| Trades | 62 | 64 | 126 |
| Net Rs (options, with theta) | +12,753 | +41,394 | +54,148 |
| Net Rs (no theta) | +31,681 | +62,678 | |
| Win rate / profit factor | 46.8% / 1.20 | 51.6% / 1.65 | |
| t-statistic | 0.57 | 1.63 | 1.60 |
| Max drawdown Rs | -20,024 | -13,769 | |

Positive on both indices and both sub-periods, but t < 2 and the trend leg was chosen after seeing it work.

## 2. Out-of-sample (Dhan download `*_3y_1min.csv.gz`: 26 Sep 2023 - 25 Sep 2026, 740 sessions per index)

Rules unchanged from section 1 (expiry weekday made date-aware: NIFTY Thu until Aug 2025 then Tue;
SENSEX Fri until Dec 2024, Tue Jan-Aug 2025, Thu after).

| | NIFTY | SENSEX |
|---|---|---|
| Days trend / range / none / skipped | 129 / 192 / 241 / 178 | 136 / 175 / 250 / 179 |
| Trades | 344 | 325 |
| Net Rs (options, with theta) | **-26,992** | **-92,837** |
| per year 2023Q4 / 2024 / 2025 / 2026 | -8,849 / -33,613 / -1,688 / +17,157 | +2,006 / -64,508 / -38,660 / +8,325 |
| Trend leg net (n) | -45,036 (124) | -67,620 (134) |
| Range leg net (n) | +18,045 (220), t 0.50 | -25,217 (191) |
| Net Rs (no theta: futures) | +94,772, t 1.39 | +36,323, t 0.55 |
| Win rate / PF | 41.9% / 0.94 | 40.0% / 0.81 |
| Max drawdown Rs | -94,275 | -152,502 |

Parameter grid on 3 years: trend leg negative at every classification time (10:15-11:00) and every
threshold on both indices. Range leg positive on NIFTY at 10:15-10:45 (+7k to +18k, never significant)
and negative on SENSEX at every setting. Opening-range window 5/15/30: all negative on both.

## 3. Verdict

The in-sample result did not survive. Both legs lose over 2024-2025; the 2026 profit that motivated the
rule was one favourable regime. As a futures strategy (no theta) it is roughly breakeven-to-mildly-positive
and not significant. It should not be traded.

Three-year confirmation of the wider study (`pa_study.py` with `PA_DATA_TAG=3y`): 15-minute returns in the
first hour mildly mean-revert on both indices (autocorrelation -0.07, t -2.8); first-hour direction does
not predict the rest of the day (corr +0.02); opening-range breakouts return inside within 15 min 69-73%
of the time; previous-day high/low touches carry no directional information (180-200 events each);
fading gaps >= 0.5% earns -2.7 pts on NIFTY over 129 gaps (the earlier +55 pts was April 2026 alone);
5-minute hammers are followed by further decline on both indices (t -2.6 and -3.9).

## Reproduce

    .venv/bin/python research/fetch_dhan_history.py --years 3 --tag 3y      # needs a valid Dhan token
    .venv/bin/python research/trend_range_strategy.py --data 3y --grid
    PA_DATA_TAG=3y .venv/bin/python research/pa_study.py
