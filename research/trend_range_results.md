# Trend-day / range-day strategy: backtest results (25 Sep 2026)

Specification: `trend_range_strategy.py` (see module docstring). Data: bundled 1-minute index bars,
NIFTY 137 sessions, SENSEX 139 sessions, Nov 2025 and Mar/Apr–Sep 2026. Pricing: 0.72-delta weekly
option at 0.7% of spot, ~9%/session time decay, 0.1% slippage each way, Dhan costs (STT at 0.10%;
actual rate since Apr 2026 is 0.15%, so real costs are ~₹5/trade higher).

## Base specification (classify 10:30, OR 09:15–09:30, trend k = 0.5, range stop 0.25, max 2 range trades)

| | NIFTY | SENSEX | Pooled |
|---|---|---|---|
| Days classified trend / range / none / skipped | 14 / 35 / 54 / 34 | 14 / 41 / 47 / 37 | |
| Trades | 62 | 64 | 126 |
| Net ₹ (options, with theta) | +12,753 | +41,394 | +54,148 |
| Net ₹ (no theta: futures / deep ITM) | +31,681 | +62,678 | |
| Win rate / profit factor | 46.8% / 1.20 | 51.6% / 1.65 | |
| Net per trade / t-statistic | +206 / 0.57 | +647 / 1.63 | +430 / 1.60 |
| Max drawdown ₹ | −20,024 | −13,769 | |
| Nov 2025 / 2026 net ₹ | +8,533 / +4,220 | +8,033 / +33,361 | |
| Trend-leg net ₹ (n) | +4,479 (14) | +26,467 (14) | t = 1.56 (28) |
| Range-leg net ₹ (n) | +8,274 (48) | +14,927 (50) | t = 0.85 (98) |

Exit mix (both indices): range trades stop out ~50% of the time and hit target ~30%; trend trades
almost always exit on time at 15:15 (1 stop in 28).

## The four evidence bars

1. **Statistical (t ≥ 2 after costs)** – FAIL. Pooled t = 1.60; per index 0.57 and 1.63. Only the
   no-theta SENSEX variant clears 2 (t = 2.42).
2. **Same sign on both indices** – PASS. Positive on both, for both legs.
3. **Same sign in both sub-periods** – PASS on both indices.
4. **Parameter stability** – PARTIAL. Range leg positive at every classification time (10:15–11:00)
   on both indices and at 8 of 9 stop/max-trade variants on each. Trend leg stable on SENSEX
   (positive at 10:15–10:45 for k 0.3–0.75), unstable on NIFTY (negative at 10:15, 10:45, 11:00).
   Opening-range window: OR15 and OR30 positive on both; OR5 negative on NIFTY.

## Reading

This is the first rule set in the study with the same sign on both indices, both sub-periods and
most parameter settings, and it trades about 1.3 times per traded day rather than 11. It is not
statistically established: 126 trades at t = 1.6 is consistent with an edge of the size shown and
also with none. Drawdowns of ₹14,000–20,000 on the modelled ₹50,000 capital mean position size
would have to be well under one lot per ₹50,000 to survive a normal losing streak. The trend leg
was also chosen after seeing it work in the earlier scan, so its in-sample result is optimistic;
the range leg is a fresh test.

Required before any live use: 3+ years of 1-minute data for both indices, the rules fixed as they
stand (no further tuning), walk-forward evaluation, and t ≥ 2 on at least 100 held-out trades
per index. If it survives that, trade it via futures or a ≥0.85-delta option, since time decay
removes roughly half the modelled profit.

## Reproduce

    .venv/bin/python research/trend_range_strategy.py --grid --out /tmp/tr
