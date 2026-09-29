# Directional mode (1-min close confirmation, structure stop, sigma target): 3-year results, 29 Sep 2026

Specification requested by the owner: no scalping; entry confirmed on a completed candle close beyond the previous
candle's high/low with a real body; stop under/over the low/high of the previous N candles (plus a small buffer);
target from standard deviation (1-sigma expected move over a 60-minute horizon from realized 1-min volatility);
skip trades with reward/risk < 1.5 or stop > 30% of premium; breakeven at +1R; 90-minute time stop; last entry 14:30.
Implementation: `algo_vpin_v2/directional.py`, `TradeStrategyMode.DIRECTIONAL`, `DirectionalConfig`; selected with
`--strategy-mode directional`. Signal direction still comes from the engine's momentum trigger / pattern override.

Harness: `backtest_12m.py --mode directional --tf <min> --lookback <n>`, 735 sessions per index, same premium and
cost model as before (delta 0.72, theta, STT 0.15%, no spread).

| Variant | Index | Trades/day | Gross Rs | Net Rs | Win % | PF | Max DD | Stop exits | Target exits | t (gross) |
|---|---|---|---|---|---|---|---|---|---|---|
| scalper baseline (28 Sep engine) | NIFTY | 3.6 | +127,879 | -64,679 | 53.7 | 0.93 | -76,831 | 55% | 0% | +2.9 |
| scalper baseline | SENSEX | 3.8 | +167,922 | -34,370 | 44.4 | 0.96 | -78,743 | 73% | 0% | +4.1 |
| directional, 1-min candles, 3-candle stop | NIFTY | 3.3 | +38,845 | -136,731 | 48.8 | 0.89 | -155,375 | 84% | 15% | +0.5 |
| directional, 1-min, 3-candle stop | SENSEX | 3.3 | -81,574 | -257,219 | 14.6 | 0.81 | -276,818 | 86% | 13% | -1.0 |
| directional, 5-min candles, 3-candle stop | NIFTY | 1.3 | -14,094 | -81,169 | 49.7 | 0.89 | -112,152 | 76% | 22% | -0.2 |
| directional, 5-min, 3-candle stop | SENSEX | 1.3 | -64,684 | -135,984 | 26.2 | 0.84 | -168,370 | 77% | 22% | -1.0 |
| directional, 5-min, 2-candle stop | NIFTY | 1.8 | -56,858 | -153,227 | 48.9 | 0.85 | -172,402 | 77% | 21% | -0.8 |
| directional, 5-min, 2-candle stop | SENSEX | 1.9 | -137,431 | -239,113 | 23.4 | 0.79 | -256,580 | 79% | 20% | -1.8 |
| directional, 15-min candles, 3-candle stop | NIFTY | 0.08 | -2,638 | -6,774 | 40.4 | 0.88 | -28,213 | 74% | 18% | -0.1 |
| directional, 15-min, 3-candle stop | SENSEX | 0.10 | -26,799 | -32,346 | 19.5 | 0.63 | -35,091 | 83% | 14% | -1.2 |

Negative in every year for every variant on both indices (one exception: NIFTY 5-min/3-candle in 2025, +2,079).

## Why

1. **The stop/target geometry cannot create expectancy.** Under a random walk the probability of reaching the target
   before the stop is stop / (stop + target). With a 1-sigma target 2 to 3 times the structure stop, 25 to 33% of trades
   win by construction, and the observed 15 to 27% win rates (SENSEX) are worse than that because stops are checked at the
   bar's adverse extreme and the option loses time value while waiting. Any edge has to come from the entry; the sigma
   multiple only sets the payoff shape.
2. **The entry has no drift.** Direction still comes from 15-minute momentum or a single-candle pattern, which the
   3-year study measured at near-zero forward drift. The candle-close confirmation removed only 9% of entries because the
   momentum trigger already fires on candles that closed beyond the previous high; on 5-minute candles it cut trades by
   60% but the survivors were no better (gross negative).
3. **Structure stops on index candles sit inside noise at every timeframe tested.** 1-min three-candle lows are 10 to 15
   NIFTY points from entry; 5-min are 25 to 40; the market's 60-minute one-sigma move is 35 to 60. Widening the candle
   timeframe raises the stop distance and the loss per stop at the same rate as it raises the target, so the ratio, and
   the outcome, barely change.
4. **Longer holds cost theta.** Median hold went from 3 minutes (scalper) to 10 to 67 minutes; on a weekly option that
   is 1 to 10% of premium per trade in decay before any move.

## Verdict

The three principles (confirmed close, structure stop, volatility target) are sound trade-management practice, and they
are now implemented and selectable. Applied to this engine's entries they lose more than the scalper mode, because the
management layer cannot fix an entry with no directional edge. This mode should not be traded, in paper or live, as a
strategy; it is available for testing a better entry signal, which is where the remaining work is.

Reproduce: `.venv/bin/python backtest_12m.py --symbol NIFTY --data 3y --mode directional --ml off --tf 5 --lookback 3`
