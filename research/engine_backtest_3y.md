# v2 engine on 3 years of 1-minute data: before vs after the 28 Sep 2026 fixes

Harness: `backtest_12m.py --data 3y` (real engines replayed bar by bar). Data: `algo_vpin_v2/data/*_3y_1min.csv.gz`,
26 Sep 2023 to 25 Sep 2026, 735 traded sessions per index. Pricing: 0.72-delta weekly at 0.7% of spot, ~9%/session
decay, 0.1% slippage per side, Dhan costs with STT 0.15%. No bid-ask spread (real costs are higher).

`legacy` = engine as it was before 28 Sep: ML vetoes on, up to 12 trades/day, kill switch off, no cooldown,
NIFTY-scaled thresholds on SENSEX. `new` = engine as committed on 28 Sep (ML trained online but vetoes off,
4 trades/day, kill switch on, cooldown armed, scaled thresholds, real candles). `new_fast` = same without ML.

| Run | Trades (per day) | Gross Rs | Costs Rs | Net Rs | Win % | PF | Max DD Rs | t (gross) | Kill-switch days | 2023Q4 / 2024 / 2025 / 2026 |
|---|---|---|---|---|---|---|---|---|---|---|
| NIFTY legacy | 5,505 (7.5) | +253,811 | 401,808 | **-147,997** | 52.9 | 0.92 | -153,300 | 4.3 | 55 | -18.8k / -12.1k / -39.5k / -77.6k |
| NIFTY new | 2,642 (3.6) | +126,334 | 192,481 | **-66,147** | 53.6 | 0.93 | -75,506 | 2.9 | 21 | -14.1k / -13.4k / -14.9k / -23.7k |
| SENSEX legacy | 5,873 (8.0) | +99,361 | 428,570 | **-329,209** | 36.2 | 0.79 | -335,492 | 1.9 | 46 | -34.8k / -92.7k / -72.5k / -129.2k |
| SENSEX new | 2,769 (3.8) | +163,444 | 202,208 | **-38,765** | 44.3 | 0.95 | -82,052 | 4.0 | 19 | -5.1k / -31.3k / -20.8k / +18.5k |

Per trade: NIFTY new gross +47.8, cost 72.9, net -25.0; SENSEX new gross +59.0, cost 73.0, net -14.0.
Legacy SENSEX gross was +16.9 per trade: the unscaled 20-point counter-trend exit alone cost Rs 1.04 million
over 2,524 exits; scaled, it fires 237 times for Rs -187k.

## Reading

1. The fixes cut the loss by 55% on NIFTY and 88% on SENSEX, halve the drawdown, and cut kill-switch days
   from 55/46 to 21/19. That is the effect of fewer trades, scaled thresholds and the cooldown, not of a new edge.
2. Gross P&L before costs is now positive and statistically clear on both indices (t 2.9 and 4.0). The trigger
   captures something. But it captures about Rs 48-59 per trade and each trade costs Rs 73 before spread.
3. The 4-trade cap binds on 75-85% of days: the engine takes the first four signals, not the best four.
4. Exit mix (NIFTY new): 1,446 stops -616k, 710 runner-trail exits +778k, 299 counter-trend exits -199k.
   Median hold 4 minutes. The structure is many small stops paying for occasional runners.

Verdict: the engine is now safe and behaves as designed, and it still has negative expectancy after costs
on every year of the sample except SENSEX 2026. It should stay in paper mode. The only lever left inside this
strategy is trade selection (which four signals), which the current signal stack cannot rank.

Reproduce: `.venv/bin/python backtest_12m.py --symbol NIFTY --data 3y --ml on` (add `--legacy` for the pre-fix engine).
