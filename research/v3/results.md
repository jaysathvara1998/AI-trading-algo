# pa_engine v3: 3-year backtest results (29 Sep 2026)

Engine built from `Price_Action_Trading_Algorithm_Fresh_Start_Specification.docx` (branch `jay/price-action-v3`,
package `pa_engine/`). This note reports the first full historical run of the spec's default hypotheses on
1-minute NIFTY and SENSEX data, 26 Sep 2023 to 25 Sep 2026 (740 sessions each, expiry days skipped).

## Headline

**The specification's default setup chain does not have an edge on this data.** Across both indices, all
three chronological splits and 22 single-change variants, no configuration is profitable on the validation
(2025) and out-of-sample (2026) periods after realistic option costs. More importantly, the raw move in the
underlying per trade is about zero (NIFTY +0.14 points, SENSEX -4.7 points), so the loss is not a cost
problem that better execution could fix. The setup itself (sweep of a level, structure shift, retest,
1-minute confirmation) does not predict direction over the next 25 minutes.

This is the outcome the spec's own philosophy anticipates: every threshold is a hypothesis, and this run
falsifies the first set. The engine, logging and research tooling are what carry forward.

## What was run

- Timeframes 5m context / 3m setup / 1m execution. Setup = liquidity sweep of a level (PDH/PDL, opening range,
  session extremes, recent 3m swings, equal highs/lows; penetration 0.10 to 1.5 ATR, reclaim within 2 candles)
  followed by a BOS/CHOCH in the trade direction within 6 setup candles (close beyond the swing by 0.15 ATR),
  a retest of the broken level (0.25 ATR zone, within 8 candles), then a 1-minute confirmation candle
  (close beyond the prior candle with body >= 50% of range, or a micro break of the last 1m swing).
- Score threshold 8 of 14. Stop beyond the sweep extreme + 0.25 ATR (min 0.5 ATR, max 3 ATR). Target =
  nearest opposing level at least 1 ATR away, minimum 2R. 1 lot fixed, risk cap Rs 1,500 per trade.
- Exits: 50% at 1R, stop to breakeven, hybrid structure/ATR trail on the rest, time exit after 20 setup
  candles without 0.5R progress, exit on an opposing CHOCH, flatten 15:15. Entries 09:30 to 14:30, max 3
  trades per day, halt after 2 consecutive losses or Rs 3,000 daily loss, 5-candle cooldown.
- Fills at the next 1-minute open. Option economics: 0.72-delta weekly option, premium 0.7% of spot, theta
  0.025%/min, 0.2% spread per side, Zerodha brokerage Rs 20/order, STT 0.15% on sell premium, exchange,
  SEBI, stamp and GST charges.
- Splits: dev = Sep 2023 to Dec 2024, val = 2025, oos = 2026 (to 25 Sep).

## Baseline results

| symbol | split | sessions | trades | net Rs | gross Rs | charges Rs | win % | PF | avg R | max DD Rs | t-stat |
|---|---|---|---|---|---|---|---|---|---|---|---|
| NIFTY | all | 740 | 189 | -32983 | -14962 | 18020 | 47 | 0.66 | -0.01 | -34644 | -2.45 |
| NIFTY | dev_2023Q4_2024 | 310 | 71 | -1516 | 5397 | 6913 | 56 | 0.95 | 0.17 | -9907 | -0.17 |
| NIFTY | val_2025 | 248 | 71 | -19029 | -12350 | 6679 | 42 | 0.53 | -0.10 | -21051 | -2.44 |
| NIFTY | oos_2026 | 182 | 47 | -12438 | -8010 | 4428 | 40 | 0.50 | -0.14 | -11301 | -1.96 |
| SENSEX | all | 740 | 203 | -55638 | -37125 | 18513 | 39 | 0.53 | -0.08 | -55573 | -3.83 |
| SENSEX | dev_2023Q4_2024 | 310 | 81 | -14562 | -7102 | 7461 | 44 | 0.65 | 0.00 | -13984 | -1.58 |
| SENSEX | val_2025 | 248 | 75 | -33347 | -26660 | 6687 | 32 | 0.33 | -0.27 | -34161 | -4.34 |
| SENSEX | oos_2026 | 182 | 47 | -7728 | -3363 | 4365 | 40 | 0.70 | 0.07 | -11796 | -0.96 |
Reading the table: the dev period is roughly flat before charges on NIFTY (gross +Rs 5.4k, avg R +0.17)
and everything after it is negative. Win rate 40 to 47%, average R about zero, so the 2R-minimum targets
are not being reached often enough to pay for the 1R losses. Average hold 24 to 26 minutes.

## Where the trades go (baseline, all periods)

| | NIFTY | SENSEX |
|---|---|---|
| Trades | 189 | 203 |
| Full stop hit (-1R) | 69 (37%) | 76 (37%) |
| Structure invalidation exit (avg -0.58R) | 24 | 39 |
| Trailed out after TP1 (avg +0.9R) | 91 | 83 |
| Time exit | 3 | 5 |
| Reached 1R at some point | 15% | 10% |

- Direction does not matter: longs and shorts lose equally.
- Regime does not help: BULL_TREND, BEAR_TREND and TRANSITION are all negative on both indices; RANGE is
  flat on NIFTY and negative on SENSEX.
- Score is weakly informative on NIFTY only: score >= 11 is about flat (43 trades, +Rs 1.5k), score 8 to 10 is
  where the loss is. On SENSEX every score bucket loses.
- Level type: sweeps of previous-day lows (PDL) were the one positive pocket on NIFTY (9 trades, +0.96R avg),
  too few to mean anything. Sweeps of 3m swing lows were the worst (62 trades, -Rs 24.8k).
- Confirmation type: rejection + micro-BOS together is the best of the three on both indices (+0.14R and
  +0.21R avg) but still not profitable after charges on SENSEX.
- Planned R:R: neither the 2 to 2.5R plans nor the 5R+ plans win; the target choice is not the issue.

## Research matrix (one change at a time)

Each row changes exactly one thing from the baseline. `avg R` is the mean realised R multiple in that split.
`threshold_6` equals the baseline because the sweep + shift + retest + momentum chain already scores at
least 8. `trail_atr` equals the baseline because the hybrid trail almost always falls back to the ATR trail.
`raw_points` and `futures_costs` have fewer trades because the Rs 1,500 risk cap is applied to full lot
notional in those modes.

### NIFTY

| experiment | trades | net Rs | dev net Rs | dev avg R | val net Rs | val avg R | oos net Rs | oos avg R |
|---|---|---|---|---|---|---|---|---|
| baseline | 189 | -32983 | -1516 | 0.17 | -19029 | -0.10 | -12438 | -0.14 |
| no_retest | 187 | -35395 | 4864 | 0.37 | -23641 | -0.02 | -16617 | -0.20 |
| no_shift_required | 921 | -115077 | -32570 | 0.17 | -42218 | 0.16 | -40289 | 0.12 |
| sweep_pen_0.05 | 198 | -36192 | -4343 | 0.13 | -20879 | -0.12 | -10971 | -0.10 |
| sweep_pen_0.20 | 167 | -29172 | -2375 | 0.18 | -15534 | -0.08 | -11262 | -0.18 |
| sweep_wick | 50 | -16426 | 24 | 0.22 | -10899 | -0.68 | -5551 | -0.09 |
| bos_band_0.05 | 210 | -33593 | -4286 | 0.17 | -20976 | -0.04 | -8331 | -0.07 |
| bos_band_0.30 | 161 | -22641 | -4947 | 0.12 | -11624 | -0.04 | -6070 | -0.09 |
| threshold_6 | 189 | -32983 | -1516 | 0.17 | -19029 | -0.10 | -12438 | -0.14 |
| threshold_10 | 111 | -11078 | 826 | 0.32 | -3818 | 0.15 | -8086 | -0.15 |
| stop_buffer_0.10 | 241 | -51948 | -11774 | 0.11 | -27036 | -0.14 | -13138 | -0.07 |
| stop_buffer_0.50 | 133 | -35252 | -7198 | 0.04 | -18327 | -0.22 | -9727 | -0.12 |
| min_rr_1.5 | 267 | -49157 | -10380 | 0.12 | -28966 | -0.18 | -9812 | 0.01 |
| min_rr_3.0 | 108 | -21364 | -4550 | 0.10 | -10738 | -0.07 | -6076 | -0.05 |
| no_partials | 189 | -32755 | 238 | 0.17 | -19481 | -0.10 | -13512 | -0.14 |
| trail_structure | 188 | -30300 | -3174 | 0.16 | -13914 | 0.18 | -13211 | -0.14 |
| trail_atr | 189 | -32981 | -1516 | 0.17 | -19027 | -0.10 | -12438 | -0.14 |
| trail_none_full_tp2 | 187 | -29051 | -5500 | 0.14 | -14648 | 0.01 | -8903 | -0.05 |
| no_time_exit | 189 | -33124 | -1657 | 0.17 | -19029 | -0.10 | -12438 | -0.14 |
| no_invalidation_exit | 188 | -35205 | -2079 | 0.18 | -21031 | -0.12 | -12096 | -0.13 |
| futures_costs | 129 | -79257 | -15262 | 0.12 | -41140 | -0.17 | -22855 | -0.30 |
| raw_points | 129 | -7143 | 9651 | 0.12 | -10331 | -0.17 | -6463 | -0.30 |

### SENSEX

| experiment | trades | net Rs | dev net Rs | dev avg R | val net Rs | val avg R | oos net Rs | oos avg R |
|---|---|---|---|---|---|---|---|---|
| baseline | 203 | -55638 | -14562 | 0.00 | -33347 | -0.27 | -7728 | 0.07 |
| no_retest | 183 | -48261 | -16323 | -0.00 | -18702 | -0.17 | -13236 | -0.11 |
| no_shift_required | 984 | -180755 | -57677 | 0.07 | -62936 | 0.07 | -60142 | -0.04 |
| sweep_pen_0.05 | 215 | -59895 | -15452 | -0.02 | -35560 | -0.26 | -8884 | 0.07 |
| sweep_pen_0.20 | 185 | -48524 | -11670 | 0.01 | -35748 | -0.35 | -1105 | 0.21 |
| sweep_wick | 71 | -21286 | -5502 | 0.06 | -10908 | -0.13 | -4875 | -0.11 |
| bos_band_0.05 | 212 | -55515 | -19017 | -0.03 | -28206 | -0.01 | -8292 | 0.12 |
| bos_band_0.30 | 177 | -64212 | -23719 | -0.19 | -32278 | -0.36 | -8214 | 0.01 |
| threshold_6 | 203 | -55638 | -14562 | 0.00 | -33347 | -0.27 | -7728 | 0.07 |
| threshold_10 | 111 | -28973 | -7778 | -0.06 | -14451 | -0.20 | -6744 | -0.21 |
| stop_buffer_0.10 | 250 | -49450 | -4263 | 0.18 | -31300 | -0.18 | -13888 | -0.04 |
| stop_buffer_0.50 | 138 | -35322 | -13289 | -0.10 | -26468 | -0.35 | 4435 | 0.42 |
| min_rr_1.5 | 274 | -67525 | -20672 | -0.05 | -36002 | -0.20 | -10851 | 0.01 |
| min_rr_3.0 | 107 | -28111 | -9043 | -0.07 | -20035 | -0.36 | 966 | 0.32 |
| no_partials | 202 | -50317 | -11310 | 0.00 | -33326 | -0.26 | -5680 | 0.07 |
| trail_structure | 203 | -56692 | -14182 | 0.13 | -37705 | -0.37 | -4805 | 0.17 |
| trail_atr | 203 | -55672 | -14597 | 0.00 | -33347 | -0.27 | -7728 | 0.07 |
| trail_none_full_tp2 | 201 | -72836 | -18612 | -0.06 | -41063 | -0.35 | -13162 | -0.07 |
| no_time_exit | 203 | -59489 | -15234 | -0.01 | -33704 | -0.28 | -10551 | 0.02 |
| no_invalidation_exit | 203 | -65410 | -21348 | -0.04 | -36503 | -0.30 | -7558 | 0.09 |
| futures_costs | 124 | -78028 | -25957 | -0.07 | -40324 | -0.35 | -11747 | 0.12 |
| raw_points | 124 | -14912 | -526 | -0.07 | -16488 | -0.35 | 2101 | 0.12 |
What the matrix says:

1. **No variant is positive on val and oos together**, on either index. The best NIFTY variant is a stricter
   score threshold of 10 (dev +Rs 0.8k, val -Rs 3.8k, oos -Rs 8.1k, 111 trades). The best SENSEX variant,
   a wider 0.5 ATR stop buffer, is positive only in 2026 (31 trades) and loses in both earlier periods.
2. **Dropping the structure-shift requirement (sweep-only entries) multiplies trades by 5 and losses by 3.**
   The shift filter is doing real work; it just is not enough.
3. **Retest, partial exits, time exit and invalidation exit are each roughly neutral**: turning any of them off
   moves the result by a few thousand rupees in either direction, inside noise.
4. **Wick requirement, larger BOS band, 3R minimum** reduce trade count 2 to 3x without improving the
   average R. The losing trades are not concentrated in the "weak" setups these filters remove.
5. **Raw underlying points** (no costs, no option model) are still negative or flat on val and oos. There is
   no edge to unlock by trading futures or a different option strike.

## What this means for the plan

The spec's development sequence is complete through its MVP list (section 34): data replay, swings,
structure, levels, sweeps, structure breaks, retests, 1m confirmation, structural stops, targets, R:R,
sizing, position manager, event log, backtester with realistic costs and chronological splits, research
matrix, Kite paper runner. What is missing is a signal.

Honest options, in order of how much I would trust them:

1. **Use the event log to find false setups by hand.** `events.jsonl` records every sweep, shift, retest,
   rejection and exit with its numeric features. Reviewing 30 to 50 losing setups on a chart against the
   trader's own judgement is the fastest way to learn which rule is wrong (my suspicion: the 3m swing-low
   sweeps, which are the largest and worst bucket, are mostly ordinary pullbacks that happen to poke a
   recent swing, not liquidity events).
2. **Paper trade the engine as-is for a few weeks** to confirm the live signal matches the backtest and to
   collect the trader's disagreement with specific entries. This is the spec's section 27 step and it costs
   nothing.
3. **Test the one thing the earlier 3-year studies did find**: a 15-minute trend pullback with 5-minute
   confirmation had positive drift (+5 to +14 points per hour) but was too small for 1-lot weekly options.
   The engine can express that as context = 15m, setup = 5m if a variant is wanted.

What I would not do is tune the 22 knobs until a split turns green; with ~70 trades per split that is
curve fitting, and the matrix already shows no direction to tune toward.

## Files

- `research/v3/nifty_baseline/`, `research/v3/sensex_baseline/`: `trades.csv`, `metrics.json`, `config.json`
  (events.jsonl is generated locally with `--events` and is gitignored: 12 MB each).
- `research/v3/matrix_nifty/matrix.csv`, `research/v3/matrix_sensex/matrix.csv`.
- Reproduce: see `COMMANDS.md`.
