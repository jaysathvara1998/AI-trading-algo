# Trend-day method: rules and 3-year evidence (30 Sep 2026)

The one-page method written as rules (`trend_day/rules.py`) and scored on 1-minute NIFTY and SENSEX,
Sep 2023 to Sep 2026, expiry days skipped. Costs: same Zerodha option model as before (0.72-delta weekly
option, spread, theta, charges) and a futures alternative (STT 0.02%, slippage 0.5 pt per side).

## The rules

1. Levels: previous day high, low, close; first 30-minute range (OR).
2. Day type: two consecutive 15-minute closes beyond the OR (variant: and beyond yesterday's range) by 11:30
   call the day UP or DOWN. Otherwise BALANCED: no trade.
3. Trade: after the call, the first 15-minute candle closing against the direction starts a pullback. Entry
   on the first 5-minute candle that closes back in the direction beyond the previous 5-minute extreme, filled
   at the next 1-minute open. No entry after 13:30.
4. Stop: pullback extreme plus 0.10 x 15-minute ATR, hard. Skip if wider than 0.6% of price.
5. Exit: 15-minute close beyond the last 15-minute swing (trail), 15-minute close back inside the OR
   (day invalidated), or 15:10. One trade a day.

## Finding 1: the day-type call has no predictive value

| | NIFTY | SENSEX |
|---|---|---|
| Days called UP or DOWN by 11:30 | 59% | 60% |
| Drift from the call to 15:10, in the called direction | +1.2 pts | +6.6 pts |
| Share of called days that continue | 51% | 53% |

Stricter versions (three closes, 60-minute range, beyond yesterday's range, earlier deadline) reduce the
number of days called to 22 to 48% but leave the continuation share at 49 to 55% with t-statistics below 1.2.
Range expansion in the opening 30 minutes and gap size were also checked: neither bucket is monotonic or
consistent across years. "Acceptance outside the opening balance" on the index is not a trend-day signal.

## Finding 2: the pullback trade has a trend-following profile with a small, unstable edge

| | NIFTY base | SENSEX base |
|---|---|---|
| Trades | 236 (100/yr) | 243 |
| Win rate | 30% | 34% |
| Average win / loss (pts) | +95 / -25 | +279 / -74 |
| Expectancy (pts) | +5.4 | +18.7 |
| 95% CI of expectancy (pts) | -1.9 to +13.7 | -4.7 to +45.0 |
| Option net (Rs) | -33,937 | -24,320 |
| Futures net (Rs) | -44,896 | -21,752 |
| Futures net by year | +5k, +53k, -60k, -43k | +14k, +12k, -48k, 0k |

Positive in points, but the confidence interval includes zero, and 2025 loses on both indices in every
variant. Exits are 77% hard stops: the method makes its money on a minority of trades that run 100+ points.

## Variant sweep

| index | variant | days called | drift after call (pts) | t | P(continue) | trades | win | pts/trade | R | option net Rs | futures net Rs | futures net by year |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| NIFTY | base | 59 | 1.2 | 0.18 | 0.51 | 237 | 0.3 | 5.3 | 0.18 | -33937 | -44896 | 23:+5k 24:+53k 25:-60k 26:-43k |
| NIFTY | beyond_pdr | 32 | 6.4 | 0.72 | 0.55 | 139 | 0.31 | 6.6 | 0.29 | -12056 | -14673 | 23:+15k 24:+42k 25:-33k 26:-39k |
| NIFTY | confirm3 | 48 | 1.6 | 0.23 | 0.53 | 212 | 0.33 | 7.8 | 0.28 | -7456 | -5289 | 23:+9k 24:+71k 25:-46k 26:-40k |
| NIFTY | or60 | 39 | -3.7 | -0.49 | 0.51 | 146 | 0.31 | 7.7 | 0.2 | -2946 | -5515 | 23:-2k 24:+52k 25:-32k 26:-22k |
| NIFTY | or60_pdr | 22 | -1.9 | -0.19 | 0.55 | 91 | 0.32 | 7.7 | 0.28 | -2624 | -3327 | 23:+1k 24:+32k 25:-21k 26:-15k |
| NIFTY | deadline_10:30 | 31 | 11.2 | 1.18 | 0.53 | 129 | 0.33 | 11.8 | 0.44 | 17565 | 30486 | 23:+11k 24:+54k 25:-28k 26:-6k |
| NIFTY | deadline_12:30 | 69 | 2.2 | 0.37 | 0.51 | 282 | 0.29 | 5.5 | 0.19 | -33088 | -49154 | 23:+31k 24:+48k 25:-87k 26:-41k |
| NIFTY | trail_none | 59 | 1.2 | 0.18 | 0.51 | 237 | 0.27 | 3.3 | 0.12 | -64429 | -75755 | 23:+3k 24:+48k 25:-76k 26:-51k |
| NIFTY | buffer_0.25 | 59 | 1.2 | 0.18 | 0.51 | 236 | 0.31 | 2.3 | 0.1 | -76205 | -91342 | 23:+5k 24:+33k 25:-77k 26:-52k |
| NIFTY | max_trades_2 | 59 | 1.2 | 0.18 | 0.51 | 240 | 0.3 | 5.2 | 0.18 | -35041 | -46587 | 23:+5k 24:+51k 25:-62k 26:-41k |
| NIFTY | pdr_or60_trail_none | 22 | -1.9 | -0.19 | 0.55 | 91 | 0.33 | 7.2 | 0.26 | -7779 | -6349 | 23:+4k 24:+36k 25:-31k 26:-15k |
| SENSEX | base | 60 | 6.6 | 0.32 | 0.53 | 244 | 0.34 | 20.1 | 0.24 | -24320 | -21752 | 23:+14k 24:+12k 25:-48k 26:+0k |
| SENSEX | beyond_pdr | 30 | -4.2 | -0.15 | 0.51 | 141 | 0.36 | 32.0 | 0.36 | 7214 | 20970 | 23:+8k 24:+20k 25:-9k 26:+1k |
| SENSEX | confirm3 | 50 | 7.9 | 0.35 | 0.54 | 220 | 0.37 | 26.9 | 0.34 | -956 | 10259 | 23:+7k 24:+32k 25:-37k 26:+7k |
| SENSEX | or60 | 40 | 8.0 | 0.35 | 0.52 | 150 | 0.4 | 42.3 | 0.49 | 31775 | 53381 | 23:+2k 24:+59k 25:-14k 26:+5k |
| SENSEX | or60_pdr | 22 | -8.2 | -0.26 | 0.49 | 89 | 0.42 | 36.0 | 0.45 | 9382 | 20158 | 23:-7k 24:+29k 25:-2k 26:+1k |
| SENSEX | deadline_10:30 | 32 | 21.9 | 0.79 | 0.53 | 131 | 0.34 | 21.8 | 0.22 | -11165 | -6862 | 23:+9k 24:-11k 25:-16k 26:+11k |
| SENSEX | deadline_12:30 | 71 | 4.7 | 0.25 | 0.52 | 289 | 0.34 | 23.1 | 0.28 | -13343 | -8082 | 23:+36k 24:+18k 25:-71k 26:+8k |
| SENSEX | trail_none | 60 | 6.6 | 0.32 | 0.53 | 244 | 0.3 | 14.6 | 0.21 | -55425 | -48752 | 23:+18k 24:-2k 25:-63k 26:-1k |
| SENSEX | buffer_0.25 | 60 | 6.6 | 0.32 | 0.53 | 244 | 0.33 | 16.9 | 0.16 | -45425 | -37264 | 23:+11k 24:+5k 25:-52k 26:-1k |
| SENSEX | max_trades_2 | 60 | 6.6 | 0.32 | 0.53 | 247 | 0.34 | 19.8 | 0.24 | -25570 | -23566 | 23:+14k 24:+12k 25:-49k 26:+0k |
| SENSEX | pdr_or60_trail_none | 22 | -8.2 | -0.26 | 0.49 | 89 | 0.35 | 30.0 | 0.37 | -810 | 9467 | 23:-5k 24:+26k 25:-6k 26:-4k |
Expectancy in points, pooled and by year, for the candidates (t in brackets):

| | n | pts/trade | t | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|---|---|
| NIFTY, call by 10:30 | 129 | +11.8 | 1.9 | +25 | +24 | -3 | +4 |
| NIFTY, beyond yesterday's range | 139 | +6.6 | 1.3 | +34 | +22 | -1 | -15 |
| SENSEX, 60-minute range | 149 | +40.2 | 2.3 | -2 | +79 | +11 | +32 |
| SENSEX, beyond yesterday's range | 141 | +32.0 | 1.8 | +60 | +50 | +17 | +27 |

## Where this leaves us

- The only candidate that is positive in every year on both the point and futures measure is SENSEX with
  acceptance beyond yesterday's range: about 47 trades a year, 36% win rate, +32 points a trade, roughly
  Rs 21,000 net over three years in futures and Rs 7,000 in options. That is one lot's worth of edge that
  is real at perhaps 90% confidence and small enough that two bad months erase it.
- NIFTY has the same shape in 2023 and 2024 and none of it in 2025 and 2026.
- Everything else tried on this data in three months of work (the old engine, sweep and retest setups,
  the specification engine, and now this) shares one feature: the index's direction over the next hour is
  not predictable from its own price action often enough to pay weekly-option costs.

Recommendation: if one algo is to be taken forward, it is this one, on SENSEX, with the beyond-previous-
day-range rule, traded in paper for two months first. The expectation should be a small edge with long flat
stretches, not a steady income. Reproduce with:

    .venv/bin/python -m trend_day SENSEX data/sensex_3y_1min.csv.gz --set require_beyond_pdr=true
