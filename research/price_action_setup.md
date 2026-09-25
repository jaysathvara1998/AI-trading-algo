# Price-Action Trading Setup: Trendlines, Structure and Trapped Traders

Prepared 25 September 2026 for NIFTY and SENSEX intraday index options. Specification only, nothing implemented.
Companion files: `pa_mechanics.py` (measurements quoted below, 740 sessions per index, Sep 2023 to Sep 2026),
`pa_study.py`, `trend_range_results.md`, and the research note `price_action_report.html`.

## 0. Read this first

Price action is a language for describing where other traders are positioned and where their stops sit.
Its one mechanism with academic support is order clustering: take-profit orders cluster at obvious levels,
stop-loss orders cluster just beyond them, so levels produce both bounces and, once crossed, fast moves
(Osler 2003, 2005; Kavajecz and Odders-White 2004). Everything else in the corpus, trendline touches,
"80% of breakouts fail", springs, order blocks, is practitioner belief with no published test.

Measured on three years of your own data at the 5-minute scale, the classic entries are close to zero
expectancy before costs:

| Event (5-min bars) | NIFTY, next 30 min | SENSEX, next 30 min | Fail rate |
|---|---|---|---|
| Trendline break, move in break direction | +0.3 pts (n 2,621, t 0.5) | −0.3 pts (n 2,620) | 28% back across within 3 bars |
| Failed trendline break, move in reversal direction | +1.0 pts (n 726) | +0.3 pts (n 739) | |
| Swing breakout (prior 60-min high/low), break direction | +0.7 pts (n 4,043) | +0.4 pts (n 4,083) | 51% fail within 3 bars |
| Trap fade (close back inside within 2 bars) | −1.1 pts (n 1,751) | −1.6 pts (n 1,782) | |
| Deep trap fade (penetration ≥ 0.5 ATR) | +1.7 pts (n 332, t 0.7) | +16.0 pts (n 355, t 2.2) | |
| Previous-day level breaks and holds, 11:00 to 14:30, continuation | +11.2 pts (n 55, t 2.1) | +12.4 pts (n 60, t 0.9) | |
| Previous-day level sweep, reversal | +0.7 pts (n 158) | −3.0 pts (n 173) | |

Median 30-minute move is 19 points on NIFTY and 63 on SENSEX; a round trip costs 1.5 and 5 points respectively.
So: a plain trendline break is a coin toss, half of all breakouts fail, fading every failure loses, and the
only two cells with a pulse are deep traps and mid-session holds of the previous day's level, each on one
index only. The setup below is built around those two mechanisms plus the with-trend retest, with the
plain breakout entry deliberately excluded. It is a hypothesis to test on the frozen rules, not a strategy.

## 1. Definitions (all codable)

**Timeframes.** 15-minute chart for direction and levels, 5-minute chart for setups and triggers. The 1-minute
chart is used only to place the order at the 5-minute close; nothing is decided on it.

**ATR.** Median 5-minute bar range of the current session so far, floored at 8 points NIFTY / 25 SENSEX.

**Swing high / low.** 5-minute bar whose high (low) exceeds the two bars on each side (2/2 fractal). A swing
is confirmed two bars after it prints; nothing uses an unconfirmed swing.

**Trendline.** Line through the two most recent confirmed swing lows (rising, support) or swing highs
(falling, resistance), at least 3 bars apart. Valid only after a third touch within 0.25 ATR that does not
close through it. Dead after the first 5-minute close 0.25 ATR beyond it; never redrawn. Stale if the most
recent touch is more than 20 bars old.

**Levels**, in priority order: previous-day high and low; opening range high and low (09:15 to 09:30);
the 15-minute swing highs and lows of the session; round numbers (NIFTY multiples of 100, SENSEX of 500).

**Day classification at bar 18 (10:45).** Trend day if the 10:45 close is beyond the opening range by at least
half its width and the 15-minute chart shows two higher highs and higher lows (or the mirror). Range day
if the close is inside the opening range and the range is at least 60 NIFTY / 210 SENSEX points. Otherwise
undefined: no trades until a level break qualifies under setup C.

**Direction.** On a trend day, only trades in the trend direction. On a range day, only trades toward the
opposite edge. Undefined days allow setup C only.

**Session windows.** Setups may trigger 09:45 to 11:00 and 13:30 to 14:30. No entries 09:15 to 09:45
(first 15 minutes carry a median range of 74 NIFTY points, four times midday), none 12:00 to 13:30, none
after 14:30. Every position is flat by 15:10.

**Skip days.** The index's weekly expiry day (NIFTY Tuesday, SENSEX Thursday), any day with an opening gap
above 1%, and any session where the opening range is below 40 NIFTY / 140 SENSEX points.

## 2. The three setups

### Setup A: trendline break and retest, with trend

The with-trend version of the classic entry. The plain break is not traded, because the data show it is a
coin toss; the retest is traded because it puts the stop where the failed breakout traders' stops sit.

1. Day classified as trend. A valid falling trendline (in an up-trend day) has been broken by a 5-minute
   close at least 0.25 ATR beyond it.
2. Within the next 6 bars price returns to within 0.25 ATR of the broken line without closing back through it.
3. Trigger: the first 5-minute bar that closes back in the trend direction with its close in the top
   (bottom) third of its range and beyond the previous bar's high (low).
4. Entry at that close. Stop 0.1 ATR beyond the retest extreme. Target: the prior swing high (low) or
   entry plus the height of the last completed swing, whichever is nearer, minimum 2R.
5. If the target is not reached within 12 bars, exit at market. One setup A trade per day.

### Setup B: deep trap reversal (the "weak hands" trade)

Breakout traders who bought a level that then fails are the trapped side; their exits fuel the reversal.
The measurements say shallow traps do not reverse (they resume), so the penetration must be deep enough
to have pulled in real participation.

1. A 5-minute bar closes at least 0.5 ATR beyond a level (previous-day high or low, opening-range edge,
   or a 15-minute swing). Shallow penetrations are ignored.
2. Within the next 3 bars a 5-minute bar closes back on the original side of the level. That bar is the
   trap bar. On a trend day the trap must be against the trend direction, so the trade is with the trend;
   on a range day the trap must be at the range edge; on an undefined day this setup is not taken.
3. Trigger: entry at the trap bar's close.
4. Stop 0.1 ATR beyond the extreme of the penetration. Target: the opposite edge of the structure
   (opening range, previous-day range, or last swing), minimum 1.5R.
5. Time stop 6 bars: if the trade is not at least 0.5R in profit after 30 minutes, exit. Maximum two
   setup B trades per day, never twice at the same level.

### Setup C: level break that holds, mid-session

The one continuation pattern with a positive reading on NIFTY. A previous-day high or low broken by a
close between 11:00 and 14:30 and still held 15 minutes later.

1. Between 11:00 and 14:30 a 5-minute bar closes beyond the previous-day high (low) by at least 0.25 ATR.
2. The next three 5-minute closes all remain beyond the level (the 15-minute close confirms).
3. Trigger: entry at the third close. Skip if the 11:00 to 14:30 window has fewer than 60 minutes left.
4. Stop 0.1 ATR back inside the level. Target: entry plus the previous day's range times 0.5, minimum 2R.
   Exit at 15:10 regardless.
5. One setup C trade per day. Allowed on undefined days; on trend days only in the trend direction.

## 3. Trade management (all setups)

- Position size: one lot per ₹50,000 of capital is the maximum; at the measured drawdowns, half a lot
  per ₹50,000 is the honest size, which on NIFTY means trading only when the setup's stop is under 25 points.
- Breakeven: after +1R, stop to entry plus 1.5 premium points. Never before +1R.
- Trailing: after +2R, trail 0.5R behind the best 5-minute close.
- Daily stop: two consecutive losses or ₹2,500 net, whichever first, ends the day.
- Instrument: setup B (fast, under an hour) can use the 0.72-delta weekly option. Setups A and C hold up
  to two hours; use index futures or an option with delta 0.85 or higher, otherwise time decay removes
  roughly half the modelled profit.
- Orders: limit at the trigger close plus one tick, cancelled if unfilled within 60 seconds. Market orders
  only for exits.

## 4. What each rule rests on

| Rule | Practitioner source | What the 3-year data says |
|---|---|---|
| 2 touches anchor, 3rd validates; 0.25 ATR tolerance; never redraw | Brooks; LuxAlgo defaults; TSG | Validated lines break no better than unvalidated ones (0.0 vs +0.3 pts) |
| Close beyond, not wick; 0.25 ATR penetration | SMC (edgeflo), LuxAlgo, ISFM | Breaks still fail 28% within 3 bars; no follow-through on average |
| Retest entry, not the break | TSG, Prometheus, Indian educators | Not measured directly; retest lets the stop sit at the trapped side |
| Deep trap (≥ 0.5 ATR) before fading | Wyckoff spring #2 (1 to 3% daily, translated) | SENSEX +16 pts (t 2.2); NIFTY +1.7 (t 0.7). Shallow traps resume: −1.7 and −5.9 pts |
| Level break that holds 15 min, midday | BOS/CHoCH; Osler's post-break acceleration | NIFTY +11 pts (t 2.1, n 55); SENSEX +12 (t 0.9). First-hour holds: nothing |
| Bar-18 classification | Brooks (80 to 90% of HOD/LOD in by bar 18) | Trend-day continuation from 10:30 is 52 to 54%, not tradeable alone |
| Skip 09:15 to 09:45; windows 09:45 to 11:00 and 13:30 to 14:30 | MarketNetra, Intraday Lab | Volatility profile confirms the open is four times midday noise |
| Skip expiry days and gaps above 1% | Sahi, Intraday Lab | Large gaps do not fill (14% and 0%); expiry not tested here |
| 1:2 R:R, stop beyond signal bar, breakeven at +1R | Brooks, ChartTalks | Not tested; standard |
| "80% of breakouts fail" | Brooks | Measured: 51% fail within 3 bars; 28% for trendlines |

## 5. Validation before any live use

1. Freeze this document. Implement the three setups exactly as written, then run on the 3-year files.
2. Pass criteria per index: at least 100 trades, net of Dhan costs at 0.15% STT plus a 0.2% spread allowance,
   with t of at least 2, positive in each calendar year, and the same sign on both indices.
3. Walk-forward: fit nothing. If a threshold must change, the change is a new hypothesis and the clock restarts.
4. If it passes, paper trade for 30 sessions with the live engine before any real order.

## 6. Sources

Osler (2003) Currency orders and exchange rate dynamics, J. Finance; Osler (2005) Stop-loss orders and price
cascades, JIMF; Kavajecz and Odders-White (2004) Technical analysis and liquidity provision, RFS; Brooks,
Trading Price Action (glossary and course notes); Raschke and Connors, Street Smarts (Turtle Soup);
Wyckoff Analytics; LuxAlgo trendline and liquidity-sweep definitions; Intraday Lab NIFTY ORB backtest
(2017 to 2026, PF 1.23 before costs); MarketNetra session-timing guide. Full URLs in the research agent
brief archived with this note.
