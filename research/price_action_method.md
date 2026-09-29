# Directional price action for NIFTY / SENSEX options: the method, the evidence, and what the data says

Written 29 Sep 2026 after three research passes (supply/demand zone doctrine, the Indian educators' method, the
published evidence) and direct measurement on 740 sessions per index of 1-minute data (Sep 2023 to Sep 2026).
Companion code: `research/zone_study.py` (measurements), `algo_vpin_v2/pullback_signal.py` and
`backtest_12m.py --signal pullback` (strategy test), `algo_vpin_v2/directional.py` (stop and target rules).

## 1. The vocabulary, as traders actually use it

**Zone selection.** A zone is a price area where the market previously paused and then left fast. In the
supply/demand school (Sam Seiden, Online Trading Academy, and the Hindi "demand-supply" channels) it is drawn from
a *base* of 1 to 6 small overlapping candles followed by an impulsive *departure* (at least 1.5 to 2 times the recent
range, closing beyond the base). Four shapes: rally-base-rally and drop-base-rally make demand; rally-base-drop and
drop-base-drop make supply. The far edge is the *distal* line (stop side), the near edge the *proximal* line (entry
side). In the Indian "decision point" school (Nifty Nirvana, from Lance Beggs) the zones are simpler and known before
the open: previous-day high, low and close; the opening-range high and low; big round numbers (NIFTY every 100,
SENSEX every 500); yesterday's 15-minute swing highs and lows; and *flip* levels (a broken supply now acting as demand).
Selection rules that every school shares: prefer **fresh** zones (never retested since formation; the first retest is
the only high-probability one), a **fast departure**, **little time spent** in the base, **confluence** of two or more
levels within about 0.1% of the index, and alignment with the **higher-timeframe** trend. Seiden's scoring rubric
(strength 0-2, reward-to-risk 0-2, higher-timeframe trend 0-2, freshness 0-2, time at level 0-1, arrival 0-1; trade at
7 or more, limit order at 9-10) is precise enough to code and has never been fitted to data by anyone.

**Make-or-break zone.** The one level whose decisive break or hold is expected to set the day's direction: the
previous-day high or low, the opening-range edge, the last higher low in an uptrend (or lower high in a downtrend),
or on expiry days the strike with the largest open interest. "Sustains" is rarely defined; the codable version is a
5-minute candle close beyond the level. Three trades are taken at it: **breakout-pullback** (close through, pull
back to the level, hold, go with the break), **breakout-failure** (poke through and close back inside, fade toward
the other side of the range), and **test** (stall in front of the level, reversal; ranked least reliable).

**Pullback.** A retracement, inside a trend defined by higher highs and higher lows on the 15-minute chart, back to
a zone, the broken level, a trendline or a short moving average. Tradeable when it is the *first* pullback after a
break, lands on a fresh or flip level, arrives within a few 5-minute candles, and prints a confirmation candle: a
5-minute close above the previous candle's high (long) or below its low (short), or an engulfing or pin bar at the
zone. Power of Stocks' variant enters on a break of the alert candle's extreme without waiting for a close.

**Liquidity.** Where resting stop orders sit: above equal highs and below equal lows, beyond the previous-day
high and low, beyond the opening range, around round numbers, at the session extremes late in the day. A *sweep* or
*grab* is a move through such a level that closes back inside (same candle for a grab, a few candles for a sweep);
*inducement* is the first sweep of the session, taught as a trap, with the second move being the real direction.
Rules: wait for the close back inside, stop 2 to 5 points beyond the sweep wick, target the range midpoint or the
opposite liquidity pool, at most three such trades a day.

**Trendline breakout.** Drawn on wicks, three touches to validate, breakout means a full-body 5-minute close beyond
the line, entry on the retest and confirmation close in the break direction, stop below the retest low, target by
measured move or the next zone, minimum 1:2. No Indian source adds anything numeric to this.

**Timeframes and limits.** 15-minute or 1-hour for structure and zones, 5-minute for entry, 1-minute only for
sweeps. Windows 09:20 to 11:00 and 14:15 to 15:00; nothing 12:00 to 13:30; flat by 15:15. Two or three trades a day;
stop after two losses. Stops under the confirmation candle or beyond the zone, capped at roughly 0.4% of the index;
targets at the next level, minimum 1:2, half booked at 1R and the rest trailed by 5-minute swings.

## 2. The method as one decision procedure

1. Before 09:15, plot the level set for the day. Mark any level with two or more confluences as make-or-break.
2. From 09:30, read the 15-minute chart: long bias if the last two swings are higher highs and higher lows and price
   is above VWAP; short bias for the mirror; otherwise a range day, fade the make-or-break levels only.
3. Pick the nearest fresh level in the direction of bias that is not mid-range (next level at least 1.5 stops away).
4. Wait for exactly one trigger on the 5-minute chart at that level: breakout-pullback, breakout-failure (sweep), or
   pullback-to-zone with a confirmation close.
5. Stop below the lower of the confirmation-candle low and the zone's distal line, plus a small buffer. Skip if the
   stop is wider than the cap.
6. Target the next level in the trade direction; skip if it is closer than 2R. Book half at 1R, breakeven, trail the
   rest by 5-minute swings.
7. Two to three trades a day, windows as above, one trade per level per day.

The contradictions a coder must resolve: limit order at the zone versus wait for the close (Seiden allows both,
Indian channels insist on the close); the first signal at a level is the best one (Nifty Nirvana) versus the first
one is inducement (SMC); opening range 15 versus 30 minutes; fixed-R versus next-level targets.

## 3. What the published evidence supports

| Element | Evidence | Verdict |
|---|---|---|
| Levels as resting liquidity (round numbers, previous-day extremes, consolidation edges) | Osler 2000, 2003, 2005 (FX, dealer order data); Kavajecz & Odders-White 2004 (NYSE book depth) | Real, small: bounce +4 to 6 points more often than arbitrary levels; the *cascade* after a break is the larger, longer effect. Useful for stops and targets more than entries. |
| Intraday time-series momentum (first half-hour predicts the last) | Gao et al. 2018; Jin et al. 2022 (16 markets); Baltussen et al. 2021 (dealer gamma) | Best-replicated intraday effect; small (+2.6 bp/trade), regime-dependent, compressed since 2025, and not visible in NIFTY spot over 740 sessions (correlation −0.02). |
| Volatility / gamma regime gating | Gao; Jin; Baule et al. 2025; one regime-gated MNQ strategy passing walk-forward | Consistent: edges exist on volatile, high-flow days. Matches our own trade-selection study (edge only on wide-range days). |
| Supply/demand zones, order blocks, fair-value gaps | StatOasis 2026: 648 daily backtests, 0 beat buy-and-hold; t-stats 1.2 / −0.1 / 0.9 | No demonstrated edge; the "unfilled institutional orders" story is unsupported. |
| Liquidity sweep reversal | Osler 2005 (real, under 30 min); Turtle Soup rated poorly on 42 futures; NIFTY base rate: 55% of PDH breaks hold, 45% sweep | Weak; a coin flip on NIFTY. |
| Pullback to a moving average or Fibonacci level | ESWA 2022 (Fib levels no better than random); StatOasis SMA-pullback no better than random | No evidence; Fibonacci negative. |
| Opening-range breakout | Zarattini & Aziz 2023; 2026 replication on five indices | Gross +0.05 to +0.13R per trade, which equals realistic costs; net about zero on indices. |
| Trendline break and retest | No disclosed-method test exists anywhere | No evidence. |
| Base rates, NIFTY | own calculation, 740 sessions | Trend days (open and close in opposite 15% bands) 13% of days; on those, the largest pullback after 10:15 retraces a median 40% of the move; first hour forms a median 53% of the day's range. |

The honest summary of the literature: the mechanism behind "zones" is order clustering at obvious levels, it is
real and small, and everything built on top of it (scoring, freshness, zone shapes, sweeps as reversals, trendlines)
is convention that has never been shown to add information.

## 4. What your data says, element by element (15-minute structure, 5-minute entries)

Forward spot move over the following hour, both indices, 740 sessions each (`zone_study.py`):

| Element | NIFTY | SENSEX | Reading |
|---|---|---|---|
| Fresh supply/demand zone, first retest | −3.8 pts, n 98; reaches +1 ATR before invalidation 36% | −21.7 pts, n 87; 33% | No reaction beyond chance. |
| Pullback to the last higher low / lower high in a 15-min trend + 5-min confirmation close | **+5.4 pts (t 2.0), n 342**, positive 3 of 4 years | **+14.5 pts (t 1.7), n 361**, positive 3 of 4 years | The one element with measurable drift. |
| Make-or-break: 5-min close through the last HL/LH, continuation | +1.7 pts, n 411 | +12.8 pts (t 1.4), n 434 | Mild on SENSEX only. |
| Make-or-break: touch and hold, bounce | −0.6 pts, n 1,210 | −0.6 pts, n 1,230 | Nothing. |
| 15-min trendline break, retest, 5-min confirmation | +0.4 pts, n 165 | −13.7 pts, n 161 | Nothing, or worse. |
| Previous-day level swept against the 15-min trend, reclaimed | −3.3 pts, n 99 | +10.2 pts, n 92 | Nothing consistent. |

(Reference: the median one-hour move is 27 NIFTY points and 87 SENSEX points; a round trip costs 1.5 and 5 points.)

The pullback-with-confirmation entry was then run as a full strategy with the owner's rules (structure stop on
5-minute candles, one-sigma target, breakeven at 1R, time stop, costs, time decay), `backtest_12m.py --signal pullback`:

| | NIFTY | SENSEX |
|---|---|---|
| Trades over 3 years (per day) | 182 (0.25) | 219 (0.30) |
| Net after costs and decay | −₹29,159 | −₹39,636 |
| Win rate / stop exits / target exits | 48% / 77% / 22% | 23% / 77% / 22% |
| Years positive | 0 of 4 | 0 of 4 |

## 5. Why a real but small drift still loses, in numbers

- The pullback entry's drift is about +5 NIFTY points per hour against an hourly standard deviation of about 27.
  With a target one sigma away and a stop a fraction of that, the target is reached first roughly stop ÷ (stop +
  target) of the time, about 30%, almost exactly the driftless value; the drift is too small to move that ratio.
- Holding for a fixed hour with no stop, the drift is worth about ₹250 gross per NIFTY lot (5 points × 0.72 delta ×
  65). Costs take ₹73 and the weekly option's time decay over the hour takes about ₹300. Net negative before spread.
- The same trade in NIFTY futures (no decay, but ₹380 of costs per round trip because STT is on notional) is also
  negative at one lot. The edge would need to be three to five times larger, or the position ten times larger, to
  clear the fixed cost floor.

This is the answer to "why is every strategy negative". It is not that the concepts are wrong; the pullback idea in
particular carries measurable information. It is that the information is worth a few index points per hour, and a
one-lot buyer of weekly options pays more than that in fees and decay every time they trade.

## 6. What would have to change for the method to pay

1. **Instrument and size.** The cost floor is fixed per trade and decay is per hour. Index futures or a deep
   in-the-money option (delta 0.85+) remove most decay; size of five or more lots spreads the ₹47 fixed brokerage.
   Neither is advisable at ₹50,000 of capital, which is a statement about capital, not about the method.
2. **Volatility gating.** Every positive reading in this project, here and in the trade-selection study, sits on
   high-range days. Any entry rule should be switched off when the day's range is narrow relative to its 20-day
   median, using only information known at entry time (the gap at 09:15, the 5-minute range from 09:20, the 15-minute
   range from 09:30).
3. **Fewer, later, longer trades.** The drift lives at the 30 to 60 minute horizon and after 09:30. One or two trades
   a day, entered after the opening range is known, held to a structural target or a time exit rather than a tight
   stop, is the only shape in which a 5-point-per-hour drift can be collected.
4. **Data.** Three years of spot data cannot validate a rule that fires 0.25 times a day: detecting a +0.1R edge
   needs roughly 800 trades. Five years of futures or option-implied prices, expanding-window walk-forward, yearly
   sign consistency, frequency-matched random baselines, and a kill criterion fixed in advance.

## 7. Verdict for the project

Keep: the directional management rules (candle-close confirmation, structure stop, volatility target), which are
implemented; the pullback-with-confirmation entry as the one entry with measured drift; and the volatility gate.
Retire: zone scoring, fresh-zone retests, trendline retests, sweep reversals and single-candle patterns as entries,
all measured at zero on this data. Do not trade any of it live with one lot of weekly options; the arithmetic in
section 5 does not depend on the entry being right.
