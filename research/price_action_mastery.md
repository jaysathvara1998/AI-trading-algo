# Price-action analysis for intraday index options: definitions, marking rules, evidence, and a testable trade model

Written 30 Sep 2026 from fresh web research (four parallel tracks, ~60 sources: peer-reviewed papers, exchange and
regulator documents, the primary ICT / TTrades / Al Brooks material, open-source SMC code, and published backtests).
Every rule below is tagged with the quality of the evidence behind it:

- **[A]** peer-reviewed or regulator/exchange primary data
- **[B]** working paper or transparent public backtest with stated method
- **[C]** practitioner definition or guide, no test
- **[D]** unverifiable claim

The purpose is a method for **1-minute execution, 3 to 4 quality trades a day, stop at the previous candle or the
liquidity level, target by standard-deviation projection, entry only on confirmation**. Section 8 is the complete
model on one page; section 9 is what must be measured on NIFTY/SENSEX before it is trusted.

---

## 1. The evidence map in one paragraph

Three things about price-action levels are established in the academic record: (i) prices where resting orders
cluster (round numbers, recent extremes, published support/resistance) bounce a few percentage points more often
than random levels (Osler 2000: 60.8% vs 56.2%) [A]; (ii) stop orders sit *just beyond* those levels, so a genuine
break accelerates for minutes to hours (Osler 2003, 2005; Donaldson & Kim 1993) [A]; (iii) technical support and
resistance coincide with depth already resting in the order book (Kavajecz & Odders-White 2004) [A]. The only
academically supported intraday *timing* effect is market-level continuation of the day's move into the last
half hour, strongest on high-volatility days (Gao et al. 2018; Baltussen et al. 2021), and in India that effect has
faded except for the last-half-hour version [A/B]. Everything else in the smart-money vocabulary (zone width,
sweep depth, order blocks, fair value gaps, OTE, BOS/CHoCH by body close, confluence counts, the standard-deviation
ladder, candle confirmation patterns) is convention: coherent, teachable, and **untested or tested negative**.
The one transparent mechanical test of order blocks / FVG / OTE (StatOasis, 648 daily backtests) found no
significant edge; the one formal test of 44 SMC concepts (SSRN 7483658, S&P daily) found none beat random entry.
The regulator's data say 88 to 93% of Indian retail F&O traders lose, options account for 92% of losses, and costs
alone flip about 6% of traders from profit to loss [A]. So the craft is real but thin: it locates where orders sit,
it does not by itself predict direction, and the edge it offers is smaller than option costs unless trades are few,
selective, and held long enough to be worth 30+ index points.

---

## 2. Levels and liquidity: what to mark before and during the session

### 2.1 Definitions

| Level | Definition | Evidence |
|---|---|---|
| Previous day high / low / close (PDH/PDL/PDC) | Extremes and close of yesterday's 09:15-15:30 session | Bounce lift at recent extremes [A]; PDH/PDL breaks behave as a *directional bias*, not a reversal level: after a PDH break the session closes green 62-81% (edgeful, YM/ES/NQ, small sample) [B] |
| Opening range (OR) | High/low of the first 15 or 30 minutes (09:15-09:30 / 09:45) | Overnight/opening-range sweeps mostly revert: 96-97% of Asia-range breaks on CME index futures return inside (tradingstats, 12 years) [B]; ORB as a *breakout* strategy has no edge after costs (pre-registered 225-cell study; NIFTY 15-min ORB 48.7% win, PF 1.23 before costs) [B] |
| Session high / low so far | Highest high and lowest low of today before the current candle | Stops rest beyond them [A, by Osler's mechanism] |
| Swing highs / lows | 3-candle rule: candle whose high exceeds both neighbours (confirmed when the third candle closes); on 5m for structure, 1m for execution | Definition [C]; bounce probability at previously visited extremes exceeds shuffled data and rises with the number of prior bounces, decaying with age (Chung & Bellotti 2021, 1-min data) [B] |
| Equal highs / lows | Two or more swings within a small band with a real pullback between them; NIFTY: two 5m swings within ~0.1 x 5m ATR (about 5-10 points), at least 3 bars apart | [C] |
| Round numbers / strikes | NIFTY 100s and 50s, SENSEX 500s/1000s | Stop and take-profit clustering at round numbers, take-profits *at* the number, stops just beyond it [A]; index-level barrier evidence weak outside the US [A/B] |
| OI walls | Highest call OI = resistance, highest put OI = support, watch day-over-day change | Untested; mechanically the Indian analogue of order-book depth [C]. Max pain: no academic support, one 19-expiry log worse than no-change [C/D] |

### 2.2 Marking procedure (before 09:15, then every 5-minute close)

1. Draw PDH, PDL, PDC. Draw the last two daily swing highs and lows (they are the "higher-timeframe draw on liquidity").
2. Draw the round numbers within 1% of the open (NIFTY every 100, SENSEX every 500).
3. At 09:30 (or 09:45) draw the opening range.
4. On every 5m close: update the session high/low, the 5m swing list, and mark any equal highs/lows.
5. Optional, from the option chain: the nearest strike with the largest call OI above and put OI below.
6. Rank each level by **freshness**: untouched today > touched once > touched twice. Zones weaken with each test
   (Seiden odds-enhancers [C]; consistent with resting depth being consumed [A]).

---

## 3. Sweep versus breakout: reading acceptance and rejection in real time

This is the central skill, and it is the one place where the practitioner rules and the evidence agree on
*mechanism* while pointing in opposite directions on *outcome* depending on the level:

- Sweeps of an **overnight or opening range** overwhelmingly revert (97%) [B].
- Breaks of the **previous day's extreme** tend to continue for the session (62-81%) [B, small].
- The academic mechanism (Osler): stops just beyond a level create a burst *through* it; whether the burst becomes
  a trend or exhausts is what the next candles tell you, not the level itself [A].

Rules, all [C] but consistent across ICT, LuxAlgo, ATAS, and Brooks:

| Signal | Sweep (rejection) | Breakout (acceptance) |
|---|---|---|
| Body vs wick | Wick through the level, body closes back inside on the same or next 1-3 candles | Body closes beyond, and the next candles hold beyond (higher lows above a broken high) |
| Time beyond | Seconds to a couple of candles | Two or more consecutive 5m closes beyond |
| Displacement | Impulsive candle *away* from the level, ideally leaving a 3-candle gap (FVG), followed by a lower-timeframe structure shift | Pause, then continuation with shallow pullbacks |
| Volume (optional) | Volume spike at the extreme with no follow-through; delta disagrees with price | Volume expands with price; note that high-volume breakouts on daily charts fail twice as often (Bulkowski) [B] |
| Penetration | Not prescribed by any source; "a few points or fifty, it does not matter" | Not prescribed |
| No-trade | Mixed closes on both sides of the level; a stop that would have to be so wide that sizing is irresponsible | |

A sweep is a **setup, not a trade** (every source). The reversal has to form: displacement plus a structure shift.
A sweep that does not reverse is a continuation setup in the other direction ("failure to manipulate", TTrades).

---

## 4. Zones: supply/demand, order blocks, fair value gaps

All definitions [C]; tests negative or absent.

- **Supply/demand zone.** Find the impulse first (a run of strong one-directional candles breaking a prior swing).
  The base is the compact 1-5 candle consolidation just before it, or the last opposite-colour candle. Distal line
  = far wick of the base; proximal line = the body edge where price left. A base that is a broad consolidation is
  not a zone. Invalid when a candle *closes* beyond the distal line, or when the departure was slow stair-stepping.
- **Order block (ICT).** The last bearish candle before a bullish impulse (mirror for bearish). Choose one boundary
  convention (full range, body, or midpoint) and never change it. Mitigation is wick-based or close-based, choose
  one. A failed order block revisited from the other side is a "breaker".
- **Fair value gap.** Three candles where candle 1's high is below candle 3's low (bullish), wick to wick. The 50%
  is the "consequent encroachment"; a fill to 50% still counts as respect; a close through the far side inverts it.
- **Freshness.** Enter only fresh zones (untouched since creation). Seiden's scoring: strength of departure,
  reward-to-risk of at least 3:1 to the nearest opposing zone, freshness.
- **Evidence.** StatOasis (648 mechanical backtests, SPY/QQQ/DIA/IWM daily, 1993-2026, frictionless): order block
  5-day edge +0.12% (t 1.22), FVG retrace -0.005% (t -0.08), OTE beat random in 0% of SPY variants, zero of 648
  beat buy-and-hold [B]. LuxAlgo, the largest vendor of these tools, states "there are no dependable published hit
  rates for these classifications" [C]. No intraday test exists; none on NIFTY.

Practical reading: a zone is a *place to look for a confirmation*, valuable because it coincides with the last
place resting orders were consumed. It is not a reason to enter.

---

## 5. Market structure: swings, BOS, CHoCH, displacement

Definitions [C], consistent across sources and the open-source `smart-money-concepts` library:

- **Swing.** 3-candle rule on the chosen timeframe, strict inequality, confirmed at the close of the third candle.
  Nest them: a short-term high flanked by lower short-term highs is an intermediate high; an intermediate high
  flanked by lower intermediate highs is a long-term high. Bill Williams fractals (5 bars) are the same idea with
  less sensitivity.
- **Trend label.** Last 3-5 swings: HH/HL = up, LH/LL = down; anything else is transition.
- **Break of structure (BOS).** Body close beyond the last same-direction swing (continuation).
- **Change of character (CHoCH) / market structure shift (MSS).** Body close beyond the last opposite swing
  (the last higher low in an uptrend). CHoCH alone is "potential"; a reversal is CHoCH followed by a BOS in the
  new direction.
- **Body vs wick.** "Bodies break structure, wicks sweep liquidity." A wick-only break is a sweep. Fix a buffer
  (for NIFTY 0.02% or 5 points) and never change it mid-session. No source supplies a tested buffer.
- **Internal vs external.** Breaks of minor fluctuations inside a leg are internal and do not count as CHoCH.
- **Displacement.** A candle whose range is large relative to recent volatility (for example, larger than 1.5 x
  the 20-candle average range) that closes beyond the level and leaves a gap. Required for a valid MSS in the
  stricter models.

Al Brooks' equivalents, useful on the 1-minute chart: a trend bar has a body; a bar pullback in an up-swing is a bar
whose low is below the prior bar's low; H1 is the first bar whose high exceeds the prior bar's high in a bull
pullback, H2 the second such attempt after a lower high; second entries (H2/L2) are preferred because trapped
counter-trend traders add fuel; H3/H4 mean the trend is weakening.

---

## 6. The pullback and the confirmation candle

### 6.1 Pullback validity and depth [C]
- A pullback holds the last higher low and does not close beyond the origin of the impulse. A close there is a
  CHoCH, not a pullback.
- Depth conventions: classic 38-62%; ICT OTE 62-79% of the impulse measured from the sweep wick (100%) to the
  displacement peak (0%), 70.5% as the mean threshold. Invalidated only by a body close beyond 100%.
- Two-legged (ABC) pullbacks are common, which is why the second entry is watched.
- **Inducement.** The first minor counter-swing inside the impulse leg is the bait. Do not enter the zone until
  that minor swing has been swept; an untouched inducement means the zone is not ready.

### 6.2 Confirmation on the 1-minute chart
The confirmation is the *only* moment money is risked, so it must be mechanical. The consistent practitioner rules:

1. Price is inside the zone (zone edge to 70-79% of the leg) or has just swept the inducement.
2. A **signal bar** closes: bullish body, close in the top 30% of its own range (Brooks H2 filter), and its close is
   above the previous 1-minute candle's high; or a 1-minute structure shift (close above the last 1-minute lower
   high) with displacement.
3. Prefer the **second attempt** (H2/L2): the first push out of the zone fails to make a higher high, the second
   push closes above the prior candle's high.
4. Enter at the open of the next candle (a stop order one tick above the signal bar high is the Brooks method).

Evidence on candle confirmation is weak: the strictest test of 28 candlestick patterns on daily DJIA data found none
different from unconditional returns (Marshall, Young & Rose 2006) [A]; Taiwan and pre-1996 S&P tests found some
reversal patterns profitable [A-]; the Indian study (17 NIFTY stocks, 2000-2015) is descriptive with no
significance test [C+]. No intraday test exists. The value of the confirmation candle is therefore **risk
definition** (it gives the stop) rather than prediction.

---

## 7. Stop, target and the arithmetic that decides everything

### 7.1 Stop placement [C, with A-level mechanism]
- Long: a buffer below the signal bar low, or below the swept liquidity level / zone's far edge, whichever
  invalidates the idea. Never inside the zone.
- Buffer: a few points beyond the level, not on it. Osler shows stops cluster *just beyond* round levels and get
  run in cascades [A], so the buffer should exceed the typical overshoot; on NIFTY 1-minute charts that is about
  0.1 to 0.25 x the 5m ATR (3-8 points), to be measured.
- Kaminski & Lo (2014): under a random walk a stop always lowers expected return; stops add value only when returns
  are serially correlated [A]. A stop is a cost of being wrong quickly, justified only if entries are selective.

### 7.2 Target by standard-deviation projection (the Fib preset 0, 1, -1, -2, -2.5, -4)
"Standard deviation" here is not statistical. One SD = the height of the anchored leg [C, all sources].

- **Anchored leg (TTrades, verbatim):** "the swing that sweeps liquidity and occurs just before a change in the
  state of delivery. It is the final high before a bearish CISD or the final low before a bullish CISD."
  Operationally: the manipulation leg = from the origin of the move into the liquidity pool (1) to the sweep
  extreme (0).
- **Direction invariant:** level 0 sits on the swept extreme, level 1 on the leg's origin, and the negative levels
  stack *away* from the extreme past the origin: below for shorts, above for longs. If the negatives appear beyond
  the swept extreme the tool is drawn backwards.
- **Meaning of the levels:** -1 = measured move, hit most often, first partial. -2 to -2.5 = expected terminus:
  "the primary area of interest where price often reacts with a retracement or reversal" (TTrades). -4 = maximum
  expansion, reachable only when a candle closes beyond -2/-2.5 ("continuation becomes more likely").
- **Mechanical rule (fxreplay):** target the 2 SD level; if that is less than 2R from the entry, use 4 SD; skip
  otherwise.
- **Cross-check:** the target must coincide with, or sit before, the opposing liquidity pool (PDH/PDL, equal
  highs/lows, session extreme). A -2.5 that lies beyond the opposing pool is reduced to the pool.
- **Time rule (ICT original):** anchored on the opening manipulation; for NIFTY that is the 09:15-09:30 range with
  the Judas move and the CISD in 09:30-09:45 (TradingFinder's stocks mode); India practitioners use 09:15-10:30,
  13:30-14:30 and 14:45-15:30 as the windows.
- **CISD** ("change in state of delivery"): a close beyond the opening price of the candle series that built the
  sweep. It is the structure-shift definition attached to this method.
- **Evidence:** none. No source publishes hit rates for -1/-2/-2.5/-4. LuxAlgo: "no audited public statistics
  exist... claimed win rates should be treated as marketing." Build Alpha: with this many degrees of freedom a
  profitable configuration is guaranteed even in random data. SSRN 7483658 (44 SMC concepts, S&P daily, 2.2M
  events): none beat random entry. Hit rates on NIFTY/SENSEX must be measured, not assumed (section 9).

Worked example (NIFTY): 09:15 open 24,000; price rallies to 24,080 by 09:27 sweeping PDH 24,075; a 5m candle closes
below 24,030, the open of the candle series that built the sweep (CISD). Leg 24,030 to 24,080, one SD = 50.
Levels: 0 = 24,080, 1 = 24,030, -1 = 23,980, -2 = 23,930, -2.5 = 23,905, -4 = 23,830. Short on a 1-minute
confirmation inside the 24,040-24,055 gap; stop 24,085; first target 23,930 (about 2.5R); hold toward 23,830 only if
a 5m candle closes below 23,905.

### 7.3 The arithmetic (this is where most methods die)
Breakeven win rate = (1 + c) / (R + 1), where R is reward-to-risk and c is cost in units of risk [B].

| Planned R | Breakeven win rate, no cost | with cost = 0.2R (typical for a 15-point NIFTY stop) |
|---|---|---|
| 1:1 | 50% | 60% |
| 1:1.5 | 40% | 48% |
| 1:2 | 33% | 40% |
| 1:3 | 25% | 30% |

Round-trip cost per lot in 2026 (STT 0.15% on sell premium, NSE 0.03553% / BSE 0.0325%, Rs 20 per order, GST,
stamp): about Rs 70 for NIFTY, Rs 66 for SENSEX, plus a bid-ask spread of Rs 30-130 per lot on NIFTY weeklies [A/B].
Delta matters as much as cost: a 5-day ATM option (delta 0.53) turns a planned 1:2 on the underlying into a
realised 1.2-1.3 after theta and spread; a 100-point in-the-money strike (delta 0.64-0.75) gives 1.4-1.5 [B, model].
On expiry morning ATM theta equals a 20-point move; only deep ITM preserves the underlying R:R. Therefore:
**5 or more days to expiry, 100 points ITM, never on expiry day, and a stop of at least 12-15 points so that costs
stay under 0.25R.** A "previous candle low" stop of 5 points on a 1-minute chart costs 0.5R in friction and needs a
60% win rate at 1:2, which nothing in the literature supports.

---

## 8. The trade model on one page (testable specification)

Every threshold marked (m) is convention and must be measured on NIFTY/SENSEX (section 9).

**Timeframes.** 15m: bias and liquidity map. 5m: structure (swings, BOS/CHoCH, displacement, zones). 1m: confirmation and stop.

**Session windows.** Trade 09:30-11:30 and 13:30-14:45 (m). No entries 09:15-09:30 or after 14:45. No trades on the index's expiry day (NIFTY Tuesday, SENSEX Thursday).

**Step 1: map (pre-open, then each 5m close).** PDH, PDL, PDC, two daily swings, round numbers, opening range (09:15-09:30), session extremes, 5m swings, equal highs/lows, largest OI strikes. Rank by freshness.

**Step 2: liquidity event (5m).** Price trades beyond a ranked level and, within 1-3 candles, a body closes back inside (sweep), *or* two consecutive 5m bodies close beyond it (acceptance). A sweep of the opening range or of an equal-highs/lows pool is a reversal candidate; acceptance beyond PDH/PDL is a continuation candidate in the break direction.

**Step 3: structure shift (5m).** Reversal candidate: a displacement candle (range > 1.5 x 20-candle average (m)) closes beyond the last opposite 5m swing (CHoCH / CISD) and leaves a gap. Continuation candidate: BOS in the break direction with a shallow pullback holding above the level. No shift within 6 candles (m): setup void.

**Step 4: zone.** Mark the last opposite-colour 5m candle before the displacement (order block, full range) and the gap (FVG). The entry zone is the overlap of that with the 62-79% retracement of the displacement leg. The zone must be fresh.

**Step 5: pullback into the zone with inducement taken.** Price returns into the zone without a 5m body close beyond the zone's far edge; the first minor 1m counter-swing formed on the way has been swept.

**Step 6: confirmation (1m).** Signal bar: bullish (for longs), close in the top 30% of its range, close above the previous 1m high; or a 1m CHoCH with displacement. Prefer the second attempt. No confirmation within 10 minutes (m): skip. Entry at the next 1m open.

**Step 7: stop.** Buffer of 0.15 x 5m ATR (m) below the lower of (signal bar low, zone far edge). If the resulting risk is under 12 points on NIFTY / 40 on SENSEX (m), the stop goes below the sweep extreme instead; if that is over 0.6% of price, skip.

**Step 8: target.** SD projection on the manipulation leg (origin = 1, sweep extreme = 0). Take one third at -1 (m); primary exit at -2 to -2.5; hold the rest toward -4 only if a 5m body closes beyond -2.5. Cap every target at the opposing liquidity pool. If -2 is less than 2R from the entry, use -4 as the primary target; if -4 is also under 2R, skip.

**Step 9: management.** Stop to breakeven after the -1 partial (m). Exit at a 5m body close beyond the last 5m swing against the position. Flatten 15:10.

**Step 10: limits.** Maximum 4 trades a day, maximum 2 consecutive losses, one position at a time, at most one trade per liquidity level per day. Instrument: 100-point ITM weekly with 5 or more days to expiry, one lot.

**Skip list (each is a documented failure mode).** Level touched twice already; sweep with no displacement; displacement but the pullback closes beyond the zone; inducement not taken; confirmation candle with a body under 50% of range; planned R under 2 after cost; entry window closed; expiry day; mixed closes on both sides of the level.

---

## 9. What must be measured before trusting any of it

Each is a single number the 3-year 1-minute data can produce; the literature offers no substitute for NIFTY/SENSEX.

1. **Hit rates of the SD ladder** on the opening manipulation leg: share of days reaching -1, -2, -2.5, -4 before returning to 0; by year.
2. **Reversal rate after a sweep** of (a) the opening range, (b) equal highs/lows, (c) PDH/PDL, defined as a 5m close back inside followed by a CHoCH within 6 candles; and the subsequent move to the opposing pool. The literature says (a) reverts and (c) continues; NIFTY must confirm.
3. **Displacement filter value:** forward 30/60-minute drift after a CHoCH with vs without a displacement candle.
4. **Confirmation candle value:** forward drift after a signal bar meeting the section 6.2 rule vs any bar in the zone.
5. **Stop buffer:** distribution of overshoots beyond swept levels (how far do stop runs go past the level on 1m).
6. **Realised R after costs** with a 100-point ITM option model versus futures, by planned R and by hold time.
7. **Time-of-day** split of all the above, since the only academically supported effect is the last-half-hour continuation and the U-shaped volatility at open and close.

Decision rule before going live: expectancy positive in every calendar year after costs, at least 150 trades in the sample, and the same sign in the walk-forward year. Any parameter (m) tuned on the data reduces the number of free years available to test it.

---

## 10. Sources

Academic: Osler 2000 "Support for Resistance" (FRBNY EPR); Osler 2003 "Currency Orders and Exchange Rate Dynamics" (J. Finance); Osler 2005 "Stop-loss orders and price cascades" (JIMF); Kavajecz & Odders-White 2004 (RFS); Brock, Lakonishok & LeBaron 1992 (J. Finance) and Fang, Jacobsen & Qin 2014 (out-of-sample failure); Lo, Mamaysky & Wang 2000; Donaldson & Kim 1993 (JFQA); Garzarelli et al. 2014 (Sci. Rep.); Chung & Bellotti 2021 (arXiv 2101.07410); Marshall, Young & Rose 2006 (JBF); Lu, Shiu & Liu 2012; Lu & Shiu 2016; Caginalp & Laurent 1998; Manoharan & Mamilla 2019 (IJITEE, India); Gao, Han, Li & Zhou 2018 (JFE); Baltussen, Da, Lammers & Martens 2021 (JFE); Li, Sakkas & Urquhart 2022 (JFM); Pacific-Basin Finance J. 2023 APAC replication; "Hedging Demand and Intraday Momentum within the Indian Stock Market" 2024; Kaminski & Lo 2014 (J. Financial Markets); Fung, Mok & Lam 2000; SSRN 7483658 (44 SMC concepts); SSRN 7428398 (pre-registered ORB); arXiv 2605.04004 (MNQ signal families).

Regulator / exchange / broker: SEBI PR 22/2024 and FY25, FY26 studies; zerodha.com/charges; Zerodha STT bulletin (Apr 2026); NSE lot-size revision Jan 2026; SEBI expiry-day framework (Tue/Thu from Sep 2025); Kite Connect historical and rate-limit docs.

Practitioner primary: ttrades.com (standard deviation projections; daily bias; failure to manipulate); ICT 2016 mentorship Month 8 notes (CBDR, projecting highs/lows); TTrades Fractal Model, TFO Range Projections, tristanlee85 ORP, TradingFinder Judas/NY setup (TradingView); fxreplay ICT AMD/PO3; backtrex SD targets and FVG playbook; LuxAlgo library (liquidity sweep, equal highs/lows, inducement, supply/demand, FVG rules, MSS, SD projections, session ranges); ictkillzone (liquidity, OTE, backtesting); Al Brooks glossary and course guide; joshyattridge/smart-money-concepts (GitHub); StatOasis "ICT backtest: what survives"; Build Alpha on backtesting ICT/SMC; Bulkowski thepatternsite (candle performers, throwbacks); tradingstats.net Asia-range cascade; edgeful previous-day range; intradaylab NIFTY ORB and gap studies; Zerodha "In The Money" ORB; vaishviktrader and dhanith (ICT for NIFTY); Sahi and MarketNetra (unsourced timing claims).
