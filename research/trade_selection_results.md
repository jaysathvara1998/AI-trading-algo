# Trade-selection study on the v2 engine (28 Sep 2026)

Question: does anything known at entry separate the momentum engine's profitable trades from the rest, enough to
clear the Rs 73 statutory cost per round trip (plus ~Rs 30 bid-ask allowance)?
Data: 3-year replay trade logs of the fixed engine (2,642 NIFTY / 2,769 SENSEX trades, gross P&L each) joined to
the 3y bar files. Scripts: `trade_selection.py`, `vol_day_filter.py`.

## What separates trades (gross Rs per trade)

| Feature | NIFTY | SENSEX |
|---|---|---|
| Opening-range width vs trailing 20-day median, narrowest -> widest fifth | -22, -56, +93, +105, +110 | -35, -19, +81, +103, +160 |
| Gap vs previous close: <0.2% / 0.2-0.5% / 0.5-1% / >1% | +7 / +55 / +128 / +154 | +29 / +60 / +89 / +236 |
| Monday / Tue-Fri | +112 / +32 | +127 / +42 |
| Trade against the day's move so far / with it | +64 / +32 | +91 / +30 |
| Entry 09:15-09:45 / 09:45-11:00 | +53 / +1 | +61 / +29 |

The engine's gross edge is concentrated on high-volatility sessions (wide opening range, large gap, Mondays).
Nothing about the signal itself (side, 15-min momentum alignment, trade number in the day) separates trades.

## Implementability kills most of it

63% of NIFTY entries and 66% of SENSEX entries occur before 09:30, when the 15-minute opening range is not yet
known. Re-evaluated using only information available at each entry (gap at 09:15, 5-min range from 09:20,
15-min range from 09:30), with a Rs 30 spread allowance, over 3 years:

| Implementable rule | NIFTY net (trades/day) | SENSEX net (trades/day) | Years positive N / S |
|---|---|---|---|
| gap >= 0.5% only | +15,375 (0.65) | +13,890 (0.69) | 3/4 / 3/4 |
| or5 >= 1.0 (09:20-30) or or15 >= 1.0 (09:30+) or gap >= 0.5% | -24,698 (1.65) | -15,369 (1.72) | 1/4 / 1/4 |
| or5 >= 1.2 or or15 >= 1.2 or gap >= 0.5% | -22,328 (1.30) | +15,159 (1.34) | 2/4 / 3/4 |
| entries 09:30-09:45 with or15 >= 1.2 | -20,679 (0.36) | -11,004 (0.36) | 1/4 / 2/4 |
| unfiltered | -145,407 | -121,835 | 0/4 / 0/4 |

Trades taken after 09:30, where the opening-range filter is usable, are the engine's worst trades; the
filter cannot rescue them. The gap rule is the only survivor: about Rs 5,000 per index per year, on 0.65
trades a day, negative in one year on each index. It does not meet the pass bar (t >= 2 after costs, every
year, both indices).

## Verdict

The volatility conditioning is real (gross t of 3 to 4) but the engine enters too early in the session to
use it, and the entries late enough to use it have no edge. The momentum engine is finished as a strategy:
no implementable selector turns it positive after realistic costs. Keep it in paper mode only.

Carried forward as a design constraint for anything new: intraday directional edge on these indices shows
up on high-volatility days only, and any future entry rule must be evaluated with a volatility gate that is
known at entry time.
