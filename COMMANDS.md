# pa_engine v3 - command reference

Price-action engine built from `Price_Action_Trading_Algorithm_Fresh_Start_Specification.docx`.
All commands run from the project root with the project virtualenv (`.venv/bin/python`).
Credentials live only in `.env` (copy `.env.example`); nothing secret is ever committed.

## Layout

| Path | What |
|---|---|
| `pa_engine/config.py` | every threshold of the spec as a dataclass; JSON save/load; dotted overrides (`--set risk.min_rr=1.5`) |
| `pa_engine/candles.py` | 1-min candles, sessions, completed-candle aggregation (5m context / 3m setup / 1m execution), ATR |
| `pa_engine/swings.py` | ATR-filtered swings with confirmation latency |
| `pa_engine/structure.py` | HH/HL/LH/LL labels, BOS / CHOCH by close beyond a noise band, context regime |
| `pa_engine/levels.py` | PDH/PDL, opening range, session extremes, recent swings, equal highs/lows, freshness |
| `pa_engine/liquidity.py` | sweep detection (penetration in ATR, reclaim, wick ratio) |
| `pa_engine/setup.py` | setup state machine: LIQUIDITY_EVENT -> STRUCTURE_SHIFT -> WAITING_FOR_RETEST -> WAITING_CONFIRMATION -> ENTRY_CONFIRMED |
| `pa_engine/risk.py` | structural stop + ATR buffer, min/max stop, structural target, min R:R, sizing |
| `pa_engine/position.py` | exit state machine INITIAL -> PROFIT_1R -> PROFIT_2R -> RUNNER -> EXIT; partials, breakeven, structure/ATR trailing, time and invalidation exits, adverse-first evaluation |
| `pa_engine/engine.py` | per-session engine shared by backtest and live; scoring; day limits |
| `pa_engine/backtest.py` | deterministic replay, next-candle fills, costs, chronological splits, metrics, research matrix |
| `pa_engine/costs.py` | option / futures / raw-points economics (Zerodha charges, STT 0.15% on sell premium) |
| `pa_engine/live.py` | paper / live runner on Kite (one process per symbol) |
| `pa_engine/broker_kite.py` | Kite Connect adapter: minute bars, contract selection by target delta, MIS market orders with verification |
| `pa_engine/logger.py` | JSONL event log (every sweep, shift, retest, rejection, order, exit, stop change) |
| `tests/test_pa_engine.py` | unit tests (`.venv/bin/python -m pytest -q`) |
| `research/v3/` | backtest outputs: trades.csv, metrics.json, config.json, matrix.csv, results note |

## Backtest

```bash
.venv/bin/python -m pa_engine backtest --symbol NIFTY  --data data/nifty_3y_1min.csv.gz  --out research/v3/nifty_baseline  --events
.venv/bin/python -m pa_engine backtest --symbol SENSEX --data data/sensex_3y_1min.csv.gz --out research/v3/sensex_baseline
# override any config key; run a date range; save the versioned config next to the trades
.venv/bin/python -m pa_engine backtest --symbol NIFTY --data data/nifty_3y_1min.csv.gz --set risk.min_rr=1.5 exit.trail=structure --start 2025-01-01
```
Output: `trades.csv` (one row per trade with setup id, score, regime, level, stop, targets, fills, costs),
`metrics.json` (all + dev 2023Q4-2024 / val 2025 / oos 2026 splits), `config.json`, optional `events.jsonl`.

## Research matrix (spec section 32)

```bash
.venv/bin/python -m pa_engine matrix --symbol NIFTY  --data data/nifty_3y_1min.csv.gz  --out research/v3/matrix_nifty
.venv/bin/python -m pa_engine matrix --symbol NIFTY  --data data/nifty_3y_1min.csv.gz  --only baseline no_retest trail_atr
```
Runs one config change at a time (retest on/off, sweep penetration, BOS band, score threshold, stop buffer,
min R:R, partials, trailing method, time/invalidation exits, futures vs option costs, raw points) and
tabulates net / trades / profit factor / average R per chronological split.

## Daily Kite login (token expires every morning)

```bash
.venv/bin/python -m pa_engine login              # prints the login URL, asks for the request_token
.venv/bin/python -m pa_engine login --request-token XXXX
```
Writes `Dependencies/kite_token_<date>.txt` (gitignored).

## Paper trading

```bash
.venv/bin/python -m pa_engine paper --symbol NIFTY
.venv/bin/python -m pa_engine paper --symbol SENSEX
```
One process per symbol. Every minute it pulls the last completed spot candle from Kite and runs the same
engine as the backtest. Entries are simulated at the live premium of the nearest-expiry option whose delta is
closest to `cost.delta` (default 0.72). Output under `runtime/` (gitignored): `events_<SYMBOL>_<date>.jsonl`,
`trades_<SYMBOL>.csv`, `state_<SYMBOL>.json`. Telegram alerts on setup, entry, exit, start and stop.
Create the file `runtime/STOP` to stop a runner cleanly (it flattens any open paper position).

## Live trading (real money)

Not enabled by default. Requires `PAPER_TRADING=false` in `.env`, the `--live` flag, and typing `LIVE` at the
prompt. Do this only after the walk-forward results and a paper-trading period justify it.

```bash
.venv/bin/python -m pa_engine paper --symbol NIFTY --live
```

## Tests

```bash
.venv/bin/python -m pytest -q
```
