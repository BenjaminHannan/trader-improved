# trader-improved

A Grinold–Kahn multi-factor, multi-asset trading system — clean-room rebuild with
point-in-time discipline as the load-bearing wall. Four sleeves (US equities, crypto,
FX via ETF proxies, commodities via ETF proxies), daily bars, weekly rebalance,
walk-forward validation with deflated Sharpe.

**Everything here is free-data only** (see the availability rules below) and
**no ML** — v1 is deliberately boring, well-documented premia combined linearly.

## The one idea

Active management is forecasting the market's errors, and the fundamental law says
IR ≈ skill × √breadth. This system is a breadth machine: many small, independent,
cost-netted bets across four weakly-correlated sleeves, sized by a factor risk model
and combined at the allocation layer.

Every layer is built around a single non-negotiable: **a value may only be used at
time t if it was knowable at time t.** That is enforced structurally, not by
convention:

- every curated row carries `available_from` (UTC knowability timestamp); the lake
  writer rejects rows without it;
- the only sanctioned data→decision join is `production/core/pit.py:asof_panel`;
- every registered signal, the risk exposures, and the macro overlay pass a
  **corruption harness** (`tests/test_no_lookahead.py`): corrupt all data after a
  pivot date, recompute, assert history ≤ pivot is bit-identical. A new signal is
  covered the moment it's registered;
- rolling IC windows embargo the unresolved forward-return tail positionally;
- trailing ADV/σ for costs and overlays are `shift(1)`-ed;
- macro comes from ALFRED vintages, COT respects its Tuesday→Friday release lag.

## Layout

```
configs/          universe / factors (registry + gate) / costs / risk / backtest
production/
  core/           config, calendars, parquet lake (schema-enforced), pit.py
  reference/      instrument master (never-reused ids), PIT universes, ticker hygiene
  data/           BaseLoader (fetch→transform→stamp→audit→write), Stage-1 loaders
  signals/        momentum, reversal, low-vol, carry, tsmom, COT positioning
  alpha/          z-scores → rank-IC → α = σ·IC·z → IC-weighted combine; hard gate
  risk/           Σ = BFBᵀ + D (equity/crypto); shrunk EWMA cov (small sleeves)
  portfolio/      cvxpy QP (factor-form), declarative constraints, overlays, ERC
  backtest/       walk-forward engine, costs, deflated Sharpe, bootstrap, attribution
  execution/      (later sessions) orders, Alpaca paper, shortfall
  monitor/        (later sessions) live IC decay alarms
scripts/          ingest.py, build_factors.py, run_backtest.py
tests/            the verification gates for every phase
data/             parquet lake (gitignored): raw / curated / reference / panels / audit
reports/          one artifact per backtest run (gitignored)
```

## Quick start

```bash
uv sync

# full test suite (no network needed)
uv run pytest -q

# synthetic end-to-end backtest (no data needed)
uv run python scripts/run_backtest.py --synthetic --start 2019-01-01 --end 2021-12-31

# real data bootstrap (needs outbound network; FRED_API_KEY optional but recommended)
uv run python scripts/ingest.py --dataset universe
uv run python scripts/ingest.py --dataset all --start 2016-01-01
uv run python scripts/build_factors.py --ic-report          # gate verdicts
uv run python scripts/build_factors.py --ic-report --apply  # record into factors.yaml
uv run python scripts/run_backtest.py --config configs/backtest.yaml
```

## Data sources (Stage 1) and their availability stamps

| Loader | available_from rule |
|---|---|
| yfinance / stooq prices (equities + ETFs) | obs_date 21:30 UTC (post-NYSE close) |
| ccxt crypto daily bars | bar close (midnight UTC + 24h) |
| ccxt funding rates | last funding timestamp of the day |
| Frankfurter ECB FX reference rates | obs_date 15:00 UTC |
| FRED / **ALFRED vintages** (VIX, HY OAS, short rates) | `realtime_start` — zero revision leakage |
| CFTC COT | Tuesday obs → **Friday 20:30 UTC** release |
| Ken French library (validation only) | obs + 5 business days |
| CoinGecko / DefiLlama / GDELT | ingest time (snapshots, no rewritten history trusted) |

Stage-2 (WEI, GDPNow, AAII/NAAIM, put/call, short interest) and Stage-3 exotics
(ENTSO-E, JODI, VIIRS, OpenSky, Zillow/Redfin) follow the same contract —
Stage-3 exists as documented stubs in `production/data/loaders/stage3_stubs/`.

## The factor gate

Factors live in `configs/factors.yaml` forever (candidate → accepted/rejected).
Acceptance requires, on data strictly before the OOS period: train rank-IC
|t-stat| ≥ 2.0; OOS slice IC same-sign and |IC| ≥ 0.005; IC decay half-life ≥ the
rebalance interval; positive net-of-cost validation return. `n_trials` — the
multiple-testing count feeding the deflated Sharpe — is part of the config,
incremented every time the gate runs, never guessed.

## Documented caveats (deliberate v1 scope)

- **Survivorship**: PIT S&P 500 membership (Wikipedia walk-back) removes selection
  bias, but delisted names' missing final returns in free price data remain a
  residual optimistic bias — documented per source in audit records.
- **GICS sectors are current-snapshot**, not PIT — used only for risk-model dummies.
- **Size proxy**: log trailing dollar ADV (no free PIT market cap).
- **No equity value factor** until an EDGAR PIT fundamentals pipeline exists.
- **Borrow assumed free** for shorts — flagged in every report as optimistic.
- FX/commodity sleeves trade **ETF proxies** (Alpaca can't trade spot FX/futures);
  `proxy_of` documents each mapping.
- No ML, no options/futures/intraday, no sentiment/NLP stack in v1.

## Reading list

Grinold & Kahn, *Active Portfolio Management* — the whole architecture is this book
run through a free-data, four-sleeve constraint set. Bailey & López de Prado for the
deflated Sharpe; Almgren-Chriss for the cost model shape.
