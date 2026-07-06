# trader-improved — hard rules

Grinold-Kahn multi-factor, multi-asset system. These rules are non-negotiable; they
encode the lessons from the reference `trader` repo's past failures.

## Point-in-time (PIT) is non-negotiable
- Every curated dataset row carries `available_from` — the UTC timestamp at which the
  value was *knowable*. Data is joined **only** through `production/core/pit.py:asof_panel`.
- Signals are pure functions of PIT inputs: value at date D uses only rows with
  `obs_date <= D` (and, upstream, `available_from <= as-of`). Every registered signal
  must pass the corruption harness in `tests/test_no_lookahead.py` — no exceptions.
- Any rolling statistic used in normalization, IC, risk, or overlays is trailing-only
  and `shift(1)`-ed where it feeds a same-day decision.
- Macro series come from ALFRED vintages (`realtime_start`), not revised FRED history.
- CFTC COT: Tuesday `obs_date`, Friday 20:30 UTC `available_from`. Respect the lag.

## Cost floors
- Costs are never zero. Per-sleeve floors from `configs/costs.yaml`:
  equities >= 5bp, crypto >= 30bp, FX-ETF >= 5bp, commodity-ETF >= 10bp.
- Impact: Almgren-Chriss square-root, `alpha = 0.15`, using **trailing, shift(1)**
  ADV and vol. Cap 100bp with an audit warning, never silently.

## Factor gate thresholds (alpha/registry.py)
A factor moves candidate -> accepted only if, on data strictly before the OOS period:
1. train-window rank-IC t-stat `|t| >= 2.0`;
2. OOS validation slice (last 20% of train span): IC same sign and `|IC| >= 0.005`;
3. IC decay half-life >= rebalance interval;
4. net-of-cost single-factor return > 0 on the validation slice.
`n_trials` for deflated Sharpe = every factor ever moved past candidate, tracked in
`configs/factors.yaml` — the multiple-testing count is config, not a guess.

## Hygiene
- `instrument_id` is a synthetic key (`class:symbol:first-listing-date`) that is never
  reused; vendor symbols map to it with PIT validity windows. This kills the
  ticker-reuse class of bug at the schema level.
- Delisted-symbol-reuse blocklist + sub-$0.10 price backstop in `reference/hygiene.py`.
- `data/` is a gitignored parquet lake. **No report files in `data/`** — backtest
  reports go to `reports/` (also gitignored) as a single artifact per run.
- `research/` is scratch; `production/` never imports from it.

## Verification
Each build phase ends with its pytest command green before the next phase starts.
`uv run pytest -q` must be green before any commit.
