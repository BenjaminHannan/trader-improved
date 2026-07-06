---
type: synthesis
title: "Research: sector exposure threading gap (iteration 9, internal audit)"
created: 2026-07-06
status: implemented
---

# Internal audit: sector dummies are dead code in engine runs

## Finding
`risk/exposures.py` supports GICS sector dummies (risk.yaml `sector_dummies: true`),
and `portfolio/constraints.py` supports the ±5% sector-band constraint — but
`backtest/engine.py` never constructs or passes a `sectors` mapping. In every run to
date the equity risk model has had NO sector factors and the sector constraint could
never bind. Grinold-Kahn context: industry factors are among the largest common
drivers of equity comovement; omitting them pushes sector risk into specific risk,
understating portfolio factor risk and overstating diversification.

## Decision → implementation
- engine.run_backtest accepts optional `sectors: pd.Series | None` (instrument_id ->
  sector label); when provided, threads it into RiskModel.build and optimize_sleeve
  for the equity sleeve only.
- scripts/run_backtest.py builds it from the instrument master's `sector` column
  when available (current-snapshot GICS — the documented v1 caveat stands).
- Synthetic path: conftest instruments gain deterministic fake sectors so the
  threading is exercised in tests (round-robin 3 sectors over the 10 equity ids).
- Test: with sectors threaded, the equity RiskModel B matrix contains sector_*
  columns during the run, and post-solve sector net exposures respect the band.

## Sources
Internal (repo audit); Grinold & Kahn Ch. 3 (industry factors) — conceptual.
