---
type: synthesis
title: "Research: signal purification against risk factors (iteration 21)"
created: 2026-07-06
status: implemented
---

# Research: pure-factor signals — strip unpaid risk exposure from alphas

## Key Findings
- Pure factor portfolios (multivariate-regression construction, zero exposure to all
  non-target factors) show materially lower volatility at similar return vs naive
  factor portfolios — realized IRs ~0.74 (value), ~0.62 (momentum) in pure form
  (CXO/Tzotchev/pure-factor literature). Grinold-Kahn: neutralize alphas against
  risk factors on which you have no view — incidental beta/sector/vol exposure adds
  variance without forecast power and mechanically dilutes IC.

## Decision → implementation
- `alpha/purify.py`: `purify_scores(z_panel, exposures_by_date_or_B, ...)` — per
  rebalance date, cross-sectional OLS of the z-scores on the risk exposure matrix B
  (market/size/vol + sectors when present; the TARGET style column, e.g. momentum
  for mom signals, is EXCLUDED from the neutralization set to avoid stripping the
  signal itself); keep residuals, re-winsorize/z. PIT: B at date t is the same
  exposures the risk model builds from data <= t.
- Engine: purification applied between z-score and refine, config flag
  `alpha.purify: true` (backtest.yaml), on by default for structural sleeves
  (equity, crypto), off for tiny ETF sleeves (N too small to regress on K
  exposures).
- Verification: planted signal = true_alpha + 0.8*beta contamination → purified IC
  vs beta-neutral forward returns strictly higher than raw IC; purified scores have
  ~zero cross-sectional correlation with each B column; corruption harness on the
  purification step; N < K+3 groups skip purification (fallback raw).

## Sources
- Grinold & Kahn Ch. 14 (alpha analysis / neutralization); CXO "Purified Factor
  Portfolios"; Tzotchev (SSRN 4902957); Flirting with Models "Pursuing Factor
  Purity" — search-level.
