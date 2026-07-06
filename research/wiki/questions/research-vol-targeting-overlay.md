---
type: synthesis
title: "Research: volatility-targeting overlay evidence (iteration 3)"
created: 2026-07-06
status: implemented
---

# Research: vol-targeting overlay — what the evidence actually supports

## Overview
Moreira & Muir (JF 2017) made inverse-variance scaling famous. The follow-up
literature substantially narrows the claim; this iteration hardens our overlay to the
robust version rather than expanding it.

## Key Findings
- M&M 2017: scaling by inverse past-month realized variance raises Sharpe across many
  factors (in-sample, cost-free).
- **Contradiction**: Cederburg, O'Doherty, Wang & Yan (JFE 2020, 103 strategies):
  out-of-sample, estimation error renders almost all of the gain insignificant.
  Barroso & Detzel (2021): leverage-driven turnover costs erode most factor-level
  gains; only **market** and **momentum** scaling partially survive costs.
  DeMiguel, Martín-Utrera et al. (JF 2024): multifactor view — gains concentrate when
  applied to the aggregate, not factor-by-factor.
- "Smoothing volatility targeting" (arXiv 2212.07288): smoothed vol estimates retain
  the risk-reduction benefit with materially less turnover than 1-month inverse
  variance.

## Verdict for trader-improved
- Total-portfolio-level overlay (our design) is the variant the literature supports.
  **Do NOT add per-factor/per-sleeve vol scaling** — filed as a deliberate negative
  result.
- Two cost-motivated hardenings adopted:
  1. EWMA-smoothed vol estimate (halflife option) instead of a raw 21d window —
     less whipsaw in the multiplier;
  2. multiplier **deadband** (hysteresis): re-use the previous multiplier unless the
     new one differs by more than `deadband` (default 10%) — directly attacks the
     Barroso-Detzel cost channel; leverage changes only when they matter.
- Clip [0, 1.5] retained (no added leverage aggression: the OOS critique).

## Verification (in-repo)
Known-answer smoothing; deadband holds multiplier within band and updates outside it;
PIT unchanged (strictly trailing, shift(1)); constant-vol series ⇒ multiplier
converges to target/realized under both estimators; turnover of the multiplier path
on noisy synthetic vol is strictly lower with deadband+smoothing than without.

## Open Questions
- Deadband width (10%) is a judgment call; cost-optimal width depends on realized
  AUM/cost regime — revisit with real-data backtest A/B once ingest runs locally.

## Sources
- Moreira & Muir, "Volatility-Managed Portfolios", JF 2017 (search snippets)
- Cederburg, O'Doherty, Wang, Yan, "On the Performance of Volatility-Managed
  Portfolios", JFE 2020 (SSRN 3357038; fetch blocked)
- Barroso & Detzel 2021 (transaction-cost erosion; via search snippets)
- DeMiguel et al., "A Multifactor Perspective on Volatility-Managed Portfolios",
  JF 2024
- arXiv 2212.07288, "Smoothing volatility targeting"
