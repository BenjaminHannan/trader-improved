---
type: synthesis
title: "Research: prediction-market sleeve (iteration 25)"
created: 2026-07-06
status: implemented
---

# The events sleeve: prediction markets as pure breadth

## What was built
Kalshi + Polymarket loaders (curated `event_markets`, ingest-time availability),
liquidity/nearness universe filters, correlated-resolution dedup (one event group =
one bet), two documented edges (favorite-longshot bias fade; resolution-convergence
drift), Bernoulli-variance fractional-Kelly sizing (0.25x) with a 200bp round-trip
fee haircut, and a PIT walk-forward return stream integrated into the ERC
allocation layer as the 8th sleeve — no optimizer/Gaussian-risk path; its risk
enters via the EWMA sleeve covariance plus a hard 10% risk-contribution cap.

## Why it earns its place
Event outcomes are ~uncorrelated with every factor sleeve — the purest breadth
available at retail scale. Capacity is tiny, which the caps encode (events: 0.10
risk cap, 2% per-market position cap, per-group Kelly caps).

## Approximations (documented in code)
- Settlement snapped from final prints (>=0.97 -> 1, <=0.03 -> 0, else last price).
- Signal scores pass through as believed edge; calibration of score->probability
  is the first real-data improvement (live resolution outcomes will supply it).
- Fees baked into the net stream (gross == net for events).

## Found along the way
`allocation.risk_caps` config was DEAD — sleeve_allocation never applied it (the
fourth wiring gap; the call-site audit lesson holds). Now generalized to arbitrary
per-sleeve caps and actually binding, with the legacy signature preserved.

## Sources
Iteration-17/23 syntheses; Kalshi trade-api v2 + Polymarket gamma/CLOB public docs
(endpoints recorded in the events agent's report for real-ingest debugging).
