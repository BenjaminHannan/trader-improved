---
type: source
title: "Makers and Takers: The Economics of the Kalshi Prediction Market"
source_type: working paper (MPRA 126350 / CEPR DP20631 / UCD WP2025_19)
author: Constantin Bürgi, Wanying Deng, Karl Whelan
date_published: 2025 (data through April 2025)
url: https://mpra.ub.uni-muenchen.de/126350/
confidence: high (full text read 2026-07-10; numbers below quoted from the paper)
key_claims:
  - sub-10c contracts lose >60% of stake on average (post-fee)
  - favorite-longshot bias is much stronger for takers than makers
  - the bias persisted every year 2021-2025 (MZ slope 0.021-0.048)
  - Economics-category slope significant but its constant is NOT
---

# Bürgi–Deng–Whelan 2025 — the anchor for diagnostic #1

Read in full (48pp) on 2026-07-10 before running our replication; exact numbers
recorded here so the pre-registration is auditable.

## Data & method
Transaction-level API data, Kalshi inception (2021) → April 2025. Filters:
volume at closure ≥ $1,000; final bid-ask spread ≤ 20c; market open ≥ 24h
(drops hourly crypto/index re-sets). 46,282 Yes contracts on 12,403 events;
313,972 contract prices (Yes+No), daily last-trade prices T-1..T-10.
Returns: pre-fee r=(y−p)/p; post-fee r=(y−p−c)/(p+c) with c imputed from a
100-lot (fee = ceil-to-cent of 0.07·C·p(1−p)).

## Headline results
- Calibration: low-price contracts win far less often than priced; **average loss
  rates for ≤10c contracts exceed 60%**; small positive post-fee returns above
  ~70c (95c contracts win ~98%, pre-fee +3.1%).
- Average pre-fee return across all contracts −20% (equal-weight per contract);
  post-fee −22%. Fees are minor relative to the mispricing itself.
- **Taker vs maker**: the pattern is much stronger for takers (trade records carry
  the taker side); makers' longshot losses are far smaller.

## Mincer-Zarnowitz y−p = α + ψp (SE clustered by event & contract)
- By year (Table 9): ψ = 0.041*** (2021), 0.023** (2022), 0.036*** (2023),
  0.048*** (2024), **0.021\* (2025 — weakest, p<0.1)**. The re-rank trigger
  ("demote if insignificant in the most recent year") is NOT tripped, but 2025
  is borderline.
- By category (Table 8): **Economics: ψ = 0.034*** but constant −0.978 (SE 0.972,
  not significant)**, n=24,405. The pooled constant is −1.736***. The
  Economics-only longshot LEVEL is noisier than the headline — recorded before
  running our Economics-only replication.

## Relevance to us
Our diagnostic #1 restricts to US macro-release ladders (a subset of their
Economics category), uses their filters/fee imputation, and pre-registers the
tradeable NO-side version (fade 5-20c longshots) net of taker fees. Their maker
finding also feeds backlog #19 (maker-side events execution: makers also lose
~10% on average — the seat alone is not an edge).

## Used by
[[questions/research-kalshi-mechanism-diagnostics]],
[[sources/practitioner-mechanism-scan-2026-07]] (idea #1 anchor)
