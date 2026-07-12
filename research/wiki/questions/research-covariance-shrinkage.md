---
type: synthesis
title: "Research: covariance shrinkage upgrade (iteration 1)"
created: 2026-07-06
status: implemented
related:
  - "[[sources/ledoit-wolf-2003-honey]]"
  - "[[concepts/ewma-shrinkage-combination]]"
---

# Research: covariance shrinkage upgrade

## Overview
The repo shipped with a hand-set shrinkage intensity (0.3, diagonal target) — flagged
in the plan as "Ledoit-Wolf later". Research confirms the LW analytic intensity is the
standard replacement and that EWMA weighting demands a T_eff correction.

## Key Findings
- Sample covariance error is systematically amplified by MV optimizers; shrinkage
  directly raises realized IR ([[sources/ledoit-wolf-2003-honey]]).
- Constant-correlation target for homogeneous asset blocks; diagonal target when the
  series are near-orthogonal by construction (our WLS factor returns).
- EWMA weights shrink the effective sample size (Kish T_eff = (Σw)²/Σw²); the LW
  intensity must use T_eff, not raw T ([[concepts/ewma-shrinkage-combination]]).

## Decision → implementation
`production/risk/covariance.py` gains `ledoit_wolf_shrinkage(returns, target=
"diagonal"|"constant_correlation", ewma_halflife=None)` returning (Sigma, delta).
risk.yaml: `factor_covariance.method: lw` (diagonal target),
`instrument_covariance.method: lw_cc`; `fixed` retains the old behavior.
Verification: Monte Carlo — LW beats sample cov in Frobenius loss at T≈2N; delta in
[0,1]; delta→0 as T→∞; PSD preserved; EWMA T_eff < raw T implies higher delta.

## Contradictions
None found at search level; nonlinear shrinkage (Goldilocks 2017) is strictly better
asymptotically but unnecessary at K≈15/N≈12 — deferred (would matter for the equity
sleeve's instrument-level cov if that path is ever built, N≈500).

## Open Questions
- Direct paper fetches blocked by sandbox proxy (ledoit.net 403, arxiv 403,
  alcapitaladvisory 403) — formulas taken from canonical literature and verified by
  simulation instead. Re-verify against the PDFs when run on a normal network.
- Nonlinear shrinkage for large-N: revisit if equity instrument-level cov is built.

## Sources
- [[sources/ledoit-wolf-2003-honey]] — Ledoit & Wolf 2003
- WebSearch corroboration: Ledoit & Wolf "Review and Guide" (2020, JFEc);
  RiskMetrics EWMA lambda=0.94 convention (multiple hits)
