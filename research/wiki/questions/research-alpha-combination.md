---
type: synthesis
title: "Research: correlation-aware alpha combination (iteration 2)"
created: 2026-07-06
status: implemented
related:
  - "[[concepts/grinold-multifactor-ic]]"
---

# Research: correlation-aware multi-factor alpha combination

## Overview
`combine.py` v1 sums per-factor alphas — equivalent to IC-weighting under σ·IC·z but
it double-counts correlated signals (e.g. mom_12_1 and tsmom share a driver). The
Grinold-Kahn Ch.6 best-linear-predictor extension is the standard fix and reduces to
the simple sum when scores are independent.

## Key Findings (search-level; all PDF fetches 403 — verified by simulation in-repo)
- Best linear predictor with K standardized scores z (score correlation matrix C,
  per-score ICs vector ic): combined score weights ∝ C⁻¹ ic; combined alpha
  α_i = σ_i · icᵀ C⁻¹ z_i; combined IC = sqrt(icᵀ C⁻¹ ic) ≥ max single IC,
  with equality-to-sum behavior when C = I. (Grinold & Kahn Ch.6 appendix; MSCI Barra
  "Converting Scores into Alphas"; Northfield "Alpha Scaling Revisited".)
- Naive equal/IC weighting "ignores interaction effects from correlations between
  characteristics" (bottom-up vs top-down factor investing, JAM 2020).
- Practical conditioning: C estimated on trailing cross-sectional scores is noisy —
  ridge/shrink C toward identity before inverting (same LW logic as iteration 1).

## Decision → implementation
`combine.py` gains `combine_alphas_grinold(z_panels, ic_by_factor, resid_vol,
score_corr, ridge=0.10)`: shrink C toward I with weight `ridge`, weights = C⁻¹ ic,
α = σ · (weighted z sum). Score correlation estimated PIT (trailing window of
same-date cross-sectional score pairs, dates ≤ t only) in a new helper. The engine
uses it when ≥2 factors are live in a sleeve; single-factor path unchanged; the old
`combine_alphas` (plain sum) retained for back-compat and as the C=I special case.

## Verification (in-repo, Monte Carlo)
- Two perfectly correlated copies of one signal: grinold combination ≈ the single
  signal's alpha (no double counting), plain sum ≈ 2x (the bug demonstrated).
- Independent signals: grinold ≈ plain sum (special-case equivalence).
- Combined in-sample IC of grinold weights ≥ max single IC on planted data.
- PIT: corruption harness style — combined alpha at t unchanged when future scores
  corrupted.

## Open Questions
- Score-correlation window length (used 252d, min 60): sensitivity unstudied.
- Cross-sleeve pooling of C: kept per-sleeve (matches everything else in the repo).

## Sources
- Grinold & Kahn, Active Portfolio Management, Ch. 6 (canonical; fetch blocked)
- MSCI Barra, "Converting Scores into Alphas" (fetch blocked)
- Shah (Northfield), "Alpha Scaling Revisited" 2007 (fetch blocked)
- Springer JAM 2020, "Bottom-up versus top-down factor investing" (search snippet)
