---
type: synthesis
title: "Research: robust-alpha ellipsoid in the QP (iteration 19)"
created: 2026-07-06
status: implemented (opt-in, default off)
---

# Research: alpha-uncertainty-aware optimization

## Key Findings
- Goldfarb & Iyengar 2003: ellipsoidal uncertainty on expected returns turns the MV
  problem into an SOCP — max α'w − κ_r·||Ω^{1/2}w|| − λ·w'Σw, where Ω is the
  estimation-error covariance of α and κ_r the confidence radius. cvxpy expresses
  the extra term as one SOC atom; the upgrade is in-place (why we chose cvxpy).
- Later work (joint ellipsoids) shows robust portfolios are MORE diversified and
  cheaper to trade than naive MV — estimation error concentrates bets; robustness
  spreads them.
- Natural Ω for our α = σ·IC*·z pipeline: the dominant error is IC estimation.
  se(IC) = std(IC_history)/sqrt(n_eff) ⇒ per-name alpha error sd ≈ σ_i·se_IC·|z_i|;
  diagonal Ω = diag((σ_i·se_IC·z_i)²) is the first-order separable set.

## Decision → implementation
- optimize_sleeve gains `alpha_se: pd.Series | None` and cfg key
  optimizer.robust_kappa (DEFAULT 0.0 = term absent, objective bit-identical —
  v1 stays deliberately boring; robustness is opt-in).
- engine computes alpha_se per sleeve from the rolling IC dispersion it already
  tracks and passes it through; robust_kappa read from config.
- Properties verified in-repo: kappa=0 identical to current solution; increasing
  kappa shrinks gross exposure and increases diversification (effective N); with
  two equal-alpha names where one has 3x the alpha uncertainty, the robust solution
  underweights the uncertain one; solver stays CLARABEL (SOCP-native).

## Sources
- Goldfarb & Iyengar, "Robust Portfolio Selection Problems", Math of OR 2003
- Lu, "Robust portfolio selection based on a joint ellipsoidal uncertainty set",
  OMS 2011 (diversification result) — search-level
