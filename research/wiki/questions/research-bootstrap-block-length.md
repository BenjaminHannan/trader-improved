---
type: synthesis
title: "Research: automatic bootstrap block length (iteration 7)"
created: 2026-07-06
status: implemented
---

# Research: Politis-White automatic block length for the Sharpe bootstrap

## Key Findings
- Our stationary-bootstrap Sharpe CI uses a fixed avg_block=21. Block length is the
  one tuning parameter that materially moves dependent-bootstrap CIs; too short
  understates autocorrelation (CIs too tight), too long wastes power.
- Politis & White 2004 (Econometric Reviews), corrected by Patton, Politis & White
  2009: data-driven optimal expected block length for the stationary bootstrap,
  b_opt = (2 G^2 / D_SB)^(1/3) T^(1/3), with G, D from flat-top lag-window
  autocovariance estimates and a data-driven bandwidth (last significant lag rule,
  threshold c*sqrt(log10(T)/T), c≈2).

## Decision → implementation
bootstrap.py gains `politis_white_block_length(returns) -> float` (PPW 2009
formulas) and `sharpe_ci(..., avg_block="auto")` uses it (fixed number still
accepted — back-compat; floor 1, cap T/3).

## Verification (in-repo)
iid data → short blocks (< 5 typical for T=1000); AR(1) rho=0.9 → substantially
longer than iid (property, seeded); CI under "auto" still brackets the true Sharpe
of a seeded generator; b in [1, T/3] always.

## Sources
- Politis & White 2004, "Automatic Block-Length Selection for the Dependent
  Bootstrap" (public.econ.duke.edu PDF; fetch blocked, formulas canonical)
- Patton, Politis & White 2009 correction (Econometric Reviews 28(4))
