---
type: concept
title: Combining EWMA weighting with Ledoit-Wolf shrinkage
confidence: medium-high (standard practice in commercial risk models; direct paper fetches blocked — validated by simulation in-repo)
---

# EWMA x Ledoit-Wolf

Two orthogonal corrections to the sample covariance:

- **EWMA** handles *non-stationarity*: exponential weights (halflife h) adapt to
  time-varying vol/correlation. RiskMetrics lambda=0.94 ≈ 11d halflife for daily VaR;
  slower halflives (60-120d) for allocation-horizon covariance. This repo uses 90d
  (factor/instrument cov) and 42d (specific) — inside the accepted band.
- **Shrinkage** handles *sampling error*: pull the (weighted) sample matrix toward a
  structured target with intensity that reflects estimation noise vs target bias.

Combination used here: compute weighted moments with EWMA weights, then apply the LW
intensity formula with the **effective sample size** T_eff = (sum w)^2 / (sum w^2)
in place of T (Kish). For halflife h, T_eff -> 2h * ln2... actually T_eff ≈ 2.885*h
for long histories — materially smaller than raw T, so EWMA estimators warrant MORE
shrinkage than equal-weight ones at the same window length. Hand-set fixed intensity
(0.3) ignores this entirely.

Decision for trader-improved (iteration 1):
- factor covariance (K≈15): LW with **diagonal** target (factor returns are
  near-orthogonal by WLS construction; diagonal is the principled target) —
  analytic intensity replaces the hand-set 0.3.
- small-sleeve instrument covariance (N=8-12): LW with **constant-correlation**
  target (ETFs within a sleeve share a common driver; r_bar is the right prior).
- config-selectable, old fixed path retained (`method: fixed`) for reproducibility.
