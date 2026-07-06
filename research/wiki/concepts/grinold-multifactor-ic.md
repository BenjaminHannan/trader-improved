---
type: concept
title: Grinold multi-factor IC (best linear predictor over correlated scores)
confidence: high (textbook result; verified by simulation in tests/test_alpha_refine.py)
---

# Grinold multi-factor IC

With K standardized signals z_k (cross-sectional mean 0, std 1), per-signal ICs
ic = (ic_1..ic_K), and score correlation matrix C = corr(z_j, z_k):

- optimal combination weights on scores: w = C⁻¹ ic
- combined forecast: α_i = σ_i · Σ_k w_k z_{k,i}
- combined IC: IC* = sqrt(icᵀ C⁻¹ ic)

Limits: C = I ⇒ w = ic ⇒ the plain σ·IC·z sum (repo v1). Duplicated signal
(corr → 1) ⇒ the pair shares one signal's worth of weight instead of double-counting.
Noisy Ĉ inverts badly ⇒ shrink toward identity (ridge) before inverting.

PIT note: C must be estimated from score history strictly ≤ decision date; per-date
cross-sectional correlations averaged over a trailing window keep it causal.
