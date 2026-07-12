---
type: source
source_type: papers
author: Ledoit & Wolf — "Goldilocks" (RFS 2017), "Quadratic Shrinkage" (Bernoulli 2022, QIS), "The Power of (Non-)Linear Shrinking: Review and Guide" (J. Financial Econometrics 2020/22)
date_published: 2017-2022
url: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3486378 ; https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3384500
confidence: medium-high (direct PDF fetches blocked — ledoit.net cert error, UZH mirror not fetched; claims from abstracts + multiple independent secondary sources; mechanism arguments are exact)
key_claims:
  - Nonlinear shrinkage (incl. QIS) is rotation-equivariant — it ONLY adjusts sample eigenvalues, keeps sample eigenvectors, and uses no cross-sectional prior/target
  - Its advantage over linear shrinkage grows with the concentration ratio c = N/T and vanishes as c → 0
  - QIS is optimal under Frobenius, Inverse-Stein and Minimum-Variance losses in large-dimensional asymptotics; no higher-than-quadratic nonlinearity helps under Frobenius loss
  - LW's own guide - linear shrinkage is the simple default; nonlinear adds a further level of improvement especially when combined with factor structure or time-varying volatility (DCC-NL)
---

# Nonlinear shrinkage (QIS / Goldilocks): where it pays and where it can't

What nonlinear shrinkage is: keep the sample eigenvectors, replace each sample
eigenvalue with an estimate of the average of the true eigenvalues it mixes with
(the large-dimensional asymptotics of Marchenko-Pastur). QIS (Bernoulli 2022) is
the current recommended variant; it is optimal under Frobenius/Inverse-Stein/
Minimum-Variance losses, and quadratic is the ceiling — no higher-order
nonlinearity improves Frobenius loss.

**The operative scaling law**: the gap between nonlinear and linear shrinkage is
driven by the concentration c = N/T (and by dispersion of true eigenvalues).
- c → 0: both converge to the sample covariance; differences vanish.
- c ≳ 0.3-0.5 (e.g., N=500 vs T≈1250, c=0.4): the regime the method was built
  for; documented GMV out-of-sample variance improvements over linear shrinkage
  in the LW empirical studies at N in the hundreds.

**Why it CANNOT beat a structured target at tiny N** (mechanism, exact):
rotation-equivariance means nonlinear shrinkage injects *no cross-sectional
information* — it can only de-noise eigenvalues. At N=8-25 with T_eff ≈ 130
(EWMA halflife 90d), c ≈ 0.06-0.2: eigenvalue noise is modest, and the dominant
estimation risk is in the correlations themselves. A constant-correlation target
pools all N(N-1)/2 correlation estimates into one number — a strong,
structurally-correct prior for a homogeneous ETF block — which is information a
rotation-equivariant estimator by definition cannot use. LW's own guide keeps
linear shrinkage as the sensible default at moderate dimensions and positions
nonlinear as the upgrade for large N/T, ideally combined with structure
(factor models, DCC-NL).

**Implication for our shapes**:
- fx/commodity/rates/intl/sector ETF sleeves (N=8-25): keep `lw_cc`. QIS is the
  wrong tool by mechanism, not just by folklore.
- equity factor covariance (K≈15 factors, T≈1250): c ≈ 0.01 — linear `lw` is
  already in the asymptotic regime where nonlinear adds nothing.
- The ONLY place QIS could matter here: a future instrument-level equity
  covariance (N≈500, T_eff≈130-1250, c = 0.4-4) — not currently built; the
  structural B F B' + D model is the standard alternative in exactly that regime.

Feeds [[questions/research-risk-model-validation]].
