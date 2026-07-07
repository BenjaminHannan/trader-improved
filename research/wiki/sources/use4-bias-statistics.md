---
type: source
source_type: vendor methodology (MSCI Barra) + practitioner assembly guide
author: Menchero, Orr & Wang (MSCI USE4 Methodology/Empirical Notes, 2011); itsjustbeta.com risk-model assembly chapter
date_published: 2011 / evergreen
url: https://www.top1000funds.com/wp-content/uploads/2011/09/USE4_Methodology_Notes_August_2011.pdf ; https://www.itsjustbeta.com/chapters/08-risk-model-assembly/
confidence: high (formulas cross-confirmed between USE4 notes summaries and the assembly guide; PDF direct fetches blocked, HTML source quoted verbatim)
key_claims:
  - "Bias statistic B = std(z_t), z_t = r_{p,t} / sigma_hat_{p,t-1}; B≈1 calibrated, B>1 under-forecast, B<1 over-forecast"
  - "95% acceptance band ≈ 1 ± sqrt(2/T) under normality (T=52 weekly obs → [0.80, 1.20])"
  - "Newey-West horizon scaling: F_NW = Gamma_0 + Σ_{q=1..Q} (1 - q/(Q+1)) (Gamma_q + Gamma_q'); Bartlett weights keep PSD; omitting it UNDERSTATES longer-horizon risk because factor returns are positively autocorrelated (illiquidity, lead-lag)"
  - "Validation portfolios: random long-only, factor-tilted, optimized, and actual books; MRAD (mean rolling absolute deviation of B from 1) as the summary; volatility-regime adjustment re-levels the whole matrix when B drifts"
---

# Barra-style risk-model validation: bias statistics, Newey-West, test portfolios

The industry-standard scoring machinery for risk models used in portfolio
construction (not return prediction):

**Bias statistic.** For each test portfolio, standardize each period's realized
return by the model's *prior-period* vol forecast: `z_t = r_t / sigma_hat_{t-1}`.
`B = std(z_t)` over the window. Perfect calibration → B = 1. The 95% acceptance
band is `1 ± sqrt(2/T)` (normality approximation; fat tails widen the true band —
USE4 notes this makes the nominal band conservative in practice).

**Portfolio set.** One portfolio proves nothing. USE4-style validation runs B on:
factor portfolios, random long-only, random long/short, optimized (min-variance)
portfolios, and (in production) actual books — each stresses a different part of
Sigma. Random-book B tests the whole matrix; factor-portfolio B tests F; MVP B
tests the smallest eigenvalues, exactly where optimizers concentrate error.

**Summary statistics.** Mean bias per portfolio family + MRAD (mean rolling
absolute deviation |B−1|) to catch models that oscillate between over- and
under-forecast while averaging to 1.

**Newey-West horizon scaling.** Daily factor covariance scaled ×21 understates
21-day risk when factor returns autocorrelate positively (they do: illiquidity,
lead-lag). USE4 applies `F_NW = Γ_0 + Σ_q (1 − q/(Q+1))(Γ_q + Γ_q')` with
Bartlett weights (PSD-preserving), lags Q matched to the horizon overlap.
Direction of the omitted-adjustment bias: **vol targets set too high** (i.e.,
realized 21d risk exceeds forecast) for positively-autocorrelated factors — a
book run at a 10% target realizes more.

Feeds [[questions/research-risk-model-validation]].
