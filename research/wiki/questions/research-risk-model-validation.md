---
type: synthesis
title: "Research: risk-model scoring harness + upgrade priorities (iteration 4)"
created: 2026-07-07
status: researched — harness spec + per-question recommendations filed, nothing implemented (research-only session)
related:
  - "[[sources/use4-bias-statistics]]"
  - "[[sources/ledoit-wolf-nonlinear-shrinkage-guide]]"
  - "[[questions/research-covariance-shrinkage]]"
---

# Research: how to score the risk model once, so every future risk change is cheap

Priority answer first: the accepted way to validate a risk model for portfolio
construction is **bias statistics on families of test portfolios + a
minimum-variance realized-vol horse race** — not likelihood, not Frobenius error.
The harness below is the deliverable; Q2-Q5 each end in a falsifiable prediction
the harness can adjudicate.

## Q1 — The scoring harness (spec, to be implemented once by the owning session)

Machinery per [[sources/use4-bias-statistics]] (Menchero-Orr-Wang USE4) and the
Clarke-de Silva-Thorley (JPM 2006/2011) MVP tradition, with the Ledoit-Wolf
(J. Empirical Finance 2011) stationary-bootstrap variance test for adoption calls.

**Core loop.** For each evaluation date t (every 5 business days over the panel),
each sleeve, each test portfolio w: forecast `sigma_hat_t(h) = sqrt(h * w' Sigma_t w)`
at h = 21; realize `r_{t→t+h}`; standardize `z = r / sigma_hat`. Use
NON-overlapping h-windows so z_t are ~iid (T ≈ 2600/21 ≈ 120 obs per cell).

**Test-portfolio families** (each stresses a different part of Sigma):
1. random long-only (Dirichlet, ~100 draws/sleeve) — whole-matrix calibration;
2. random dollar-neutral L/S — correlation structure, no market-vol free ride;
3. equal-weight sleeve book — the aggregate;
4. **MVP** `w ∝ Sigma^{-1} 1` (long-only-constrained variant too) — smallest
   eigenvalues, where the QP concentrates error; ALSO the horse-race portfolio;
5. factor-tilted books (equity structural sleeve: unit-exposure pure-factor
   portfolios from the WLS regression) — tests F specifically;
6. cross-sleeve hedged books (e.g., long equity names / short matching sector
   ETF) — tests the sleeve-aggregation design (Q4).

**Metrics per (sleeve × family) cell:**
- Bias statistic `B = std(z)`; 95% band `1 ± sqrt(2/T)` → **[0.87, 1.13] at
  T=120**. B > 1.13 = under-forecast (dangerous), B < 0.87 = over-forecast.
- MRAD (mean rolling |B−1|, 12-obs windows) — catches oscillating miscalibration.
- MVP horse race: annualized realized vol of family-4 books, candidate vs
  incumbent vs raw-sample-cov reference.

**Pass criteria for ADOPTING any future risk-model change:**
1. no new calibration failures: every cell inside the band under the incumbent
   stays inside under the candidate;
2. |B−1| improves (weakly) in ≥ 60% of cells;
3. MVP realized vol: candidate ≤ incumbent with LW-2011 stationary-bootstrap
   p < 0.05 for a claimed improvement (p ≥ 0.05 → adopt only if strictly simpler);
4. corruption harness (PIT) still green.

**Baseline health check for the CURRENT model** (run once at harness birth):
≥ 90% of random-book cells inside [0.87, 1.13]. Failures localize the problem:
family-2/6 failures → correlations; family-4 → eigenvalue floor / shrinkage;
family-5 → F; uniform B > 1 at h=21 with B ≈ 1 at h=1 → the Q3 horizon effect.

Risk-model changes carry **zero n_trials cost** (no alpha selection touched), but
log every harness run like a gate verdict — same discipline, separate ledger.

## Q2 — Nonlinear shrinkage (QIS): not for our current shapes

Mechanism-level answer in [[sources/ledoit-wolf-nonlinear-shrinkage-guide]]:
nonlinear shrinkage is rotation-equivariant (adjusts eigenvalues only, injects no
cross-sectional prior); its edge over linear grows with c = N/T and vanishes as
c → 0. Our small sleeves (N=8-25, c ≈ 0.06-0.2) get more from the
constant-correlation target's pooling than any eigenvalue de-noising can give;
our factor cov (K≈15, c ≈ 0.01) is effectively asymptotic already. QIS becomes
relevant only if an instrument-level equity covariance (N≈500) is ever built.
**Falsifiable in-repo (no web claims needed): Monte Carlo with a true one-factor
DGP at N=8, T=130 — prediction: lw_cc beats QIS on minimum-variance loss.
At N=500, T=1250 the ranking flips.** Recommendation: no change; close the
iteration-1 open question.

## Q3 — Horizon mismatch (daily cov → 21d hold): the one likely live bias

Factor returns and sleeve indexes are positively autocorrelated (illiquidity,
lead-lag), so `21 × daily Sigma` UNDERSTATES 21-day risk — the book runs hotter
than its vol target. Industry fix (USE4): Newey-West with Bartlett weights,
`F_NW = Γ_0 + Σ_q (1 − q/(Q+1))(Γ_q + Γ_q')`, lags Q matched to the horizon,
applied to factor covariance (and, for us, the sleeve-return EWMA cov in
`allocation.py`) before horizon scaling. Magnitude is factor-dependent — measure,
don't import: **falsifiable pre-check: variance ratio VR(21) =
Var(21d sums)/(21 × Var(daily)) per estimated factor-return series and per sleeve
index. Prediction: VR(21) > 1 for momentum and the small-N sleeve indexes;
if VR ≈ 1 across the board, do NOT build the adjustment** (it would add noise
via Γ_q estimation for nothing). The harness detects the same thing as
family-agnostic B > 1 at h=21: harness first, then this diagnosis names the fix.

## Q4 — Per-sleeve + sleeve-ERC vs one global model: test, don't rebuild

Our design does capture cross-sleeve correlation — at sleeve-aggregate
resolution (EWMA cov of sleeve returns in `allocation.py`). What it cannot see is
instrument-level cross-sleeve netting: a book long equity names and short the
matching sector ETF has an effective correlation to its hedge that differs from
the equity-sleeve↔sector_etf-sleeve aggregate (~0.8+). The integrated-model
literature (Barra BIM-style `F = Y G Y' + Φ`, global factors linking asset-class
blocks) exists precisely for this. But a global rebuild is expensive and our
books are sleeve-allocated, not cross-sleeve-hedged — the loss may be immaterial
for how we trade. **Falsifiable prediction for the harness (family 6): |B−1| on
cross-sleeve hedged books exceeds |B−1| on within-sleeve books, in the
over-forecast direction (B < 1: model misses netting). If family-6 B stays
inside [0.87, 1.13], the per-sleeve design is adequate and the global model is
not worth building.** Recommendation: run the test before any redesign; HRP-style
approaches are an allocator, not a risk model, and are out of scope here (Phase-7
HRP evidence in the reference repo was negative anyway).

## Q5 — Equity exposure set: two computable additions, both risk-only

USE4 style set: Beta, Momentum, Size, Nonlinear Size, Residual Vol, Growth,
Earnings Yield, Book-to-Price, Dividend Yield, Leverage, Liquidity. We carry
market/size/momentum/vol + GICS dummies. From our lake (prices + eps/revenue/
shares), the computable missing ones are:
- **Liquidity** (turnover: ADV / shares outstanding) — among the highest
  explanatory-power style factors in USE4's own empirical notes;
- **Earnings Yield** (already built as a rejected *alpha* signal; reusing it as a
  *risk exposure* touches no gate and no n_trials — different job);
- Growth (revenue YoY) — weaker risk factor, optional third.
NOT computable now: Book-to-Price, Leverage, Dividend Yield (need balance-sheet /
distribution data). **Falsifiable prediction: adding liquidity + earnings-yield
exposures moves family-5 and E/P-tilted family-1 bias stats toward 1 and raises
average cross-sectional R² of the factor regressions; if R² doesn't move ≥ 1pp,
the sector dummies were already spanning the value effect and the exposures
should be dropped.** Caveat from tonight's live run: factor-return quality
depends on the hygiene layer (TIE-class corruption distorted factor returns) —
re-run the harness after any lake remediation.

## What the harness adjudicates (the point of this iteration)

| Question | Harness cell that decides it | Decision rule |
|---|---|---|
| Q3 NW horizon | any family, B at h=21 vs h=1 + VR(21) pre-check | build NW only if B>1.13 pattern & VR>1 |
| Q4 global model | family 6 (cross-sleeve hedged) | rebuild only if B < 0.87 there |
| Q5 exposures | family 5 + E/P-tilted family 1 | keep only if B→1 and R² +1pp |
| Q2 QIS | (in-repo Monte Carlo, not harness) | expected: confirm no-change |
| any future risk change | all cells | pass criteria 1-4 above |

## Open Questions

- Exact NW lag count Q for h=21 with EWMA weighting (USE4 uses horizon-matched
  lags; interaction with EWMA halflife needs a small simulation).
- Whether crypto's structural model has enough factor-return history yet for
  family-5 tests (min_obs=252 gate in model.py).
- Direct PDFs (USE4 notes, Goldilocks, QIS) remain fetch-blocked — formulas were
  cross-confirmed via secondary sources; re-verify on an open network.

## Sources

- [[sources/use4-bias-statistics]] — Menchero-Orr-Wang USE4 notes + assembly guide
- [[sources/ledoit-wolf-nonlinear-shrinkage-guide]] — Goldilocks 2017, QIS 2022, Review & Guide 2020
- Clarke, de Silva & Thorley, "Minimum-Variance Portfolios in the U.S. Equity
  Market" (JPM 2006) and "Minimum-Variance Portfolio Composition" (JPM 2011) —
  the MVP horse-race tradition
- Ledoit & Wolf, "Robust Performance Hypothesis Testing with the Variance"
  (J. Empirical Finance 2011) — stationary-bootstrap equal-variance test
- Engle, Ledoit & Wolf, "Large Dynamic Covariance Matrices" (JBES 2019) — DCC-NL,
  the GMV-vol-as-loss evaluation pattern
