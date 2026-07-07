---
type: synthesis
title: "Research: rejected-factor forensics — regime break vs construction divergence, 2024-2026 OOS"
created: 2026-07-07
status: researched — diagnosis filed, nothing implemented (research-only session)
related:
  - "[[sources/trend-cot-value-2024-2026-regime]]"
  - "[[sources/canonical-factor-constructions]]"
  - "[[questions/research-oos-gate-design]]"
---

# Research: why did tsmom, cot_positioning, earnings_yield, mom_12_1, low_vol fail the gate?

Question: were tonight's rejections regime (2024-2026 OOS window hostile to these
families for everyone), construction (our signal diverges from the canonical one),
or noise (OOS window too weak to measure)? Verdict per factor:

## Per-factor diagnosis

| Factor | Gate result | Regime evidence | Construction divergence | Noise plausible? |
|---|---|---|---|---|
| tsmom | train flip → OOS flip | STRONG — SG Trend -9.3% YTD Apr-2025, 12m -12.5%; 2024 flat with Aug reversal | REAL — canonical TSMOM is time-series vol-scaled (MOP 2012), ours is a cross-sectional rank of a two-valued sign() at N≈8-16 | yes (29-39% flip prob) |
| cot_positioning | train t=6.4 → OOS flip | MODERATE — literature documents COT signal instability generally | REAL — 3y z of noncomm LEVEL mixes KRT's two opposite-signed premia (level = hedging premium; changes = liquidity provision) | yes (~29% at N≈30) |
| earnings_yield | train t=-2.6 → OOS flip | STRONG — megacap-AI rally 2024-25 (headwind), 2026 broadening (flip direction matches) | REAL — no sector demean → factor is substantially a static sector bet; practitioner canon is industry-relative value | no (>2 SE at N≈500) |
| mom_12_1 | pooled t=0.85 | mixed (equity momentum was fine 2023-25) | PARTIAL — per-sleeve rank IC is canonical, but the gate t pools per-date-per-sleeve ICs unweighted across sleeves with ~60x variance differences | pooling dilutes power |
| low_vol | t=10.1, net validation very negative | STRONG — shorting high-vol through a momentum/AI rally is the textbook low-vol failure mode | construction is standard; the gate's net-of-cost criterion did its job | n/a — gate worked |

## Key Findings

1. **The OOS window is independently documented as hostile to trend and value.**
   Professional trend followers (SG Trend constituents) lost money over most of
   the same window; the megacap-AI rally made non-sector-neutral value a
   short-tech bet ([[sources/trend-cot-value-2024-2026-regime]]). Our rejections
   replicate the industry record — which means the gate is measuring reality, not
   malfunctioning. It also means these OOS ICs are estimates of a *regime*, not
   of the factor's unconditional edge.

2. **Three of the five rejections carry a genuine construction divergence from
   the canonical form** ([[sources/canonical-factor-constructions]]):
   - tsmom: cross-sectional rank of sign() vs canonical per-instrument
     vol-scaled time-series bet;
   - cot_positioning: 3y z of the noncommercial *level* conflates KRT 2020's two
     opposite-signed premia (levels vs changes);
   - earnings_yield: no industry adjustment → dominated by a static sector tilt.
   These are diagnoses, NOT proposals for new factors. Re-specifying any of them
   and re-running the gate is a NEW TRIAL (n_trials increments) and is an owning-
   session decision, deferred until the idea inventory reopens.

3. **mom_12_1's t=0.85 is partly an aggregation artifact.** Per-date rank IC is
   correctly within-sleeve, but pooling IC observations unweighted across sleeves
   whose per-date IC variance differs by ~60x (1/(N-1): N=8 vs N≈500) lets the
   noisy small-N sleeves dominate the t-stat. Precision-weighted aggregation
   (weight per-date IC by N-1, or Stouffer-combine per-sleeve t-stats) is a
   *measurement correction*, not a threshold change and not a new factor — the
   same trial re-scored with a better estimator. Whether to treat re-scoring as a
   free measurement fix or a trial increment is a discipline question for the
   owning session; the conservative reading (charge a trial) is also defensible.

4. **low_vol is the gate working as designed** — a huge train t with strongly
   negative net validation is exactly the case criterion 4 exists for. No action.

## Falsifiable checks (zero trial cost, computable from existing gate artifacts)

- Sector-tilt attribution for earnings_yield: regress its per-date IC on a
  tech-minus-financials sector-return spread over the OOS window. Prediction:
  R² > 0.25 — most of the flip is the sector bet, not stock selection.
- Per-sleeve decomposition of mom_12_1's pooled t=0.85: prediction — equity-sleeve
  IC t ≥ 1.5 standalone, small-N sleeves near zero, and a precision-weighted
  pooled t materially above the unweighted 0.85 (measurement, not selection).
- cot_positioning: split the existing score's IC by z-component — level (156w
  mean-relative) vs 4w change. KRT predicts opposite signs on the two components
  in-train. If confirmed, the train t=6.4 was a lucky mix, and the OOS flip is
  expected behavior.

## Open Questions

- Whether the 2026 "broadening" (value recovery) persists — determines if
  earnings_yield's OOS-positive sign is the start of a regime or noise.
- SG Trend 2026-H1 exact YTD (not found in accessible sources; Q1 positive on
  energy per Kpler) — refresh when SG publishes.

## Sources

- [[sources/trend-cot-value-2024-2026-regime]] — SG/CFM/Kpler regime record
- [[sources/canonical-factor-constructions]] — AMP 2013, MOP 2012, KRT 2020,
  hedging-pressure and industry-relative-value canon
- [[questions/research-oos-gate-design]] — SE arithmetic used for the noise column
