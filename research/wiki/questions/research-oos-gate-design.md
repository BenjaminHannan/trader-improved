---
type: synthesis
title: "Research: OOS gate design — single-split reliability, sign-flip base rates, CPCV diagnostic"
created: 2026-07-07
status: researched — recommendation filed, nothing implemented (research-only session)
related:
  - "[[sources/bailey-cscv-pbo-2015]]"
  - "[[sources/arian-norouzi-seco-2024-oos-methods]]"
  - "[[sources/oos-decay-sign-flip-base-rates]]"
---

# Research: is the single chronological 80/20 gate split reliable, and what would we change?

Context: first real-data gate run (2026-07-07) passed 2/12 factors. Several
rejections were train/OOS **sign flips** (cot_positioning train t=6.4 → OOS flip;
tsmom flip; earnings_yield t=-2.6 flip). Question: how reliable is one chronological
split as the OOS instrument, and what does a sign flip actually tell us?

## Key Findings

1. **A single hold-out split is the weakest of the standard OOS schemes.**
   Bailey et al. call hold-out "unreliable and inaccurate" for backtest selection
   ([[sources/bailey-cscv-pbo-2015]]); the only controlled comparison we found
   (synthetic ground truth: Heston/Merton/regime-switching) finds walk-forward has
   the worst false-discovery prevention and the highest temporal variability of
   estimates, because it scores one historical path; CPCV's C(S, S/2) purged
   recombinations give a distribution instead
   ([[sources/arian-norouzi-seco-2024-oos-methods]]).

2. **Sign-flip base rates favor keeping the same-sign criterion — but its power is
   cross-section-dependent.** Real published factors decay OOS (26% pre-publication,
   58% post-publication) yet overwhelmingly retain sign; spurious factors flip with
   p≈0.5 ([[sources/oos-decay-sign-flip-base-rates]]). So same-sign is a good
   likelihood-ratio test in principle.

3. **The SE arithmetic says our OOS window cannot distinguish flip-from-noise vs
   flip-from-overfit for small sleeves.** Per-date rank IC has std ≈ 1/√(N-1) under
   the null. OOS slice ≈ 520 days; with horizon-h overlapping labels, effective
   independent observations ≈ 520/h. For h=20:

   | Cross-section | per-date IC std | SE(mean OOS IC), T_eff≈26 | P(sign flip \| true IC=0.02) |
   |---|---|---|---|
   | N=8 (fx sleeve) | 0.378 | 0.074 | ~39% |
   | N≈30 (COT futures) | 0.186 | 0.037 | ~29% |
   | N≈500 (equities) | 0.045 | 0.009 | ~1% |

   Reading tonight's results through this table: **cot_positioning and tsmom flips
   are consistent with sampling noise even if the factors are real** (29-39% flip
   probability for a true-IC-0.02 factor); **earnings_yield's flip in a ~500-name
   cross-section is NOT explainable by noise** (>2 SE) — that one is construction
   or regime, to be diagnosed in iteration 2. Conversely carry_rate_diff's OOS IC
   of .077 at N=8 is itself only ~1 SE from zero — the pass is real per the gate
   but the point estimate deserves a wide error bar.

4. **Boundary leakage: purge + embargo at the train/validation boundary.** With
   overlapping h-day labels, the last h train days share forward-return windows
   with the first OOS days. AFML's prescription: purge = label horizon; embargo ≈
   1% of bars (≈26 days on a 2600-day panel) for serial-correlation leakage.
   Whether our split currently purges the boundary is an implementation question
   for the owning session — if it doesn't, measured OOS IC is slightly inflated
   for every factor.

5. **Harvey-Liu-Zhu (RFS 2016) context**: with hundreds of factors ever tried in
   the literature, credible new-factor t-stats should be ≳3.0, not 2.0. Our gate's
   t≥2.0 is the *train* screen, not the final claim — the deflated Sharpe at
   n_trials=18 is the HLZ-style correction and it currently (correctly) reports
   0.000. No change needed; this is the discipline working.

## Recommendation (falsifiable; no threshold loosened)

**Keep the single chronological 80/20 split as the binding gate** — repeated OOS
reuse burns trials and invites selection-on-OOS, the exact failure CSCV documents.
Three additive changes, in priority order:

1. **Tighten criterion 2/4's boundary** (the exact criterion changed: the
   *validation-slice start* used by criterion 2 "OOS same-sign, |IC| ≥ 0.005" and
   criterion 4 "net validation return > 0"): purge the last `horizon` train days
   and embargo the first ~26 calendar days (1% of panel) of the validation slice.
   Strictly non-loosening (can only reduce measured OOS IC).
2. **Add a within-train CPCV sign-stability diagnostic** (advisory, reject-only):
   S=8 blocks over the TRAIN span only → C(8,4)=70 purged+embargoed paths; report
   `frac_paths_same_sign_as_train`. Optional hard rule that can only ADD
   rejections: flag any factor with frac < 0.6 even if it passes criteria 1-4.
   Zero trial cost (never touches the OOS slice; can't rescue a fail).
3. **Report SE(OOS IC) next to the sign-flip verdict** using T_eff = T_oos/horizon
   and per-date IC std — so a rejection log entry distinguishes "flip > 2 SE
   (evidence of overfit/regime)" from "flip within 1 SE (uninformative cell)".
   Pure reporting; the rejections stand either way.

## Pre-registered predictions (testable by the implementing session)

- carry_rate_diff shows same-sign IC in ≥80% of within-train CPCV paths;
  cot_positioning and tsmom in <60% (train t-stats concentrated in one regime).
- Adding purge+embargo moves measured OOS IC by <1 SE for h≤20d factors; if any
  factor's OOS IC drops by more, the old split had material boundary leakage.
- earnings_yield's flip magnitude exceeds 2 SE at N≈500 — the noise explanation
  is ruled out and the diagnosis must come from construction/regime (iteration 2).

## Open Questions

- Bagged/adaptive CPCV variants (Arian et al.) — incremental; not worth complexity
  until plain CPCV diagnostic exists.
- Whether per-date IC std should be measured empirically per sleeve rather than
  1/√(N-1) (fat-tailed cross-sections push it higher; measurement is one line).

## Sources

- [[sources/bailey-cscv-pbo-2015]] — Bailey, Borwein, Lopez de Prado, Zhu
- [[sources/arian-norouzi-seco-2024-oos-methods]] — synthetic controlled comparison
- [[sources/oos-decay-sign-flip-base-rates]] — McLean-Pontiff 2016; Chen-Zimmermann 2022
- Harvey, Liu, Zhu 2016, "...and the Cross-Section of Expected Returns", RFS 29(1)
  (t≳3.0 for new factors under multiple testing; via NBER w20592 abstract)
