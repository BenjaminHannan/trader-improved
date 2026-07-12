---
type: synthesis
title: "Research: macro de-risk indicator — VIX term structure (iteration 14)"
created: 2026-07-06
status: implemented
---

# Research: VIX level vs term structure for the macro de-risk overlay

## Key Findings
- The VIX **term-structure slope** (VIX/VIX3M ratio; backwardation = stress) is the
  robust regime indicator; the spot VIX **level** is much weaker — backwardation
  predicts (confirms) stress phases while contango carries no significant timing
  signal ("markets stay complacent longer than they can panic").
- Extreme VIX-level spikes are, if anything, CONTRARIAN over 3m horizons — meaning a
  pure level-z de-risk trigger can cut exposure exactly when forward returns are
  best. Term-structure inversion does not share this pathology at regime scale.
- Composite indicators (vol term structure + credit spreads) beat single series by
  combining orthogonal channels — equity-vol stress and credit stress can diverge.
- HY OAS (already in our overlay) remains the standard credit channel.

## Decision → implementation
- FRED loader alias list gains VXVCLS (CBOE 3-month vol index).
- macro_derisk_multiplier gains `ratio_pairs` (config: [[VIXCLS, VXVCLS]]): derive
  the trailing ratio series PIT (both legs available_from <= t), causal z exactly
  like level series, and include it in the composite z. VIXCLS drops OUT of the
  level-series list (replaced by the ratio — the level's contrarian pathology) while
  BAMLH0A0HYM2 stays as the credit channel.
- Back-compat: absent ratio_pairs -> behavior unchanged.

## Verification (in-repo)
Ratio construction known-answer; PIT corruption (future macro rows don't move the
multiplier at t); planted backwardation episode (VIX above VIX3M for a stretch)
triggers the scale while a same-magnitude parallel level shift (ratio unchanged)
does NOT — the discriminating test between ratio and level triggers.

## Sources
- Macrosynergy, "VIX term structure as a trading signal"; MDPI JRFM 12(3):113 "VIX
  Futures as a Market Timing Indicator"; CBOE Inside Volatility Trading; Preprints
  202602.1048 (VIX spikes contrarian, robust to credit-spread controls) — search-level
