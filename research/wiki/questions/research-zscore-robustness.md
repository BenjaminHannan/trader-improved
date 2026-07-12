---
type: synthesis
title: "Research: cross-sectional score normalization (iteration 8)"
created: 2026-07-06
status: researched — design CONFIRMED, no change
---

# Research: winsorize+z vs rank transforms

## Key Findings
- Winsorize-then-z is the documented practice of index providers (values winsorized
  first, z-scores computed on the winsorized values) — exactly what
  alpha/zscore.py does (±3 MAD, then z).
- Rank transforms are robust but distort cross-factor correlations and distances —
  they would corrupt the score-correlation matrix the Grinold combination
  (iteration 2) now depends on. Avoid.
- MAD-based clipping is preferred over sigma-based when the raw cross-section is
  heavy-tailed (sigma is itself inflated by the outliers being clipped).

## Verdict
Current design confirmed by the literature; no change. Filed to prevent re-testing
the same question later (the wiki is also a record of what NOT to redo).

## Sources
- Quantdare "Scaling/normalisation/standardisation" ; SAS "Winsorization: good, bad,
  ugly" ; Wikipedia Winsorizing — search-level
