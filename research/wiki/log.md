# autoresearch log

## [2026-07-06] autoresearch | iterations 2-17 batch summary
- Implemented: Grinold combination (2), overlay smoothing+deadband (3), cost
  ledger/sensitivity (4), DSR trial variance w/ unit fix (5), PW block length (7),
  sector threading + attribution guard (9), vendor cross-check + feed priority via
  availability stamps (12), drawdown ramp+hysteresis (13), VIX term structure (14),
  hygiene enforcement (15), asof_panel vintage dedup (16), basis factor in flight (17)
- Negative/confirmed: GP partial adjustment (6), zscore design (8), funding carry
  direction (10), IC n0 deferred (11)
- Meta-lesson (three wiring gaps found): audit CALL SITES for every hard rule, not
  just unit-tested existence.
- Checkpoint: 304 tests green at iteration 16.

## [2026-07-06] autoresearch | covariance shrinkage upgrade (iteration 1)
- Rounds: 2 (broad + EWMA-combination gap fill)
- Sources found: 3 usable at search level; ALL direct fetches 403 (sandbox proxy) — logged per failure rule
- Pages created: [[sources/ledoit-wolf-2003-honey]], [[concepts/ewma-shrinkage-combination]], [[questions/research-covariance-shrinkage]]
- Synthesis: [[questions/research-covariance-shrinkage]]
- Key finding: replace hand-set shrink 0.3 with LW analytic intensity (diagonal target for factor cov, constant-correlation for small-sleeve cov), using Kish effective sample size under EWMA weights; verified by Monte Carlo in-repo.
