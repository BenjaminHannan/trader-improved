# autoresearch log

## [2026-07-06] autoresearch | covariance shrinkage upgrade (iteration 1)
- Rounds: 2 (broad + EWMA-combination gap fill)
- Sources found: 3 usable at search level; ALL direct fetches 403 (sandbox proxy) — logged per failure rule
- Pages created: [[sources/ledoit-wolf-2003-honey]], [[concepts/ewma-shrinkage-combination]], [[questions/research-covariance-shrinkage]]
- Synthesis: [[questions/research-covariance-shrinkage]]
- Key finding: replace hand-set shrink 0.3 with LW analytic intensity (diagonal target for factor cov, constant-correlation for small-sleeve cov), using Kish effective sample size under EWMA weights; verified by Monte Carlo in-repo.
