# trader-improved research wiki

Autoresearch loop output: each iteration researches one candidate improvement, files
findings here, and (when the evidence survives) lands a verified implementation in
`production/`. `research/` is scratch — production never imports from it.

## Syntheses
- [[questions/research-covariance-shrinkage]] — iteration 1, implemented

## Concepts
- [[concepts/ewma-shrinkage-combination]]

## Sources
- [[sources/ledoit-wolf-2003-honey]]

## Backlog (prioritized frontier)
1. ~~Covariance shrinkage: LW analytic intensity + EWMA T_eff~~ (iteration 1)
2. Correlation-aware multi-factor alpha combination (Grinold multi-factor IC) —
   combine.py currently sums per-factor alphas, double-counting correlated scores
3. Vol-targeting overlay evidence (Moreira & Muir 2017) — parameterization + which
   sleeves benefit; drawdown-control interaction
4. Cost-model calibration: published ETF/crypto spread + sqrt-impact coefficients vs
   our floors (never lower them — validate they're not optimistic)
5. Turnover-aware alpha: Garleanu-Pedersen partial adjustment vs one-shot QP
6. Funding-carry factor evidence in crypto (basis/funding premia literature)
7. IC shrinkage n0 choice + factor-timing (avoid) literature
8. Winsorization/z-score robustness (MAD vs sigma-clip) cross-sectional evidence
9. Purge/embargo sizing for CPCV on overlapping-label panels
10. Deflated Sharpe var_trials estimation from the registry's actual gate history
