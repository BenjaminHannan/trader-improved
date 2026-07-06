# trader-improved research wiki

Autoresearch loop output: each iteration researches one candidate improvement, files
findings here, and (when the evidence survives) lands a verified implementation in
`production/`. `research/` is scratch — production never imports from it.

## Syntheses
- [[questions/research-covariance-shrinkage]] — iteration 1, implemented
- [[questions/research-alpha-combination]] — iteration 2, implemented
- [[questions/research-vol-targeting-overlay]] — iteration 3, implemented
- [[questions/research-cost-model-calibration]] — iteration 4, implemented (reporting)
- [[questions/research-dsr-var-trials]] — iteration 5, implemented
- [[questions/research-garleanu-pedersen]] — iteration 6, NOT adopted (negative result)
- [[questions/research-bootstrap-block-length]] — iteration 7, implemented
- [[questions/research-zscore-robustness]] — iteration 8, design confirmed
- [[questions/research-sector-threading-gap]] — iteration 9 (internal audit), implemented
- [[questions/research-crypto-carry]] — iteration 10, confirmed + backlog add

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
10. ~~Deflated Sharpe var_trials estimation from the registry's actual gate history~~ (iteration 5)
11. Crypto basis factor: ccxt perp-mark-price loader + registered signal (strongest
    documented cross-sectional crypto predictor — see iteration 10)
12. GP persistence weighting — revisit once live IC decay curves exist (iteration 6)
13. ~~Robust-alpha ellipsoid in the QP~~ (iteration 19, opt-in via robust_kappa)
14. Wire lake `cost_overrides` consumption into the engine's CostModel construction
    (TCA calibration writes the table — iteration 20 — but run_backtest doesn't read
    it yet; thread overrides_table through to the CostModel call)
15. Accumulate per-run shortfall into the `shortfall_log` reference table from
    daily_run --live fills (feeds --calibrate-tca)
