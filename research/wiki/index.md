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
- [[questions/research-oos-gate-design]] — 2026-07-07 research-only: keep 80/20 split,
  tighten boundary (purge+embargo), add within-train CPCV sign-stability diagnostic
- [[questions/research-rejected-factor-forensics]] — 2026-07-07 research-only: per-factor
  regime-vs-construction-vs-noise diagnosis of tonight's 5 headline rejections
- [[questions/research-data-remediation]] — 2026-07-07 research-only: source/coverage/
  price/effort matrix for the four data gaps (crypto pre-2021, delisted equities,
  nowcast vintages, ETF adjustment normalization)

## Concepts
- [[concepts/ewma-shrinkage-combination]]

## Sources
- [[sources/ledoit-wolf-2003-honey]]
- [[sources/bailey-cscv-pbo-2015]] — CSCV/PBO: hold-out is the weakest OOS scheme
- [[sources/arian-norouzi-seco-2024-oos-methods]] — controlled comparison: CPCV > walk-forward
- [[sources/oos-decay-sign-flip-base-rates]] — McLean-Pontiff + Chen-Zimmermann decay/sign base rates
- [[sources/trend-cot-value-2024-2026-regime]] — SG Trend/CTA + value regime record for the OOS window
- [[sources/canonical-factor-constructions]] — AMP/MOP/KRT/industry-value canonical constructions vs ours
- [[sources/etf-adjustment-methodology]] — why cross-vendor adjusted closes disagree; the normalization fix
- [[sources/practitioner-mechanism-scan-2026-07]] — 8 mechanism-backed niche ideas
  (prediction-market microstructure primary), pre-registered predictions, promotion
  protocol; top 2 need a Kalshi resolved-market backfill first

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
16. Kalshi resolved-market historical backfill (free API: bucket prices + settlements,
    2023+) — unblocks the top-2 practitioner-scan mechanisms ([[sources/practitioner-mechanism-scan-2026-07]]:
    longshot-fade calibration test + nowcast-drift regression), both with
    pre-registered falsifiable predictions. Nowcast-vintage access options researched
    in [[questions/research-data-remediation]] (c): start forward archiver now, email
    Cleveland Fed for the EC-2023-06 vintage dataset for backfill
17. Turn-of-year tax-loss-rebound diagnostic (bottom-decile prior-year losers, last
    3 Dec days -> first 5 Jan days, incrementality vs plain reversal) — testable on
    the existing lake, no new ingest; n_trials-guarded if promoted
18. Cross-check follow-ups from first live run: per-name verification of the 66
    quarantined instruments (ticker-reuse class -> blocklist extensions; FX-ETF
    distribution-adjustment class -> vendor-methodology doc); wire quarantine list
    consumption into backtest universe filtering. Distribution-adjustment class now
    diagnosed in [[sources/etf-adjustment-methodology]]: cross-check UNADJUSTED
    closes + event tables, not vendor adjusted closes — most of that class should
    un-quarantine
19. Events-sleeve maker-side execution: rest limit orders at model fair-value bands
    (extends edge-C1 passive execution to Kalshi) instead of crossing. Whelan caveat:
    makers ALSO lose ~10% on average — the maker seat needs the calibration model
    (backlog #16) as the signal; seat alone is not an edge. Mandatory release-window
    pull rule (cancel resting quotes before scheduled prints — otherwise we supply
    the under-reaction edge to faster traders) + per-ladder inventory caps. Execution
    layer, not signal: does not touch n_trials
