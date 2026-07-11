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
- [[questions/research-risk-model-validation]] — 2026-07-07 research-only (iteration 4):
  bias-stat + MVP-horse-race scoring harness spec; QIS not for our shapes; NW horizon
  check; per-sleeve-vs-global and exposure-set questions reduced to harness cells
- [[questions/research-kalshi-mechanism-diagnostics]] — 2026-07-10 (iteration 5): both
  pre-registered Kalshi tests negative (longshot fade fee-eaten; market beats nowcast);
  backfill + monthly nowcast vintages permanent; zero trials burned
- [[questions/research-political-underconfidence]] — 2026-07-11 (iteration 7): FIRST
  PASS of the loop — favorite-backing +4.8% post-fee t=2.9, robust to date clustering
  and liquidity floor; favorite-tilt signal landed in the events sleeve

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
- [[sources/use4-bias-statistics]] — Barra bias statistics, NW horizon scaling, validation portfolios
- [[sources/ledoit-wolf-nonlinear-shrinkage-guide]] — QIS/nonlinear shrinkage: where it pays (large N/T) and why not at N=8
- [[sources/practitioner-mechanism-scan-2026-07]] — 8 mechanism-backed niche ideas
  (prediction-market microstructure primary), pre-registered predictions, promotion
  protocol; top 2 need a Kalshi resolved-market backfill first
- [[sources/kalshi-historical-api]] — endpoint archaeology for the settled-market
  archive (external-api host, KX aliasing, trades-not-candles, fee formula)
- [[sources/burgi-deng-whelan-makers-takers]] — the longshot-fade anchor paper, exact
  MZ slopes by year/category recorded pre-replication

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
14. ~~Wire lake `cost_overrides` consumption into the engine's CostModel~~
    (2026-07-11: done — run_backtest(lake=...) reads and threads the table, loud
    warning on an active cost regime; floor_bps overrides clamped upward-only —
    a real hard-rule violation vector found and closed. FOLLOW-UP: the live path
    (daily_run / latest_target_weights) still does not consume overrides)
15. Accumulate per-run shortfall into the `shortfall_log` reference table from
    daily_run --live fills (feeds --calibrate-tca)
16. ~~Kalshi resolved-market historical backfill~~ (iteration 5, 2026-07-10: done —
    `--dataset kalshi_hist`, 71k rows; both pre-registered mechanism tests ran and
    FAILED, see [[questions/research-kalshi-mechanism-diagnostics]])
17. ~~Turn-of-year tax-loss-rebound diagnostic~~ (iteration 5, 2026-07-10: ran
    pre-registered — FAIL on the placebo-concentration gate despite +249bp t=2.26
    spread; near-miss details + re-spec observations in the 2026-07-10 log entry)
18. Cross-check follow-ups: CONSUMPTION HALF DONE (2026-07-11: run_backtest drops
    quarantined equities, keeps+warns the ETF adjustment-artifact class; 64 names
    on this machine's re-run). REMAINING: per-name verification of the quarantined
    equities (ticker-reuse -> blocklist extensions) and the unadjusted-close +
    event-table cross-check that should un-quarantine the FX/commodity-ETF class
    ([[sources/etf-adjustment-methodology]])
19. Events-sleeve maker-side execution: rest limit orders at model fair-value bands
    (extends edge-C1 passive execution to Kalshi) instead of crossing. Whelan caveat:
    makers ALSO lose ~10% on average — the maker seat needs the calibration model
    (now buildable, see #21) as the signal; seat alone is not an edge. Mandatory
    release-window pull rule (cancel resting quotes before scheduled prints —
    otherwise we supply the under-reaction edge to faster traders) + per-ladder
    inventory caps. Execution layer, not signal: does not touch n_trials
20. ~~Live `longshot_bias` events signal: category audit~~ (2026-07-11: documented
    re-spec landed — the signal now excludes the 12 measured US-macro Kalshi series
    (exact-match, venue-gated, fail-open); evidence block with all three diagnostic
    numbers lives in production/events/signals.py. Other categories unchanged —
    Whelan Table 8 concentrates the bias there)
21. ~~Events-sleeve calibration curve from RESOLVED outcomes~~ (2026-07-11: ran
    pre-registered — FAIL. Mid-range macro-ladder prices are already calibrated
    (0.594→0.600); all miscalibration is the fee-eaten ≤5c tail; OOS Brier
    p=0.41. Curve filed as diagnostic record (diagnostics/events_calibration.json).
    CONSEQUENCE for #19: the maker seat's validated-calibration prerequisite
    failed — #19 is demoted until a calibration input with real OOS lift exists)
