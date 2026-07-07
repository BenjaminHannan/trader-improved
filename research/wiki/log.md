# autoresearch log

## [2026-07-07] autoresearch (research-only) | data remediation (iteration 3/3)
- Rounds: 3 (vendor sweep → CM/Sharadar/Cleveland/adjustment gap fill → Binance
  delisted-retention check, unresolved)
- Sources found: Coin Metrics community docs, Norgate/Sharadar/Polygon pricing +
  delisted coverage, Cleveland Fed nowcasting page (403; EC 2023-06 confirms an
  internal vintage archive), FirstRate/PortfolioOptimizer/Alvarez adjustment forensics
- Pages created: [[sources/etf-adjustment-methodology]], [[questions/research-data-remediation]]
- Synthesis: [[questions/research-data-remediation]]
- Key finding: Coin Metrics community daily reference rates (free, full history,
  documented methodology, dead-asset coverage) for pre-2021 crypto; Norgate Platinum
  US$630/yr closes the 184-name delisted-equity gap AND adds a second PIT-membership
  source; Cleveland Fed vintages have no public archive — start a forward archiver
  now + request the EC-2023-06 vintage dataset; the FX/bond-ETF quarantine class is
  a vendor adjusted-close artifact — cross-check unadjusted closes + event tables.
  Backlog #16/#18 annotated. Sources went specific, not circular; stopping at 3
  iterations per program.

## [2026-07-07] autoresearch (research-only) | rejected-factor forensics (iteration 2/3)
- Rounds: 2 (regime record + canonical constructions; plus read-only code check of
  production/signals/{momentum,positioning,value}.py and alpha/ic.py)
- Sources found: SG/CFM/Kpler trend record; COT-instability studies; AMP 2013;
  MOP 2012; KRT JF 2020; hedging-pressure + industry-relative-value canon
- Pages created: [[sources/trend-cot-value-2024-2026-regime]],
  [[sources/canonical-factor-constructions]], [[questions/research-rejected-factor-forensics]]
- Synthesis: [[questions/research-rejected-factor-forensics]]
- Key finding: rejections replicate the industry record (SG Trend -12.5% rolling
  12m in-window; megacap-AI value headwind) — the gate measured reality. But three
  constructions genuinely diverge from canon: tsmom (cross-sectional sign() vs
  MOP time-series vol-scaled), cot_positioning (3y z of noncomm LEVEL conflates
  KRT's two opposite-signed premia), earnings_yield (no sector demean → static
  short-tech bet). mom_12_1's pooled t=0.85 is partly an unweighted-pooling
  artifact across sleeves with ~60x IC-variance differences. Three zero-trial-cost
  falsifiable checks filed; any re-specification is a new trial and stays deferred.

## [2026-07-07] autoresearch (research-only) | OOS gate design (iteration 1/3)
- Rounds: 3 (broad CPCV/HLZ/decay → primary-source fetch → sign-flip base rates)
- Sources found: 4 usable (Bailey CSCV, Arian-Norouzi-Seco 2024, McLean-Pontiff, Chen-Zimmermann); ScienceDirect + PDF fetches blocked, abstracts verified via search
- Pages created: [[sources/bailey-cscv-pbo-2015]], [[sources/arian-norouzi-seco-2024-oos-methods]], [[sources/oos-decay-sign-flip-base-rates]], [[questions/research-oos-gate-design]]
- Synthesis: [[questions/research-oos-gate-design]]
- Key finding: keep the 80/20 split as binding; tighten the criterion-2/4 boundary
  with purge=horizon + ~26d embargo; add a within-train CPCV sign-stability
  diagnostic (reject-only, zero trial cost). SE arithmetic: tonight's cot/tsmom
  flips are compatible with sampling noise (29-39% flip prob at N=8-30, h=20);
  earnings_yield's flip at N≈500 is >2 SE — construction/regime, not noise.

## [2026-07-06] edge-implementation goal | iterations 21-25 (all edges landed)
- IC multipliers: signal purification (planted IC 0.53->0.83), factor-momentum tilt
  (Ehsani-Linnainmaa), both ON for structural sleeves
- Breadth: rates/intl/sector ETF sleeves (4->7 tradable sleeves, 76 instruments,
  carry_curve factor), tranching K=5, crypto twice-weekly cadence, prediction-market
  events sleeve (8th, 10% risk cap) via Kalshi/Polymarket
- Foundations: EDGAR PIT fundamentals (earnings_yield + pead), passive limit orders,
  audit extension-dtype hardening
- Fourth wiring gap found and fixed: allocation.risk_caps was never applied
- Factor registry: 13 candidates; n_trials still 7 (bumps only at gate time)
- Suite: 402 tests green at close-out

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
