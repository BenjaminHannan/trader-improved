# autoresearch log

## [2026-07-07] autoresearch (research-only) | risk-model validation harness (iteration 4)
- Rounds: 3 (bias-stat/MVP/QIS broad → USE4+Goldilocks fetch attempts (PDFs blocked)
  → NW-horizon + Barra factor list + hierarchical gap fill; itsjustbeta.com assembly
  chapter supplied the formulas verbatim)
- Sources found: USE4 Methodology/Empirical Notes (Menchero-Orr-Wang), LW Goldilocks
  RFS 2017 + QIS Bernoulli 2022 + Review-and-Guide 2020, CDT JPM 2006/2011 MVP
  tradition, LW 2011 bootstrap variance test, Engle-Ledoit-Wolf DCC-NL
- Pages created: [[sources/use4-bias-statistics]],
  [[sources/ledoit-wolf-nonlinear-shrinkage-guide]], [[questions/research-risk-model-validation]]
- Synthesis: [[questions/research-risk-model-validation]]
- Key finding: the deliverable is the harness, not any single model change — bias
  statistic B = std(r/sigma_hat) per (sleeve × portfolio-family) cell with band
  1±sqrt(2/T) (=[0.87,1.13] at T=120 non-overlapping 21d windows) + MVP realized-vol
  horse race with LW-2011 bootstrap p<0.05 for adoption. QIS is mechanically wrong
  at N=8 (rotation-equivariant, no cross-sectional prior — lw_cc's pooling wins);
  the likely live bias is horizon scaling (daily→21d understates risk under positive
  autocorrelation; NW/Bartlett is the fix, but only if VR(21)>1 in our own factor
  returns); per-sleeve-vs-global and exposure-set questions are reduced to specific
  harness cells with pre-registered pass/fail directions. Zero n_trials impact.

## [2026-07-07] FIRST LIVE-DATA RUN — full-history ingest, first real gate, first real backtest

The gate and backtest ran on real vendor data for the first time. Headline: 2 of 12
factors accepted (carry_rate_diff fx_etf, basis_carry crypto); backtest of the
surviving book: net Sharpe +0.022, **deflated Sharpe 0.000 at n_trials=20**, 95% CI
[-0.60, +0.62] — an honest near-empty book, no fabricated edge. French validation
after data remediation: market↔Mkt-RF **0.879**, momentum↔Mom 0.666 (size↔SMB -0.27
is the expected large-cap-universe artifact). Report artifact:
reports/backtest_20260707T094456Z.json.

### Ingested (final lake state, 2016-01-01 → 2026-07-07)
| dataset | rows | sources |
|---|---|---|
| prices | 3,027,222 | yfinance 1.77M (primary), alpaca 1.07M, tiingo 132k (secondaries), ccxt 55k (crypto) |
| macro | 110,290 | ADS vintages 72.7k, FRED/ALFRED 21.5k, ECB FX 16.1k |
| fundamentals | 106,425 | SEC companyfacts, PIT filing vintages (630 ids; eps/shares/revenue) |
| funding | 40,718 | bybit/okx paginated, 2020-06+ (MATIC/MKR unserved) |
| basis | 33,567 | perp-vs-spot 2021-07+ (perps barely existed cross-sectionally before) |
| french | 13,080 | Ken French daily zips |
| cot | 5,803 | CFTC Socrata, server-side market filter |
| crypto_meta / defi_tvl | 25 / 455 | snapshots — accrual starts tonight |

Universe: 950 instruments (874 PIT equities from the Wikipedia walk-back, GICS
sectors threaded), 905 equity membership intervals, 7 sleeve membership tables.
Survivorship caveat: 184/874 delisted equities have no vendor prices (see
[[questions/research-data-remediation]] — Norgate is the recommended closer).

### Gate verdicts (n_trials 7 → 20 across two rounds)
- **PASS carry_rate_diff** (fx_etf): train IC .040 t=3.92, OOS IC .077 same-sign,
  no decay past horizon; re-passed after data remediation.
- **PASS basis_carry** (crypto): negative-IC crowding signal, t=-2.01, OOS -.027;
  re-passed after remediation.
- FAIL carry_funding: honest flip — corrected weekly-cadence economics stopped
  inflating gross ~5x; the 30bp crypto floor eats its L/S validation return.
- FAIL cot_positioning (train t=6.4 → OOS sign flip), tsmom (flip), earnings_yield
  (t=-2.6, flip), mom_12_1 (pooled t=0.85), str_reversal, carry_curve, low_vol
  (t=10.1 but deeply negative net validation), pead (t=1.37).
  Post-hoc forensics ([[questions/research-rejected-factor-forensics]], claims
  code-verified): tsmom/cot/earnings_yield carry real construction divergences from
  canon; cot/tsmom flips are within noise at their N; earnings_yield's flip is >2 SE
  (sector-bet explanation). Rejections stand; any re-specification is a new trial.
- SKIP mcap_tvl (TVL snapshots have 1 day of history; no trial burned).
- Equity-factor caveat: rejections were measured on a panel later found to contain
  2 corrupt vendor series (below). t-stat rejections are robust to 2 names in ~600;
  low_vol's net-validation magnitude was TIE-distorted — a clean re-measure is a
  deliberate next-session decision with its trial cost.

### Gate-measurement bugs found by first data contact (fixed, thresholds untouched)
1. decay_halflife anchored at h=1 misread slow carry (IC -0.05 at h=1 rising to
   +0.05 at h=42) as "decayed in 1.6d" — now anchors at the factor's own horizon.
2. _net_validation_return iterated every date: ~horizon-times overlapping forward
   returns + daily churn costs (a 500-name book "returned" -166/2y). Now rebalances
   on the horizon grid as documented.

### The TIE incident — corrupt vendor series and the new hygiene layer
French validation first returned market↔Mkt-RF = 0.056; diagnosis found dead-ticker
vendor series alternating between price regimes in multi-day runs (EQ:TIE $2 ↔
$15,600, fake +810,000% returns; CFC, MI, BMC similar) — an equal-weight market off
this panel had 1,750% ann vol. Cross-check could not catch them (no second vendor
covers dead tickers → insufficient overlap). Fixes, all regression-tested:
- hygiene.apply_flap_screen — row-level rolling-median screen (spares GME/NKTR).
- hygiene.drop_corrupt_series — series-level: >5 catastrophic (>400%) day moves =
  wrong-entity series, dropped wholesale. This is what run-alternators require.
- Lake remediation with audit records (2,330 rows incl. TIE/BMC series); equal-weight
  market vs Mkt-RF verified 0.92 after.
Lesson recorded: row-level screens cannot fix entity-level corruption, and the
French-benchmark check is the instrument that catches what cross-vendor checks
structurally cannot.

### Cross-check first light (yfinance vs alpaca+tiingo)
697 instruments compared, 66 quarantined: ticker-reuse class (APC, MI — corr ~0,
the exact class the synthetic-id schema exists for), FX-ETF distribution-adjustment
class (month-end-clustered divergences, see [[sources/etf-adjustment-methodology]]),
and squeeze-day microstructure noise (GME 2021, corr 0.9995 otherwise). Audit at
data/audit/cross_check_prices.json; consumption of the quarantine list by the
backtest universe is backlog #18.

### Loader/orchestration fixes landed tonight (each with a canned-payload test)
universe build (S&P walk-back never wired → 874 equities minted, loud-fail on empty),
prices fan-out across 7 sleeves, equity symbols from the master, ccxt OHLCV
pagination + kraken 720-bar depth fallback (crypto 2016+), perp-basis pagination,
funding pagination + pre-listing probing + venue pacing (IP bans are real), ALFRED
vintage-window chunking, COT server-side filter + Socrata pagination, EDGAR
master-derived symbols + Q4-vs-FY duration dedup + same-day-amendment dedup,
parquet raw-zone row-per-key + large_string (2.6 GB companyfacts) + degradation
guard, ADS colon-dates, signal-bundle unification (basis/fundamentals/mcap/tvl),
Tiingo + Alpaca secondary loaders (stooq is dead vendor-side:
diagnostics/stooq_verdict.md), coingecko/defillama wiring.

### Vendor status
DEAD/BLOCKED: stooq (JS wall), NAAIM (JS page), FINRA (OAuth), AAII (paid file),
CBOE put/call (commercialized S3). DEGRADED: kraken funding (unsupported),
MATIC/MKR funding (unserved), tiingo free tier (~50 symbols/hr — ETF sleeves have
2016+ depth, equities accrue). Alpaca IEX history starts ~2021.

### Open items filed
backlog #16-19 (Kalshi backfill, TOY diagnostic, quarantine follow-ups, maker-side
events execution), validate_against_french series-id + monthly-input wiring,
comment-preserving registry writer (ruamel), stage-2 shared-watermark design wrinkle,
lake-level purge for rows a re-ingest no longer emits. Research-only session ran in
parallel and filed [[questions/research-oos-gate-design]],
[[questions/research-rejected-factor-forensics]], [[questions/research-data-remediation]].

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
