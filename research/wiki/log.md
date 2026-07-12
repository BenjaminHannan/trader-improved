# autoresearch log

## [2026-07-12] Factor-health rule + basis_carry demotion round

HONEST FRAMING: this is a governance DECISION on observed OOS degradation,
not a pre-registered experiment — the trigger evidence (basis_carry IC
sign-flip: 2024/25/26 = +0.018/+0.019/+0.040 vs the accepted NEGATIVE sign,
2026 t=+2.5, full-sample ≈0) is already on the record above. What is being
committed BEFORE implementation is the RULE, so future demotions are
mechanical, symmetric, and not cherry-picked:

- **Demotion rule (all accepted factors, applied at every health round):**
  a factor is demoted (new registry status `demoted`: excluded from the
  traded book — the engine trades `accepted` only — history retained) when
  its trailing-24-month rank-IC, sign-adjusted to the accepted direction,
  has month-clustered t ≤ 0. Two years of wrong-or-no sign = the live edge
  is gone or reversed.
- **Symmetry:** the rule is measured for EVERY accepted factor in the same
  round — today that is basis_carry AND carry_rate_diff. No factor is
  singled out.
- **Re-admission = a NEW trial** (n_trials increments) with a fresh
  pre-registration; demotion itself burns nothing.
- Measurement spec: daily cross-sectional rank-IC of the production signal
  vs forward returns at the factor's rebalance horizon, trailing 24 months
  from the latest obs, clusters = calendar month, vendor-deduped panel.

VERDICT (`factor_health_20260712T032904Z.json`, machinery 109f79e):
**NO-DEMOTE for either factor — the rule held against my expectation.**
basis_carry sign-adjusted trailing-24m IC +0.0497 (month-t +2.36, 726d);
carry_rate_diff +0.111 (t +1.55, 481d). The demotion I anticipated did not
survive mechanical measurement, for two documented reasons: (1) SIGN
FORENSICS — the regime-flip narrative above was computed on RAW basis vs
forward returns; the traded signal is the negated/smoothed construction,
and adjusting by the registry's recorded accepted sign the edge is intact;
(2) the crypto sleeve's backtest loss (−0.177 over 7.5y ⇒ t≈−0.5) is
statistically weak evidence of death. OPEN QUESTIONS filed, not acted on:
a fresh gate ic-report on the deep lake shows basis_carry train +0.013 /
OOS −0.083 (sign records disagree across shallow-lake gate_stats, raw-basis
forensic, and deep-lake re-run — a sign-convention audit is warranted);
and a P&L-based health rule (trailing factor-attributed net Sharpe, which
measures what the trailing-IC-weighted engine ACTUALLY trades) is a
candidate SECOND rule for a future round — if added, BOTH rules get
reported at every round regardless of which fires (no rule-shopping).

## [2026-07-11] Iteration 9 PRE-REGISTRATION — Kalshi post-move drift (written before any data contact)

Motivation: the honest headline is zero; the alpha frontier per the burned-trial
record (mom/tsmom re-specs failed gate v2; earnings_yield/COT re-spec mechanisms
refuted at step-0) points at the one domain with a measured live edge — Kalshi.
Practitioner-scan candidate #3 (Angelini-De Angelis, arXiv 2606.07811: β=0.63
post-move continuation, sports only): does binary-market under-reaction drift
exist in OUR panel (politics/econ, 2021-2026)?

Diagnostic (zero trial cost; a trial is funded ONLY if all three pass):
- Sample: `event_markets_hist` (kalshi) market-days with **≥5 days to close**
  (excludes mechanical settlement convergence), pre-move `yes_price` in
  [0.10, 0.90] (avoids boundary compression), settled markets only.
- Trigger: |1-day Δ yes_price| ≥ 0.05. Direction = sign of the move.
- Response: signed continuation over the next 3 obs days (Δp in move direction).
- Q1 (existence): mean signed 3d continuation ≥ +1.0c, cluster-robust t ≥ 2.0
  (clusters = event_key), n ≥ 1,000 triggers.
- Q2 (economics): net edge > 0 after the Whelan taker curve charged TWICE
  (entry at post-move price, exit 3d later — a drift trade round-trips, unlike
  the hold-to-settlement tilt).
- Q3 (capacity/liquidity, the documented Angelini caveat): effect present with
  t ≥ 1.5 in the above-median-volume half on its own.
- Any fail → file the negative, no trial burned. All pass → fund ONE trial
  (n_trials 24 → 25) for a drift signal whose spec (trigger size, holding
  days, price band) is fixed to THESE pre-registered values — no post-hoc bin
  shopping; different parameters = a different (new) trial.
- Guard: the diagnostic runs only AFTER the in-flight category remediation of
  `event_markets_hist` lands (do not read a half-written vintage).

VERDICT (run after remediation, n=18,090 triggers / 4,062 markets / 1,207
event clusters): **FAIL on all three — and the effect is REVERSED.** Mean
signed 3d continuation **−1.82c, t=−12.75**; net of fees −4.27c (t=−29.6);
liquid half −1.25c (t=−7.2). Kalshi politics/econ markets MEAN-REVERT after
large 1-day moves — over-reaction, not the sports under-reaction (Angelini
β=0.63 does not transfer). Consistent −2c/trigger every year 2023-2026. No
trial burned. The reversal flip side is fee-eaten as a naive taker round-trip
(fade gross +1.82c vs ~2.45c average double-taker fee ⇒ ~−0.6c net) — same
shape as the longshot fade (#iteration-5): real anomaly, taker fees eat it.
FILED as a future hypothesis, not pursued: a MAKER-side fade (limit orders
inside the post-move spread) would capture the reversion without taker fees,
but requires fill-probability modeling we have no data for.

## [2026-07-11] Iteration 8 — Q5 exposures adjudication: NO-ADOPT (0.86pp vs 1.00pp)

Full pre-registered run (equity, 574 ids, 112 eval dates, walk 2021-2026,
artifact `adjudication_q5_exposures_20260711.json`; machinery mirror-checked
vs production estimate_factor_returns, max|diff|=0). Candidate = liquidity +
earnings_yield equity risk exposures (config-flip, alpha gate untouched).
- Criterion (a) PASSED emphatically: pure-factor books snap into band —
  f5_earnings_yield B **1.696 → 1.001**, f5_liquidity **1.494 → 1.053**,
  E/P-tilted f1 0.920 → 0.920 (in band both). The model's risk forecasts for
  DELIBERATE value/liquidity tilts are badly hot without these exposures and
  essentially perfect with them.
- Criterion (b) FAILED by 0.14pp: mean cross-sectional R² 0.2145 → 0.2231,
  **+0.86pp vs the ≥1.00pp bar** (2,484 regression days). This is precisely
  the pre-written alternative: sector dummies already span the value effect
  on AVERAGE days.
- **NO-ADOPT per the registration.** The conjunction was the prediction; half
  a prediction is a fail. Filed for a future round (NEW registration, not a
  tweak): (i) adopt if equity family-4 MVP B (2.185, THE structural failure)
  moves materially toward band under the candidate — the residual-co-movement
  mechanism Q5 was aimed at was never directly scored; (ii) or an R² criterion
  measured on factor-tilted days/books rather than the unconditional mean.

PRE-REGISTERED (iteration 8b, committed before running — path (i) above made
binding): same candidate (liquidity + earnings_yield equity exposures),
re-adjudicated against its mechanism target, the equity MVP cell. ADOPT iff
ALL of: (1) equity family-4 |B_cand − 1| ≤ 0.5·|B_inc − 1| (B_inc = 2.185 from
ledger 211638Z ⇒ B_cand must land in [0.408, 1.593]); (2) equity f1-f3 stay
in band under the candidate; (3) equity MVP horse race (step=5, min_obs=252,
LW-2011 paired bootstrap): candidate realized vol LOWER with p < 0.05 —
candidate is NOT simpler, no leniency; (4) incumbent cells reused verbatim
from ledger 211638Z (n=100/seed=0/min_obs=252 — identical panel args).
NO-ADOPT if any fails; any threshold moved after seeing results = a new
registration. Zero n_trials cost (risk-model change, alpha gate untouched).

**A second corruption class the MI-incident fixes could not see.** After the
name-driven blocklist purge (KG/MI/SBNY/CHK/MNK) the fresh French check still failed:
market↔Mkt-RF **0.691** (below the healthy 0.88–0.92), momentum↔Mom **0.106**
(gate 0.6). A pattern-driven forensic scan of the whole equity lake found the residue
the blocklist never named — and, crucially, **every existing hygiene guard is
volume-blind**, so two survivors slipped all three:

- **EQ:SLE (Sara Lee, renamed 2012)** — a dead single-vendor series that *drifts* to
  $97,344 at **~12 shares/day** (75% of days < 100 shares). It makes ZERO >400%
  single-day moves (drop_corrupt_series counts moves) and never reverts to a stable
  median (apply_flap_screen needs reversion), and it sits far above the $0.10 floor.
  All three screens pass it. The volume-aware guard is the only thing that sees it.
  The scan surfaced two more of the same class no one had named: **EQ:HPH** (1
  share/day, 130× range) and **EQ:CPWR** (Compuware, taken private 2014 → all 2016+
  data is reused-ticker garbage).
- **EQ:COL (Rockwell Collins, delisted 2018-11)** — the MI class again: tiingo serves
  the real ~$87 entity, yfinance a recycled ~$0.30 penny. Each leg is internally
  smooth (per-vendor ingest screens pass both); the merged/deduped series flaps and
  racks up 595 raw (21 post-dedupe) >400% moves. drop_corrupt_series *would* catch it
  — but only on the MERGED panel, and it runs PER VENDOR at ingest, before the merge.

**Root cause (structural, not a threshold miss):** `apply_hygiene` runs inside each
vendor loader on that vendor's own frame, *before* `write_curated` merges vendors
under one `instrument_id`. Cross-vendor entity disagreement (COL) and single-vendor
dead-drift (SLE) are both invisible pre-merge — and nothing in hygiene ever looked at
volume.

**Fix.** (1) New guard `hygiene.drop_illiquid_series` — drops a series only on the
CONJUNCTION *median volume < 100 shares AND max/min close > 100×*. The conjunction is
what keeps it safe: an NVDA-like 100×+ survivor is spared by volume (millions of
shares); a quiet flat delisted tail (EQ:PCL/EQ:CA, ~1.1× range) is spared by range.
Wired into `apply_hygiene` (reported as `counts["illiquid_series"]`), so future dead
vendor series are caught at ingest. (2) New pattern-driven pass
`scripts/remediate_illiquid_equity.py` re-runs the series guards on the vendor-DEDUPED
panel (the view the risk model consumes) — this is where COL's flaps and SLE's
drift are finally visible. It dropped {COL, CPWR, HPH, SLE} = 5,603 rows across 11
year partitions (audit: `data/audit/equity_illiquid_series_remediation_20260711.json`);
CTRA (Coterra — real tiingo series, a brief alpaca garbage leg already resolved by the
dedupe: 0 post-dedupe moves) was correctly spared. 5 new tests in
`tests/test_universe.py` pin the guard (drops dead-wild, spares liquid-explosive and
dead-flat, no-ops without a volume column).

**Verdict, post-remediation:** market↔Mkt-RF **0.920 PASS** (0.691 → 0.920, squarely
in-band), momentum↔Mom **0.581** (0.106 → 0.581, a 5.5× recovery). Momentum lands on
the previously-documented *benign* floor (see the MI-incident entry below: alpaca-only
delisted-collapse fragments — real FRC/SIVB histories that reduce survivorship bias but
lower the Mom correlation; yfinance-only view PASSes ~0.63). size↔SMB -0.377 (expected
large-cap-universe artifact, ungated). The COL wholesale drop loses a real
(delisted-2018) name; a future vendor-aware *leg* drop could keep the good tiingo leg —
noted, not built.

## [2026-07-11] Iterations 6-7 + the MI incident + first PASS of the loop

**Iteration 7 — political underconfidence: PASS, first of the loop.** Pre-registered
BEFORE the Politics backfill existed; 499 series / 168,898 rows ingested (category
enumeration + min-settled-markets cost control; the 4h fetch survived a stamp-time
crash via raw-zone replay — mixed-precision ISO fix in stamp_availability). Verdicts:
Q1 MZ ψ=+0.0371 t=5.64 (n=6,011); Q2 favorite-backing [0.70,0.95] **+4.82% post-fee
t=2.93**; Q3 recency +4.56% t=2.28. Promotion checks: date-clustered t=2.94 (332
settle dates), liquid-≥10k-contracts +4.1% t=1.81 — both PASS. Favorite-tilt signal
(p̂ = p + 0.03 haircut, category-in-data fail-closed, same-night group caps) built
per the binding design in [[questions/research-political-underconfidence]].
Domain conditionality measured: the same mechanism is dead in macro (iteration 5).
SLEEVE-SCALE MEASUREMENT (walk-forward, real engine): as-wired the tilt LOSES
(Sharpe −0.67 — entries weeks before close where no edge was claimed + the flat
200bp haircut ~2×-overcharging favorites); the pre-registered evidence-matched
variant (≤10d nearness, actual taker fee once) WINS: +0.43 full span, +1.30 last
18mo, +0.43 ex-election-month, corr ~0, ERC combined 1.29 vs 1.21 book-alone.
Decision rule PASS → execution-layer re-spec landed (signal nearness gate +
accurate Whelan fee curve replacing iteration-25's flat 200bp for the whole
sleeve). Caveat on record: settlement-night lumpiness (top-5 days ≈ 105% of
additive P&L) — bounded by the same-night group caps.

**The MI incident — a NEW corruption class.** First-ever French validation on this
machine fired (market↔Mkt-RF 0.538): cross-vendor ENTITY DISAGREEMENT on reused
tickers (KG/MI/SBNY) — each vendor internally smooth (every ingest screen right to
pass them), but yfinance and alpaca serve DIFFERENT entities; vendor-mixed reads
manufacture run-alternators. Remediated (9,125 + 1,294 rows incl. CHK/MNK
bankruptcy-splices, audit records), blocklist extended + made machine-portable
(TIE/BMC/CFC added), analysis readers now vendor-dedupe by the lake's
latest-vintage rule. Post-fix: **market 0.948 PASS** (best ever measured);
momentum 0.565 vs 0.6 gate with verified benign cause (alpaca-only delisted
fragments — real FRC/SIVB collapse histories reducing survivorship bias;
yfinance-only view = 0.635 PASS).

**Iteration 6 — risk-harness baseline + first adoption adjudication.** Baseline
(coverage-core panels): families 1/3 in-band everywhere; the failure map is
family-2/4 — equity L/S B=1.69 and MVP B=2.34 (structural model treats residual
co-movement as zero — THE model weakness, iteration-8 target via wiki Q5
exposures), crypto uniformly over-forecast (halflife-90 EWMA too slow both ways:
fc/real ≈ 1.24 median in calm years, B>1.4 in spike years — per-sleeve halflife is
the candidate), fx MVP B=0.591 (shrinkage inflates the spectrum bottom ~3x; probe:
raw EWMA fc/real 1.15 vs lw_cc 2.19). Q3 (NW horizon) formally CLOSED — predicted
signature absent, do not build. Candidate `fixed:0.0` for covariance sleeves:
fx f4 0.591→1.074 into band, ALL five sleeves' MVPs realized lower vol (intl
p=0.004), no new failures — but **NO-ADOPT: criterion 2 failed at 50% vs the 60%
bar** (magnitude-blind cell counting; the rule held against a tempting change;
criteria revision, if any, must be pre-registered first). Equity+crypto cells were
re-run on the remediated deduped lake: equity f1-f3 now **IN BAND** (0.956 /
1.024 / 0.971) — the L/S family-2 failure (was 1.69) was CONTAMINATION, not
structure; equity MVP barely moved (2.34→2.185, still OUT) — structural
residual co-movement confirmed as THE remaining equity weakness (Q5/iteration-8
target). Crypto: all four cells OUT (f1 1.207, f2 1.701, f3 1.189, f4 1.411 —
net under-forecast; the spike-year lag dominates the calm-year over-forecast),
baseline health FAIL on the crypto random books — exactly the failure the
pre-registered halflife candidate below targets. Caveat: crypto coverage core
is 5/25 ids at the 98% bar. Ledger `risk_harness_20260711T211638Z.json`.

PRE-REGISTERED CANDIDATE (crypto halflife, written before running): probe evidence
(quarterly as_ofs 2016-2026, equal-weight crypto book) shows halflife-90 EWMA lags
crypto's vol cycle in BOTH directions — median fc/realized ≈ 1.24 in calm years,
B>1.4 in spike years (2017/2021/2024). Candidate: `ewma_halflife_days: 30` for the
CRYPTO sleeve only (structural factor cov + its small-sleeve fallback), all else
unchanged. Adjudication: the standard 4 criteria via adoption_verdict on the
crypto cells + the crypto MVP horse race; ADDITIONALLY the by-year fc/real spread
(max−min of yearly medians) must NARROW vs incumbent — the specific failure being
fixed. NO-ADOPT if any criterion fails; a different halflife value after seeing
results = a new pre-registration.

VERDICT (adjudicated same day, `adjudication_crypto_halflife_20260711.json`):
**NO-ADOPT — decisively.** 0/4 cells improved |B−1| (f1 1.207→1.270, f2
1.701→1.793, f3 1.189→1.206, f4 1.411→1.512); MVP horse race WORSE (cand
realized 1.272 vs inc 1.157 ann vol, p=0.606). The pre-registered spread
criterion alone passed (by-year fc/real max−min 0.738→0.453) — which exposes
the diagnosis error: hl-30 tracks the vol cycle more TIGHTLY year-by-year yet
forecasts 21d-ahead vol WORSE, because at h=21 vol mean-reverts — a fast EWMA
extrapolates transient spikes/calms that revert, while hl-90 partially anchors
to the long-run mean. "Too slow" was the wrong read of the probe; the correct
candidate shape is a two-component forecast (fast component blended with a
long-run anchor, GARCH-style) — that is a NEW registration if pursued. Crypto
under-forecast (all B>1) remains the open risk-model failure.

**Infra closed today:** #14 cost overrides (backtest + live path; floor-lowering
vector found and clamped), #15 shortfall_log (TCA loop closed end-to-end), #18
consumption half (quarantine filters backtest equities), #20 longshot_bias macro
exclusion (documented re-spec), #21 events calibration (FAIL — mid-range already
calibrated; #19 demoted). Data layer: ALFRED-vintaged macro (owner key; 63k
fallback rows purged), alpaca 1.07M + cross-check re-run (64 quarantined), tiingo
frontier fix + 404 negative cache (~206/924 and advancing), CM crypto CSVs (BTC
2010+, floor/ratchet/fallback fixes), Binance vision bulk (delisted-retention
VERIFIED; ms→µs timestamp migration caught), events live-snapshot CLI path.
Backtest on the remediated lake + DEEP basis (50,312 rows, 2020+, was 33.6k
2021+): net Sharpe **−0.041** on the protocol window (2019-01..2026-07;
was +0.183 pre-remediation/shallow-basis), deflated 0.000, CI [−0.63,+0.56] —
the honest headline remains *indistinguishable from zero*, now with a
negative point estimate. Window robustness: 2013+ −0.035, 2022+ −0.024.
Attribution is ENTIRELY the crypto sleeve (+0.162→−0.177; fx unchanged
+0.05), and the forensic says it's a REGIME FLIP, not early-data noise:
basis_carry per-year rank-IC vs fwd-5d = 2021 −0.037 (t=−2.9, the validated
fade era) → 2022/23 weakly negative → **2024/25/26 POSITIVE (+0.018/+0.019/
+0.040, 2026 t=+2.5)**; full-sample IC 0.0003. Factor-health action is a
decision for a pre-registered round: demotion/re-spec of basis_carry burns a
trial; monitoring note filed. n_trials unchanged at 24.

CORRECTION (same day, post-forensic): the old +0.183 was NOT era-selection —
it was a **measurement artifact**. In the shallow-basis run the crypto sleeve
never traded at all (turnover 3.02e-9: the optimizer correctly sat out below
the 30bp floor, leaving ~1e-9 CVXPY residual weights), `sharpe()` reported
the resulting solver noise as +0.162 (scale-invariant mean/std on a ~0/~0
series — fixed 223ce1d, NaN-guard + 3 regression tests), and a SECOND bug
levered the headline onto that noise: `allocation.py`'s ERC fixed point
treats a near-zero-variance sleeve as risk-free and allocated **99.994% of
capital to the dead sleeve** (fixed b27b834 warm path + b43f1f8 cold
inverse-vol path — the cold fallback had the identical failure). In the
DEEP-basis run the sleeve genuinely traded (turnover 0.338) and genuinely
lost −0.177 — the regime-flip losses are real. Every historical headline
Sharpe from this engine predating these fixes is untrustworthy.

**HONEST HEADLINE (all three fixes live, 2019-01..2026-07): net Sharpe
−0.005** (gross +0.005), ann vol 0.9%, PSR(>0) 0.494, deflated 0.000, CI
[−0.60, +0.63] (`backtest_20260712T013448Z`). The gated book — basis_carry
(crypto, regime-flipped) + rate-differential carry (fx, tiny) — is FLAT
ZERO at trivial deployed risk.

UPDATE (events sleeve live in the book, 9deb7ec + category remediation):
**net Sharpe +0.247** (PSR 0.791, deflated 0.000, CI [−0.33, +0.92],
`backtest_20260712T045301Z`-era report). Per-sleeve: events **+0.423 —
matching the +0.43 adoption-round walk-forward independently**, ann ret
6.6%; crypto −0.177 (the regime-flipped fade — demotion stays with the
pre-registered basis_carry health round); fx +0.05. The book's edge is
now exactly where the evidence said it was. Known cosmetic: events
turnover prints 0 (series injected post-optimizer; not in the weights
frame). Strategic implications, in order: (1) the
events favorite-tilt (+0.43 walk-forward, the one measured live edge) is
not in this book — integration is the highest-value move, but NOTE the
prior ERC-combined measurement (1.29 vs 1.21) predates the allocation
fixes and must be re-measured; (2) equity breadth: nothing has passed the
alpha gate — iteration-8b (risk-side exposures) is in flight and factor
re-specs from the forensics backlog are the alpha path; (3) basis_carry
factor-health round (pre-registered demotion/re-spec decision).

## [2026-07-10] autoresearch | Kalshi mechanism diagnostics (iteration 5) — both negatives, machine migration

New machine (benja): lake was EMPTY (the 2026-07-07 lake lives on the PC machine).
Rebuilt core lake here: universe 874 equities, prices equity 1.62M rows + all ETF
sleeves + crypto 54.6k, french/fx/cot; macro via keyless fredgraph fallback
(conservative obs+1d stamps — owner: set FRED_API_KEY, re-pull). ~213 delisted
tickers skipped (the known Norgate gap). Also fixed 4 stale-fixture test failures
vs the post-gate registry (same class as 33e0126's).

- Rounds: 1 research (endpoint archaeology + full-text Whelan read) + build + 2
  pre-registered experiments. Pre-registrations COMMITTED before data contact.
- Backfill landed (backlog #16 done): 71,180 rows, 12 US macro series, 2021-07 →
  2026-07, via external-api.kalshi.com /historical/* (legacy events aliased under
  KX series names; trades-not-candles for archived markets; expiration_value = the
  actual print). Live snapshot loader was doubly broken vendor-side (404 path +
  cents→dollar-string schema) — fixed with canned tests.
- Cleveland MONTHLY nowcast vintages ingested (nowcast_month.xlsx, target-month-
  tagged series ids, 2013-07→now; macro rows 12,936 → 33,198).
- **Diagnostic #1 (longshot fade): FAIL as a trade.** P1 replicates hugely (≤10c
  YES: −76.9% post-fee, t=−9.7) but P2 (the tradeable NO-side fade) = −0.9%,
  t=−0.4: the fee + calibration noise eat the complement's ~1-2% edge. Macro-only
  MZ slope insignificant (ψ=0.018, t=1.3) — matches the paper's own noisy
  Economics column. No promotion; maker-seat variant stays gated on #19.
- **Diagnostic #2 (nowcast drift): INCONCLUSIVE → dead.** β=+0.0091, t=1.60. The
  reason: the market's terminal MAE (0.066) BEATS the nowcast's (0.091) — market
  leads nowcast (Jia et al. direction); nothing to fade. Anti-edge noted.
- Trial accounting: n_trials unchanged at 24 (research diagnostics; registry
  untouched). Synthesis: [[questions/research-kalshi-mechanism-diagnostics]].
- Pages: [[sources/kalshi-historical-api]], [[sources/burgi-deng-whelan-makers-takers]],
  synthesis above. Backlog: #16 closed; #20 added (live longshot_bias category
  audit); #21 added (events calibration curve from resolved outcomes — zero new data).
- **Diagnostic #3 (TOY rebound, backlog #17): FAIL — a near-miss worth recording.**
  T1 PASS: mean 8-day loser-minus-winner spread **+248.6bp, t=2.26** across 10
  year-events; T3 PASS (survives 20bp costs easily). T2 FAIL: pooled TOY loser
  coefficient b1=−0.0234 (t=−2.07) is negative and significant but lands at ~the
  12-15th percentile of the 400-window placebo b1 distribution — above the
  pre-registered 10th-pct bar (−0.0282). Read: large-cap year-end loser rebound is
  economically real but not sharply separable from generic Oct-Mar reversal under
  our own rule. No promotion; verdict stands. Two observations FOR ANY FUTURE
  RE-SPEC (new pre-registration required): (a) placebo windows drawn from Oct-Mar
  can themselves sit adjacent to the year turn, contaminating the placebo left
  tail toward the TOY effect; (b) per-year C1 percentiles were upper-tail in 8/10
  years (median ~90th). Artifact: diagnostics/toy_rebound.json.
- Delegated builds in flight (Sonnet agents, per owner's orchestrator directive):
  Coin Metrics pre-2021 crypto rates loader, risk-model validation harness
  (iteration-4 spec). TOY diagnostic (above) was also a delegated build,
  pre-registered by the orchestrator.

## [2026-07-07] Gate v2 + re-specification round — n_trials 20 -> 24

Step-0 zero-trial diagnostics steered the round: momentum aggregation artifact
CONFIRMED (per-sleeve train t: equity +3.26, commodity +2.34 vs pooled unweighted
0.85); earnings_yield sector-tilt REFUTED (R2=0.000 vs pre-registered >0.25); COT
opposite-premia NOT CONFIRMED (both components same-sign positive). Two re-spec
trials funded, two declined.

Gate v2 (thresholds untouched): precision-weighted per-date IC aggregation
(weight (N-1)), purge+embargo at the 80/20 boundary, SE(OOS IC) reporting,
within-train CPCV sign-stability (reject-only, floor 0.60). Verdicts under v2:
- carry_rate_diff PASS (t=3.77, CPCV 0.96), basis_carry PASS (t=-2.06, CPCV 0.86)
- mom_12_1 (re-candidated): train t 0.85 -> 2.32 under weighting — criterion 1 now
  passes — but FAILS net validation (-0.66): the OOS L/S book loses net-of-cost.
  Measurement objection resolved; economics objection stands.
- tsmom (canonical MOP vol-scaled re-spec): train t=0.61, CPCV 0.37 — construction
  fix does not rescue it. Question closed.

Cleveland Fed nowcast VINTAGE history captured: the chart JSON carries the full
as-published daily path per quarter since 2013:Q3 — 12,908 rows ingested
(CLEV_NOWCAST_{CPI,CORECPI,PCE,COREPCE}), falsifying the research-session finding
of "no public archive" (believed-as-published caveat in the loader docstring;
daily re-pull is the ongoing verification). `--dataset nowcast` wired; the daily
scheduled task is an OWNER action (permission classifier blocks schtasks).

Engine tests + synthetic smoke pinned to an all-candidate registry snapshot (they
test mechanics and were authored pre-gate; live-registry coupling broke them once
real verdicts landed). OPUS.md handoff written at repo root.

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
