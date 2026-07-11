# OPUS.md — session handoff (2026-07-11, rewritten by Fable mid-iteration-6/7)

You are picking up an autonomous research loop mid-stride on the BENJA machine
(C:\Users\benja\...; the pre-2026-07-10 history ran on a different machine).
Read first: CLAUDE.md (hard rules), research/wiki/log.md (top entry),
research/wiki/index.md (backlog), memory dir MEMORY.md (owner directives:
Sonnet-for-implementation orchestration, fill-wait-time-with-work,
token-efficiency practices).

## Non-negotiables (compressed — CLAUDE.md governs)
- `uv run pytest -q` green before any commit. Never weaken tests/thresholds/floors.
- configs/factors.yaml n_trials (=24) is machine-written; research diagnostics are
  zero-trial; any re-spec after seeing results = NEW pre-registration (the
  pre-register-then-commit-then-run pattern is established practice — see
  c23ed9e, and the political one committed before its data existed).
- Push to claude/intelligent-davinci-fiplrq (never main; PR #1 exists).

## State (all committed & pushed through 'Tiingo negative-result memory')
- Lake REBUILT on this machine + extended well beyond the old one:
  prices equity 1.62M (yfinance) + alpaca secondary 1.07M + tiingo trickle
  (~150/924 symbols, frontier-ordered, 404 negative-cache NEW), crypto 75.3k
  across ccxt + coinmetrics CSVs (BTC 2010+) + binance vision bulk (SOL/AVAX/
  NEAR/MATIC-incl-dead), french/fx/cot, macro = ALFRED VINTAGED (owner's FRED
  key; the fredgraph-fallback contamination was purged with an audit record),
  nowcast quarterly+monthly vintages, event_markets_hist 71k (12 macro series;
  POLITICS category ingest RUNNING in background right now).
- Iteration 5 (closed): three pre-registered negatives — longshot fade (P1
  replicates -77% t=-9.7; tradeable P2 ~0 net), nowcast drift (market BEATS
  nowcast, MAE 0.066 vs 0.091), TOY rebound (+249bp t=2.26 but fails placebo
  concentration). Wiki synthesis: research-kalshi-mechanism-diagnostics.
- Iteration 6 (open): risk-harness baseline vs the live model. First partial run:
  families 1/3 in-band everywhere scored; family-2 misses in 3/6 ETF sleeves
  (mixed directions); fx family-4 B=0.591. Probe evidence (commit 3bbf01d): MVP
  fc/realized = 2.19 lw_cc vs 1.15 raw; ALL shrinkage variants inflate the
  spectrum bottom; degeneracy is FX-basket structure (not just UDN/UUP).
  CANDIDATE staged: `--instrument-cov fixed:0.0 --sleeves <covariance sleeves>`
  one-command run + production/risk/validation.py adoption_verdict (4 criteria).
  BLOCKED ON: the full baseline (background python PID was ~15276, running since
  06:14, unmemoized code — if it died, re-run scripts/score_risk_model.py
  --start 2016-01-01; the committed CLI now memoizes Sigma (4x faster) and
  coverage-cores the panels).
- Iteration 7 (staged): political underconfidence, pre-registration COMMITTED
  (research/diagnostics/kalshi_political_underconfidence.py) BEFORE data; run it
  once the kalshi_hist_politics ingest lands.
- Execution/infra closed today: #14 cost-overrides in backtest + live path,
  #15 shortfall_log accrual (TCA loop closed end-to-end), #18 quarantine
  consumption (equity drop / ETF keep+warn), #20 longshot_bias macro exclusion
  (documented re-spec with evidence block in signals.py), #21 events calibration
  (FAIL — prices already calibrated mid-range; #19 demoted, prerequisite gone).
- Events sleeve: live snapshot loader FIXED (endpoint + dollar-schema breaks).

## In flight at handoff (check TaskList / background shells)
1. Harness baseline (equity leg) -> then: candidate run + adoption_verdict +
   consolidated iteration-6 wiki entry (family-2 heterogeneity analysis mine).
2. kalshi_hist_politics ingest -> then run the pre-registered diagnostic #4.
3. Sonnet agents: daily_run smoke-test consolidation (tests/test_execution.py);
   French-validation wiring+run (scripts/validate_french.py NEW) — the
   TIE-detector has NEVER run on this machine's rebuilt lake; interpret its
   correlations when it reports (market ~0.9 healthy, size negative expected).
4. Tiingo: re-run `--dataset tiingo --full` (keys in user env; export inline for
   background shells) roughly hourly; the 404 cache makes each run advance.

## Traps (new ones since the 07-07 list — old list still applies)
- Kalshi: query series by KX-prefixed name only; /historical/trades pages
  NEWEST-first (truncation keeps the right end); market records for legacy
  events exist ONLY on external-api host; community CM API + newer-asset CSVs
  are gutted to ~7 days.
- asof_panel ties break to LATEST visible available_from — backfill feeds must
  be non-overlapping BY CONSTRUCTION (see coinmetrics/binance loaders), never
  by stamp games.
- The sub-$0.10 hygiene floor is EQUITY-only; never apply to crypto loaders.
- Agents pause on their own background pytest — pick their work up directly
  (git status; run their tests yourself) instead of waiting.
- PowerShell mangles inline quotes/regex — scratchpad script files.

## Owner action list
1. Norgate Platinum decision ($630/yr) — delisted-equity gap (~213 names) +
   independent PIT membership.
2. Daily nowcast archiver scheduled task (schtasks blocked for agents).
3. Optional: CryptoCompare free key if the sol/avax-class pre-Binance-listing
   crypto tail ever matters (Binance vision covered most of it).
