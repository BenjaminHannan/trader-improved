# OPUS.md — session handoff (2026-07-07, written by Fable at plan exhaustion)

You are Opus 4.8 picking up mid-stride. The working tree contains UNCOMMITTED
in-flight work with ONE known failing test. Finish it, then work the queue.
Read first: CLAUDE.md (hard rules), research/wiki/log.md (top two entries),
research/wiki/questions/research-oos-gate-design.md and
research-rejected-factor-forensics.md.

## Non-negotiables (compressed from CLAUDE.md — full text governs)
- Every commit gated on `uv run pytest -q` green. Never weaken tests/thresholds/
  cost floors. PIT via available_from everywhere. No reports in data/.
- `configs/factors.yaml` n_trials is machine-written ONLY. Statuses may be flipped
  candidate<->rejected only as a documented re-specification decision (commit msg
  must say why). Every gate --apply run charges trials — that's the point.
- If a factor fails the gate, it fails. Record it. Diagnose with zero-trial checks
  before funding a re-spec (see Step-0 pattern in the wiki log).

## State
- Committed through `0e61b20` + research-session wiki commits. Pushed to
  claude/intelligent-davinci-fiplrq (never push main; PR #1 exists).
- First live gate ran: 2/12 accepted (carry_rate_diff, basis_carry), n_trials=20.
  Backtest artifact committed: deflated Sharpe 0.000 (honest near-empty book).
- UNCOMMITTED in working tree (all mine, all intentional):
  1. Gate infra (scripts/build_factors.py): precision-weighted per-date IC
     aggregation (weight by n_names-1), purge+embargo at the 80/20 boundary,
     SE(OOS IC) + within-train CPCV sign-stability (reject-only, `_cpcv_sign_stability`).
     Tests in tests/test_signal_bundle.py — GREEN.
  2. Canonical vol-scaled tsmom (production/signals/momentum.py, MOP 2012 form,
     trailing vol shift(1)) replacing the near-degenerate sign() version — tests
     mostly green, ONE failure remains (next section).
  3. Cleveland Fed nowcast loader (production/data/loaders/stage2/cleveland_nowcast.py)
     — DONE, tested, wired as `--dataset nowcast` + stage-2 batch; 12,908 as-published
     vintage rows already ingested (2013-07→now). This falsified the research page's
     "no public vintage archive" claim; caveat re believed-as-published is in the
     loader docstring.
  4. tests/test_alpha_refine.py fixed (stale live-registry fixture assumption).

## Finish line — COMPLETED by Fable before handoff (kept for context)
(All steps below are DONE and committed: gate v2 ran, n_trials=24, mom_12_1
failed net-validation, tsmom-canonical failed train-t; suite green. Start at
the "Queue after that" section.)
1. Fix `tests/test_sleeves.py::test_factors_config_parses_and_carry_curve_resolves`:
   it asserts `spec["status"] == "candidate"` against the LIVE registry — stale
   fixture assumption now that the real gate has run (status is 'rejected'). Fix the
   ASSERTION to accept any valid status in {candidate, accepted, rejected} or compare
   against the loaded file itself — same pattern I used in test_alpha_refine.py
   (see its "unrelated factors untouched" comment). Do NOT touch the registry.
   Then scan for any OTHER test hardcoding pre-gate statuses: `grep -rn "candidate"
   tests/` and judge each.
2. Flip statuses to candidate for exactly two factors in configs/factors.yaml, as a
   documented re-specification decision (commit message must carry the evidence):
   - tsmom: construction replaced with canonical vol-scaled form (old sign() verdict
     stands in history).
   - mom_12_1: rejection shown to be an aggregation artifact (per-sleeve train t:
     equity +3.26, commodity +2.34; pooled unweighted 0.85 reproduced exactly; the
     new precision-weighted estimator is the fix). Leave gate_stats blocks as-is;
     --apply will overwrite them.
   Do NOT re-candidate earnings_yield or cot_positioning: their pre-registered
   diagnostics FAILED (sector-tilt R²=0.000; COT components both same-sign positive
   full-window, not the KRT opposite-sign pattern). Zero-trial checks said no.
3. `uv run python scripts/build_factors.py --ic-report --start 2016-01-01` — INSPECT
   (new columns OOS_SE + CPCV should print; expect ~4 factors gated: 2 accepted
   re-scored under the new estimator + the 2 re-candidates). Then `--apply`.
   n_trials will rise by the number gated — correct and intended.
4. If mom_12_1 or tsmom pass: re-run the backtest
   (`uv run python scripts/run_backtest.py --start 2016-01-01`, ~50-90 min,
   background it), force-add the new reports/*.json, commit.
5. Full suite green → commit everything in logical chunks (gate infra + tests;
   tsmom re-spec + status flips + gate results; cleveland loader; backtest artifact)
   → push → append a wiki log entry (follow the top entry's format: numbers, deltas,
   decisions, trial accounting).

## Queue after that (specs already exist)
- Kalshi resolved-market backfill loader (wiki backlog #16; free API; schema like
  form-style loaders; then run the TWO pre-registered tests in
  research/wiki/sources/practitioner-mechanism-scan-2026-07.md ideas 1-2 as
  research diagnostics — they only touch the ledger if promoted). The nowcast-drift
  test's Cleveland data is ALREADY ingested (series CLEV_NOWCAST_*).
- Coin Metrics community-rates loader for pre-2021 crypto depth
  (research-data-remediation.md has the endpoint/format; stamp availability earlier
  than ccxt's 24h rule so coinbase stays primary).
- Risk-model scoring harness per research-risk-model-validation.md (bias stats +
  MVP horse race). Zero trial cost; build once, use forever.
- Turn-of-year diagnostic (backlog #17) — cheap, zero new data.

## Traps learned tonight (do not relearn them expensively)
- Background Bash shells do NOT inherit user-scope env vars set mid-session: export
  TIINGO_API_KEY / APCA_* inline in the command. Keys live in Windows user env.
- Secondary price feeds share the `prices` watermark: their backfills need `--full`.
- bybit/okx IP-ban aggressive callers (~1h). Loader pacing now handles it; never
  probe venues in tight loops.
- SEC ~8 req/s; one process at a time. Tiingo free ≈ 50 symbols/hr (429 guard stops
  cleanly; re-run to extend). Alpaca IEX history starts ~2021.
- Vendor series for DEAD tickers can be wrong-entity garbage that cross-vendor
  checks cannot see (no overlap). hygiene.drop_corrupt_series +
  apply_flap_screen now guard ingest; the French-benchmark check
  (production/risk/model.py:validate_against_french — needs FF_* renames and
  monthly compounding at the call site, see wiki log) is the detector. Re-run it
  after any big equity re-ingest.
- write_curated never DELETES rows a re-ingest no longer emits — lake remediation
  is a separate explicit step with an audit record (see
  data/audit/*remediation*.json for the pattern).
- schtasks /Create is blocked by the permission classifier — the daily nowcast
  archiver task is on the OWNER's action list, not yours.
- A second research-only session may be running: it writes ONLY research/wiki/.
  pull --rebase before pushing; if log.md conflicts, both entries survive, order
  newest-first.

## Owner action list (remind them if unaddressed)
1. Approve/register the daily nowcast archiver:
   `schtasks /Create /TN trader-improved-nowcast-archiver /TR "cmd /c cd /d C:\Users\PC\Downloads\trader-improved && uv run python scripts/ingest.py --dataset nowcast" /SC DAILY /ST 10:20`
   (time = shortly after 10:00 ET publication, adjust for machine TZ; machine must be on).
2. Norgate Platinum decision ($630/yr) — closes the 184-name delisted-equity gap AND
   provides an independent PIT-membership cross-check.
3. Optional: email Cleveland Fed research for the EC-2023-06 vintage dataset to
   verify our believed-as-published caveat (draft on request).
4. Tiingo stays free-tier (equity depth accrues ~50 symbols/hr when run); no
   multi-key workarounds — ToS.
