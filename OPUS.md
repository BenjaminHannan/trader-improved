# OPUS.md — session handoff (2026-07-12, rewritten by Fable post-iteration-9)

Autonomous research loop on the BENJA machine (C:\Users\benja\...). Read first:
CLAUDE.md (hard rules), research/wiki/log.md (top entries are newest and carry
every verdict), MEMORY.md (owner directives: Sonnet-for-implementation
orchestration, fill-wait-time-with-work, token-efficiency practices).

## Non-negotiables (compressed — CLAUDE.md governs)
- `uv run pytest -q` green before any commit. Never weaken tests/thresholds/floors.
- configs/factors.yaml n_trials (=24) is machine-written; diagnostics are
  zero-trial; any re-spec after seeing results = NEW pre-registration committed
  to git BEFORE data contact (house pattern, several precedents in the log).
- Push to claude/intelligent-davinci-fiplrq (never main).

## Headline state (all committed & pushed through e9485e5)
- **Honest book Sharpe +0.247** (2019-2026, PSR 0.791, deflated 0.000 at
  n_trials=24, report backtest_20260712T*). Per-sleeve: events +0.423
  (independently matches the +0.43 adoption-round measurement), crypto −0.177
  (basis_carry — health round said KEEP, see below), fx +0.05.
- That number is only trustworthy because THREE measurement bugs were found and
  fixed same-day: (1) sharpe() reported solver-noise on dead sleeves (metrics.py
  NaN-guard, 223ce1d); (2)+(3) ERC warm path AND cold inverse-vol path levered
  ~100% of capital onto near-zero-variance sleeves (allocation.py degenerate
  exclusion, b27b834 + b43f1f8). Every pre-fix headline was an artifact —
  the wiki CORRECTION entries document the whole chain.
- Events sleeve is IN the main book (d665a3b + 9deb7ec): run_backtest unions
  event_markets_hist (240k rows, category-stamped by remediation, audit record
  20260712T025315) with the live snapshot; tilt signal fail-closes without
  category=="politics" — that was why its first run traded zero.
- Four rules held against tempting changes in one day: fixed:0.0 (earlier),
  crypto halflife-30 (NO-ADOPT, mean-reversion mechanism identified),
  Q5-as-registered (NO-ADOPT by 0.14pp R²), factor-health round (NO-DEMOTE —
  the basis_carry "regime flip" narrative was a raw-vs-traded sign confusion;
  sign-adjusted trailing IC is +0.0497 t=+2.36).
- Iteration 9 closed negative: Kalshi post-move drift is REVERSED
  (over-reaction, −1.82c t=−12.8, n=18k) and the fade is taker-fee-eaten.
  Maker-side fade filed as data-blocked (no fill-probability data).

## In flight at handoff
1. **Iteration 8b** (pre-registered d1471df): Q5 exposures re-adjudicated
   against the equity MVP mechanism cell. Runner: scratchpad/adjudicate_q5_mvp.py
   (background id br4ul1xw2, 6h+ — the step-5 MVP horse race over 574 ids is
   the long tail). ADOPT iff f4 B_cand ∈ [0.408,1.593] AND f1-f3 stay in band
   AND race p<0.05. If ADOPT: flip configs/risk.yaml equity exposures
   (+liquidity +earnings_yield), French validation + full suite, commit.
2. 30-minute session loop (cron 0e31c20b) + tiingo tranches every ≥70min
   (~465/924 symbols; key in user env, export inline for background shells).
3. Registry now has demote()/status "demoted" + scripts/factor_health.py —
   health rounds are one command (+ --apply gated on the committed rule).

## Open frontier (ordered)
1. 8b verdict → wiki + (maybe) config flip.
2. Sign-convention audit: gate_stats (shallow lake), raw-basis forensic, and
   deep-lake ic-report disagree on basis_carry's direction story. One script,
   one wiki entry — clears a landmine under every future health round.
3. P&L-based second health rule (trailing factor-attributed net Sharpe) —
   if adopted, BOTH rules reported every round (anti-shopping clause in log).
4. Two-component crypto vol forecast (fast + long-run anchor) — the halflife
   post-mortem's candidate shape; needs a fresh registration.
5. Alpha breadth is EXHAUSTED on current data (mom/tsmom re-specs burned,
   earnings_yield/COT mechanisms refuted, drift reversed+fee-eaten): new alpha
   needs new research or new data (Polymarket cross-venue is the unexplored
   scan candidate; needs an ingest).

## Traps (additions — old lists in git history still apply)
- Background Bash with `| tail` buffers everything: the output file stays
  0-byte until exit. Launch long jobs WITHOUT pipes when you'll want progress.
- Sonnet agents stall on their own background runs ("waiting on monitor").
  Don't wait: check git status / run their verification yourself, or
  SendMessage-resume them once.
- The tree may contain ANOTHER session's uncommitted files (hygiene.py,
  test_universe.py, french_validation.json, remediate_illiquid_equity.py).
  Never `git add` them; commit specific paths only.
- PowerShell `$_` inside bash double quotes gets eaten by bash — single-quote
  the whole PowerShell command.
- Kalshi: KX-prefixed series names; /historical/trades pages NEWEST-first;
  legacy market records only on external-api host; CM community API gutted
  to ~7d. asof_panel ties → latest available_from wins: backfills must be
  non-overlapping BY CONSTRUCTION. Sub-$0.10 hygiene floor is EQUITY-only.
- bybit is geo-blocked from this machine; okx funding retains ~96d (funding
  history cannot deepen without VPN/egress — owner option).

## Owner action list
1. Norgate Platinum decision ($630/yr) — delisted-equity gap + PIT membership.
2. Daily nowcast archiver scheduled task (schtasks blocked for agents).
3. VPN/egress if funding-rate history ever matters (bybit geo-block).
4. Polymarket data source decision if cross-venue research is wanted.
5. Kalshi LOB data decision — NOW QUANTIFIED: the post-move maker-side fade
   upper-bounds at +1.21c/trigger (t=+8.5, ~3.3k triggers/yr) and needs fill
   data to promote. Options: vendor history (lycheedata, cost unknown) for
   immediate calibration, and/or an always-on WS capture daemon
   (orderbook_delta + resnapshot-on-gap; needs a Kalshi API key + scheduled
   task — both owner actions). See research-kalshi-maker-execution.
