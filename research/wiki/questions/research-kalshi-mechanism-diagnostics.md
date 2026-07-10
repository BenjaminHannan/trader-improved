---
type: synthesis
title: "Research: Kalshi mechanism diagnostics (iteration 5) — both pre-registered tests negative"
created: 2026-07-10
updated: 2026-07-10
status: closed — two clean negatives, zero n_trials burned, infrastructure permanent
related:
  - "[[sources/kalshi-historical-api]]"
  - "[[sources/burgi-deng-whelan-makers-takers]]"
  - "[[sources/practitioner-mechanism-scan-2026-07]]"
  - "[[questions/research-events-sleeve]]"
---

# Kalshi mechanism diagnostics: the top-2 practitioner-scan ideas, adjudicated

Both pre-registrations were committed to git BEFORE the backfill finished
(commit "Pre-register the two Kalshi mechanism diagnostics BEFORE data contact");
data: 71,180 rows (daily trade bars + settlements), 12 US macro series, 2021-07 →
2026-07, 2,180 qualifying markets / 429 events after Whelan-style filters.

## Diagnostic #1 — longshot-fade (idea #1): FAIL as a trade, replicated as a fact

| pre-registered test | result | pass? |
|---|---|---|
| P1: YES ≤10c mean post-fee return ≤ −0.40, t ≤ −1.645 | **−76.9%, t = −9.71** (n=665, 301 clusters) | ✅ |
| P2: NO vs 5-20c YES, post-fee mean > 0, t ≥ +1.645 | **−0.9%, t = −0.40** (n=352) | ❌ |
| P3: P2 point estimate > 0 in last 12m | +2.6%, t = 0.56 | ✅ (vacuous) |
| **Verdict (conjunction)** | | **FAIL — no promotion** |

Why P1 and P2 coexist: the ≤10c bucket is priced 2.1x its true frequency (3.5c
mean price, 1.65% win rate) — but its complement trades at ~0.90-0.95, where the
same mispricing is worth only ~1-2% of stake and the taker fee + the calibration
noise absorb it. This is exactly the Whelan makers-takers asymmetry. Two local
facts sharpen the negative:
- the 10-20c band was UNDERpriced in-sample (win 19.4% vs 15.5% price, +17%
  post-fee) — the overpricing lives strictly below 10c;
- our macro-only pooled MZ slope is ψ = 0.018, t = 1.30 — insignificant, matching
  the paper's own noisy Economics column (their pooled 0.034*** is carried by
  other categories). No year is significant at 5% in our sample.

Implication for the LIVE events sleeve: `production/events/signals.py::longshot_bias`
fades longshots across all categories. Its macro-ladder slice is, on this
evidence, ~zero net-of-fee from the taker seat at daily granularity. Zero-trial
follow-up filed in backlog (#20): measure the live signal's category mix; consider
a category condition. The maker-seat variant stays gated on backlog #19's
calibration model (Whelan: makers also lose on average — the seat alone is not
an edge).

## Diagnostic #2 — nowcast drift (idea #2): INCONCLUSIVE → dead, with a reason

Regression (368 obs, 59 event-clusters, KXCPI/KXCPICORE/KXPCECORE vs
target-month Cleveland MoM vintages): **β = +0.0091, t = 1.60** → fails the
pre-registered |t| ≥ 1.96 on either side. Even if real, 0.9%-of-gap-per-day is
economically negligible.

The secondary horse race explains the null: **terminal |error| vs the print —
market 0.066, nowcast 0.091 (n=63 events)**. The market is ALREADY more accurate
than the nowcast at close; there is no under-reaction to harvest. This is the
Jia et al. sign (market leads nowcast) expressed in accuracy rather than in the
drift coefficient. Anti-edge note: do NOT build strategies that pull ladder
prices toward the Cleveland nowcast; if anything the information flows the other
way (a nowcast-error trade would need a different counterparty than Kalshi).

## What survives (permanent infrastructure, all committed)

- `KalshiHistoryLoader` (`event_markets_hist`, `--dataset kalshi_hist`): full
  settled-market archive with taker-side flow and PIT-clean settlement rows —
  reusable for ideas #3/#5/#6 (need non-macro series added to DEFAULT_SERIES)
  and for calibrating the events sleeve on RESOLVED outcomes (iteration-25's
  known approximation).
- Cleveland MONTHLY nowcast vintages (target-month-tagged series ids) — usable
  for any future macro-timing work with exact-unit MoM mapping.
- Live events snapshot loader fixed (endpoint + schema had both broken
  vendor-side ~2026-07).
- research/diagnostics/{_common,kalshi_longshot_fade,kalshi_nowcast_drift}.py +
  committed JSON artifacts in diagnostics/.

## Trial accounting

n_trials unchanged at 24. Both tests were research diagnostics on a dataset no
registered factor consumes; nothing touched configs/factors.yaml. Re-running
either idea with ANY changed knob (entry timing, bucket, maker seat, series set)
requires a fresh pre-registration.

## Open questions

- Idea #6 (political-market underconfidence) is now the top live scan candidate;
  needs political series added to the backfill (one DEFAULT_SERIES edit + re-run).
- Trade-cap truncation on 4 hyper-liquid FEDDECISION markets (30k trades) — raise
  the cap or page from the settlement end before any FEDDECISION-specific study.
- Events-sleeve calibration curve from resolved outcomes (score → realized
  probability) is now buildable from `event_markets_hist` — the iteration-25
  documented approximation can be closed with zero new data.
