---
type: synthesis
title: "Research: political-market underconfidence (iteration 7) — FIRST PASS of the loop"
created: 2026-07-11
updated: 2026-07-11
status: PASSED pre-registration — promotion study is the next gate (spec below)
related:
  - "[[sources/practitioner-mechanism-scan-2026-07]]"
  - "[[questions/research-kalshi-mechanism-diagnostics]]"
---

# Political-market underconfidence: the favorite-tilt PASSES

Idea #6 of the practitioner scan (Le, arXiv 2602.19520: political prediction
markets show calibration slope > 1 — favorites underpriced — the OPPOSITE sign
to the favorite-longshot bias, domain-conditional). Pre-registration was
committed BEFORE the Politics-category backfill existed (commit "Pre-register
diagnostic #4..."); data: 499 political series / 168,898 rows (settled markets
2021-09 → 2026-07), the ≥5-settled-markets series filter as cost control.

## Verdict (diagnostics/kalshi_political_underconfidence.json)

| pre-registered test | result | pass? |
|---|---|---|
| Q1 sign: MZ ψ > 0, \|t\| ≥ 1.96 | **ψ = +0.0371, t = 5.64** (n=6,011; 1,201 event clusters) | ✅ |
| Q2 tradeable: YES at [0.70,0.95] post-fee > 0, t ≥ 1.645 | **+4.82%, t = 2.93** (n=575; 385 clusters) | ✅ |
| Q3 recency (18m) | **+4.56%, t = 2.28** (n=425) | ✅ |
| **Conjunction** | | **PASS** |

Contrast with the macro-economics panel (iteration 5): there the MZ slope was
insignificant (0.018, t=1.3) and the tradeable fade fee-eaten. Domain
conditionality is real and measured in OUR data, matching Whelan Table 8's
category heterogeneity. Fees at a 0.85 entry are ~0.9% of stake — the +4.8%
edge clears them ~5x.

## Promotion study (the NEXT gate — pre-registered here, before running)

A favorite-tilt signal lands in the events sleeve ONLY if all of:
1. **Election-night cluster robustness**: Q2 recomputed with SEs clustered by
   settle DATE (same-night state ladders co-resolve; event-level clusters
   understate that correlation). Bar: t ≥ 1.645 under date clustering.
2. **Liquidity reality**: Q2 restricted to markets with life volume ≥ 10,000
   contracts (tradeable size at retail scale) — point estimate > 0.
3. **Category purity in production**: the signal predicate must be a
   series-enumeration check (Politics category listing), mirrored from the
   longshot_bias macro-exclusion mechanism, NOT a title regex.
4. Sizing through the existing events-sleeve Kelly/caps machinery (10% sleeve
   risk cap, per-market and per-group caps unchanged). No registry factor;
   n_trials untouched; the signal addition itself is a documented spec.
Any change to these four after seeing their results = new pre-registration.

## Capacity caveat (recorded now)
Political favorites at 70-95c are the LIQUID side of the most liquid Kalshi
markets — but the edge per stake is ~5%, and election cycles concentrate
settlements (Q3's n=425 within 18 months is mostly 2024-11 + primaries).
Expect lumpy, cycle-synchronized P&L; the events sleeve's Bernoulli-variance
sizing already prices per-market variance but NOT cross-market same-night
correlation — the dedupe-by-event-group machinery must treat same-night
POLITICAL settlements as one correlated group (design note for the signal spec).

## Sources
- Le (arXiv 2602.19520) — 292M trades, political underconfidence
- [[sources/burgi-deng-whelan-makers-takers]] — category heterogeneity (Table 8)
- diagnostics/kalshi_political_underconfidence.json — full calibration table
