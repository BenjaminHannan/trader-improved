---
type: synthesis
title: "Research: Kalshi maker-side execution — the fee wall is 4x lower on the passive side"
created: 2026-07-12
updated: 2026-07-12
tags: [research, events, execution, fees]
status: developing
related:
  - "[[sources/kalshi-maker-fees-and-lob-data]]"
  - "[[sources/burgi-deng-whelan-makers-takers]]"
  - "[[questions/research-political-underconfidence]]"
sources:
  - "[[sources/kalshi-maker-fees-and-lob-data]]"
---

# Research: Kalshi maker-side execution

## Overview
Three separate findings died at the same wall this loop: taker fees. Maker fees
are 25% of taker (max 0.44c vs 1.75c). If passive fills are attainable at
acceptable adverse selection, the wall drops 4x — re-pricing two filed
negatives and cutting the live tilt's cost. The blocker shifts from "no data"
to "fill-probability calibration", which IS solvable (own WS capture or vendor
history).

## Re-priced arithmetic (gross of adverse selection — the honest caveat)
- Post-move REVERSAL fade (iteration 9 negative): +1.82c gross/trigger.
  Taker×2 = 2.45c → −0.6c (filed FAIL). Maker-in/taker-out ≈ 1.53c → +0.3c.
  Maker×2 ≈ 0.61c → **+1.2c/trigger** on n=18k triggers/5y.
- Longshot fade (iteration 5 negative): tradeable variant was ~0 NET OF TAKER;
  at 25% fees the sign is undetermined — needs a re-priced diagnostic (new
  pre-registration, zero-trial).
- LIVE favorite tilt: currently pays the full taker curve at entry. Entries
  have a ≤10-day window — ample time to rest limit orders. A filled-as-maker
  entry keeps 75% of the fee. Direct P&L improvement, contingent on fill.

## The catch (why nothing changes today)
Maker fills are conditional on flow hitting you — adverse selection means the
gross edges above are upper bounds. Required measurements, all needing LOB
data: fill probability vs (distance-from-touch, time-to-close, volume),
post-fill drift (how much of the edge evaporates conditional on filling), and
queue position dynamics.

## Data paths (owner decisions + one thing we can start free)
1. **Start our own capture now** (free, forward-only): a daemon on the
   `orderbook_delta` WS channel + periodic REST snapshot re-anchoring (the
   desync-on-gap pitfall is documented — must resnapshot). Months to
   accumulate enough politics-market fills; costs nothing but a process.
2. **Vendor history** (lycheedata 36GB+; kalshibacktest for crypto series):
   immediate backtestable fills; cost unknown — owner call.
3. Marriott (SSRN 6583921) proves full reconstruction from free streams is
   feasible; check for released code before writing our own.

## Open Questions
- Fill probability at the touch for 70-95c politics contracts inside 10d of
  close — the tilt's exact habitat.
- Does the 2026-07 tier table change Tier-0 formulas or only add discounts?
  (Fetch the PDF when unblocked.)
- Post-fill adverse drift vs the +1.82c reversal edge: does passive capture
  survive conditioning on being filled?
