---
type: synthesis
title: "Research: tranching + per-sleeve rebalance cadence (iteration 24)"
created: 2026-07-06
status: implemented
---

# Breadth through time: tranching and crypto cadence

## Design (pinned)
1. **Tranching**: run K weekly sub-portfolios on grids offset by 0..K-1 trading
   days and average the weights (cfg walk_forward.n_tranches, default 1 = off,
   recommended 5). Kills rebalance-day timing luck; equivalent to more independent
   revisits of the same signals (fundamental-law breadth via time diversification
   of the decision point). Costs: turnover of the AVERAGED book is what gets
   charged — tranching naturally smooths trades (1/K of the book moves per day),
   so cost per unit of book does not increase.
2. **Per-sleeve cadence**: cfg walk_forward.rebalance becomes overridable per
   sleeve (map), crypto -> "twice-weekly" grid (Mon+Fri close; a middle step
   before daily — 30bp floors make daily marginal at v1 alpha strength; the gate's
   net-of-cost check adjudicates). Other sleeves stay weekly.

## Verification
Tranche average = plain run when K=1 (bit-identical); K=5 book's daily turnover
strictly lower than K=1 at same average exposure on synthetic data; per-sleeve grid
respected (crypto trades on the extra dates, equities don't); PIT corruption test
still passes with both features on.
