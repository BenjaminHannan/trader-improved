---
type: synthesis
title: "Research: Garleanu-Pedersen dynamic trading (iteration 6)"
created: 2026-07-06
status: researched — NOT adopted (deliberate negative result)
---

# Research: GP partial adjustment vs our cost-penalized QP

## Key Findings
- GP (JF 2013, closed form): optimal policy = (1) aim in front of the target —
  down-weight fast-mean-reverting signals in the aim portfolio; (2) trade partially
  toward the aim. Superior net returns on commodity futures vs naive rebalancing.
- The "trade partially toward" half is already emergent in our optimizer: the QP's
  linear transaction-cost penalty produces a no-trade region and partial adjustment
  toward the unconstrained optimum — the first-order behavior GP prescribes.
- The genuinely new half — persistence-weighting each signal by its mean-reversion
  speed — requires reliable per-factor alpha-decay estimates. Ours come from
  ic_decay half-lives, which on free daily data are noisy, and the factor gate
  already rejects anything with half-life < the rebalance interval (the worst
  offenders GP protects against).

## Verdict
NOT adopted now. Expected marginal gain is small (weekly rebalance, gated slow
factors, tcost-aware QP) while the input (estimated decay speed) is the noisiest
number in the system — misestimated persistence tilts are a net risk. Revisit when
the live IC monitor has accumulated real decay curves from paper trading; that data
turns GP persistence weights from a guess into an estimate.

## Sources
- Garleanu & Pedersen, "Dynamic Trading with Predictable Returns and Transaction
  Costs", JF 68(6) 2013 (SSRN 1364170 / NBER w15205) — search-level
