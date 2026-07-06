---
type: synthesis
title: "Research: factor momentum conditioning (iteration 22)"
created: 2026-07-06
status: implemented
---

# Research: factor momentum in the combination weights

## Key Findings
- Ehsani & Linnainmaa (JF 2022) / Gupta & Kelly (JPM 2019): factor returns are
  positively autocorrelated — the average factor earns ~1bp/month after a down
  trailing year vs ~53bp/month after an up year. Time-series factor momentum
  subsumes stock-level and industry momentum entirely, robust internationally and
  in commodity futures. This is the one factor-timing effect with JF-grade support
  (contrast the vol-timing critique, iteration 3).

## Decision → implementation
- The combination already weights factors by rolling shrunk IC (w = C⁻¹ ic). Factor
  momentum enters as a multiplicative tilt on those weights:
  `fm_k(t) = clip(1 + fm_gamma * sign(trailing_12m_factor_return_k), 0, 2)` where
  the factor return is the realized return of the single-factor top-minus-bottom
  quintile portfolio (already computable from z panels + prices, same machinery as
  the gate's net-validation check), embargoed positionally like the IC window.
  fm_gamma config `alpha.factor_momentum_gamma` (default 0.25; 0.0 = off exactly).
- Applied per (factor, sleeve) in the engine where IC* is computed; flows through
  both the plain and Grinold combination paths (tilt ic_k before C⁻¹ ic).
- Verification: gamma=0 bit-identical; a factor with persistently NEGATIVE trailing
  factor return gets down-weighted vs its IC-only weight (planted case); embargo
  corruption test (future factor returns cannot change the tilt at t); tilt bounded
  in [0, 2].

## Sources
- Ehsani & Linnainmaa, "Factor Momentum and the Momentum Factor", JF 77(3) 2022
  (SSRN 3336502, NBER w25551); Gupta & Kelly, "Factor Momentum Everywhere", JPM
  2019; QuantPedia replication notes — search-level.
