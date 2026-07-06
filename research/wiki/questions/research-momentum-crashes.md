---
type: synthesis
title: "Research: momentum crashes (iteration 18)"
created: 2026-07-06
status: researched — partially covered by existing overlays; DM scaling deferred
---

# Research: Daniel-Moskowitz momentum crashes vs our stack

## Key Findings
- Momentum crashes are forecastable: they cluster in "panic" states (market below
  trailing highs + high volatility) and are driven by the option-like payoff of the
  SHORT losers leg during rebounds. Robust across every quarter-century of US data,
  international equities, futures, FX, commodities (Daniel & Moskowitz, JFE 2016).
- DM dynamic momentum (scale by conditional mean/variance of momentum itself)
  roughly doubles static momentum's Sharpe — and momentum is one of only two
  factors whose vol-scaling survives costs (Barroso-Detzel, iteration 3).

## Analysis vs the existing stack
Our protections already fire in exactly DM panic states, at book level:
- vol-target overlay (EWMA, deadband) de-risks when realized vol spikes —
  Barroso-style scaling, the cost-robust variant;
- drawdown ramp de-risks after declines (the other half of the panic definition);
- VIX/VIX3M backwardation trigger (iteration 14) is a market-level panic flag;
- long-short construction with position caps + beta-band bounds the short-losers
  convexity that drives the crash mechanics.
What we do NOT have: momentum-FACTOR-specific conditional scaling (the DM alpha
doubler). It is genuine factor timing; estimating momentum's conditional mean needs
long real histories (their evidence uses 80+ years) and cannot be validated on our
synthetic data.

## Verdict
DEFERRED, with a concrete trigger: once real ingest + gate history exist, the IC
monitor's momentum decay series IS the conditional-mean estimate DM needs — add a
panic-state momentum IC haircut then, as a monitored live adjustment rather than a
backtest-fitted rule. Filed to prevent premature factor-timing complexity now.

## Sources
- Daniel & Moskowitz, "Momentum Crashes", JFE 122(2) 2016 (NBER w20439)
- Barroso & Santa-Clara momentum risk management (via iteration-3 sources)
- Alpha Architect summaries — search-level
