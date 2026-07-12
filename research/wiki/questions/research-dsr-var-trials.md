---
type: synthesis
title: "Research: DSR variance-of-trials from the registry (iteration 5)"
created: 2026-07-06
status: implemented
---

# Research: estimating V[SR_trials] from the gate ledger instead of a proxy

## Overview
DSR needs three inputs: n_trials, the variance of Sharpe ratios ACROSS trials, and
the winning strategy's return moments. Our report uses n_trials from factors.yaml
(good — config, not a guess) but proxies var_trials with the Sharpe-estimator
variance (documented limitation from Phase 7).

## Key Findings
- Bailey & López de Prado 2014: SR0 = sqrt(V[SR_n]) * ((1-γ)Φ⁻¹(1-1/N) +
  γΦ⁻¹(1-1/(Ne))) — the expected max under N independent trials REQUIRES the
  empirical variance of the trial Sharpes; understating it understates SR0 and
  inflates DSR.
- The trials are exactly what our gate ledger records: every factor put through the
  gate is a trial, and its single-factor net validation performance is the trial's
  Sharpe. The registry is therefore the correct, auditable source for V[SR_trials].
- Sidak/Bonferroni are cruder alternatives; DSR with empirical variance dominates.

## Decision → implementation
- GateStats gains `val_sharpe` (annualized net single-factor Sharpe on the
  validation slice); build_factors records it on every gate attempt (pass or fail —
  failed trials count, that is the whole point of multiple-testing control).
- report.py: when >= 2 recorded trials exist in factors.yaml, var_trials =
  population variance of recorded val_sharpe values **converted to per-observation
  units (÷ sqrt(252)) to match the DSR algebra's sr_pp** — caught at implementation
  review: annualized-units variance over-deflates by ~252x; else fall back to the
  existing proxy WITH a report caveat naming which path was used.
- deflated_sharpe.py unchanged (it already takes var_trials as an argument).

## Verification (in-repo)
Registry roundtrip stores/reads val_sharpe; report picks the empirical path when
trials exist and the proxy otherwise, caveat string switches accordingly; DSR is
monotonically decreasing in var_trials (property test).

## Sources
- Bailey & López de Prado, "The Deflated Sharpe Ratio" (davidhbailey.com PDF;
  SSRN 2460551) — search-level
- Wikipedia "Deflated Sharpe ratio" (formula cross-check)
