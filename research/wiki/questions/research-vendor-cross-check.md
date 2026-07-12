---
type: synthesis
title: "Research: cross-vendor price validation (iteration 12)"
created: 2026-07-06
status: implemented
---

# Research: defending against yfinance corporate-action corruption

## Key Findings
- Documented, recurring yfinance failure modes: mixed split-adjusted prices with
  unadjusted dividends (adjusted closes silently wrong), missing bars, zero-volume
  rows; yfinance even ships a "price repair" feature acknowledging the problem.
- The standard defense named across sources: cross-check corporate actions and
  prices against an independent second source. We already ingest one (Stooq) as a
  fallback — but nothing compares them.
- Compare RETURNS, not levels: vendors differ legitimately in adjustment convention;
  a same-day return divergence beyond noise (>50bp on liquid names) signals a bad
  adjustment or bad bar in one feed.

## Decision → implementation
`production/data/cross_check.py`: `cross_vendor_report(primary, secondary,
return_divergence_bp=50, min_overlap=60)` — aligns overlapping (obs_date,
instrument_id) daily returns from two curated price frames, emits per-instrument
divergence stats + flagged dates; `quarantine_list(report, max_flag_frac=0.02)` —
instruments whose flagged fraction exceeds the threshold, to be excluded from the
universe until resolved (hygiene-layer integration point). Audit JSON written via
Lake.write_audit. ingest CLI gains --cross-check to run it after price ingest.

## Verification (in-repo)
Planted bad split (2:1 on one vendor only) flagged on exactly the corrupt date;
identical frames -> zero flags; quarantine triggers at the configured fraction;
mixed-calendar overlap handled (missing dates don't false-positive).

## Sources
- yfinance Price Repair docs (ranaroussi.github.io); quantmod issue #253 (split vs
  dividend adjustment mixing); PyQuantNews clean-data guide — search-level
