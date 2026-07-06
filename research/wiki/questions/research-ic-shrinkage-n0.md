---
type: synthesis
title: "Research: IC shrinkage strength n0 (iteration 11)"
created: 2026-07-06
status: researched — DEFERRED to real-data phase
---

# Research: is n0=126 the right IC shrinkage strength?

## Key Findings
- Search surfaced no direct literature on IC-specific shrinkage constants; the
  principled frame is empirical Bayes / James-Stein: IC* = IC · n/(n+n0) is the
  posterior mean under a zero-centered prior, and the "right" n0 equals
  (within-factor IC sampling variance) / (between-factor variance of true ICs).
- Both variance components are estimable from OUR OWN ledger once real data flows:
  per-date IC dispersion (within) and cross-factor spread of long-run mean ICs
  (between). On synthetic GBM data the between-factor variance is zero by
  construction — n0 would blow up to infinity, correctly implying "shrink
  everything to zero", which is degenerate for testing.

## Verdict
DEFERRED: n0=126 stays (a defensible half-year prior). Revisit with an
empirical-Bayes estimator once >= 6 months of real IC history exists in
panels/ic/ — the estimator belongs in ic_monitor territory at that point.
Filed so the loop doesn't re-open this without new data.

## Sources
- Empirical-Bayes/James-Stein framing (standard); Grinold & Kahn forecasting
  chapter (conceptual). No usable direct sources at search level.
