---
type: synthesis
title: "Research: multi-vintage dedup gap in macro consumers (iteration 16, internal audit)"
created: 2026-07-06
status: implemented
---

# Internal audit: asof_panel is never called in the run paths

## Finding
`core/pit.py:asof_panel` (the "only sanctioned join") is exercised by its own tests
and nothing else. Signals handle availability themselves (merge_asof on availability
date), which the corruption harness validates — and for single-vintage sources that
is equivalent. The hole: **multi-vintage ALFRED macro series**. The curated frame
legitimately carries several rows per (obs_date, series_id) (original + revisions).
- Overlay `macro_derisk_multiplier` filters `available_from <= t` then aligns on
  obs_date — duplicate obs_dates from superseded vintages inflate/distort the
  causal z window.
- Signals over macro (carry_rate_diff) ffill by availability date, so a later
  vintage naturally overrides from its own availability onward — right semantics,
  but superseded rows can still double-count in window stats (cot z windows).
- The synthetic fixtures are single-vintage, so no existing test can catch this.

## Decision → implementation
- Overlay macro path: dedup to the latest visible vintage per (obs_date, series)
  via `asof_panel(macro, t)` before any z computation — the sanctioned join, used
  where it was always meant to be.
- Multi-vintage regression tests: macro frame with a revised observation —
  (a) no duplicated obs in the z window (multiplier equals the single-vintage
  equivalent built from the visible-vintage rows); (b) a revision published after t
  is invisible at t (vintage corruption test); (c) carry_rate_diff value at D uses
  the vintage visible at D.
- Wiki pattern note: third wiring gap found (sectors, hygiene, now PIT dedup) —
  every "capability exists" claim needs a call-site test, not just a unit test.

## Sources
Internal audit; CLAUDE.md PIT rules; ALFRED vintage semantics (iteration-0 design).
