---
type: synthesis
title: "Research: TCA feedback — shortfall calibrates the cost model (iteration 20)"
created: 2026-07-06
status: implemented
---

# Research: closing the pre-trade/post-trade cost loop

## Key Findings
- Standard TCA practice (Perold framework): pre-trade estimates vs realized
  implementation shortfall, decomposed into spread/impact/delay; the post-trade
  numbers exist to RECALIBRATE the pre-trade model — a one-way estimate that never
  learns from fills stays optimistic forever (our alpha=0.15 concern, iteration 4).
- Our pieces already exist: cost model with per-instrument overrides
  (configs/costs.yaml instrument_overrides), shortfall measurement
  (execution/shortfall.py), paper fills incoming via Alpaca. Nothing connects them.

## Decision → implementation
- New `production/execution/tca.py`:
  `calibrate_overrides(shortfall_df, min_fills=20, safety=1.25) -> dict` — per
  instrument, when >= min_fills fills exist, realized median |shortfall bps| x
  safety becomes the half_spread_bps override candidate; overrides only ever
  RAISE the effective cost above the sleeve default (floors are floors — never
  lowered; CLAUDE.md), and are capped at cost cap.
  `write_overrides(overrides, lake)` / `read_overrides(lake)` — a lake reference
  table (cost_overrides.parquet) with stamped calibration date + fill counts.
- CostModel: constructor accepts overrides_table (dict) merged OVER yaml overrides
  (lake-calibrated wins); engine/daily_run read it from the lake when present.
- Verification: calibration arithmetic known-answer; never-lower property (an
  instrument with tiny realized shortfall keeps the sleeve default); min_fills
  respected; roundtrip through the lake; CostModel picks up the table.

## Sources
- Perold implementation-shortfall framework (CFA curriculum treatments); TCA
  pre/post-trade calibration practice — search-level.
