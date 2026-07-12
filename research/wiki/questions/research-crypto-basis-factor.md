---
type: synthesis
title: "Research: crypto basis factor data path (iteration 17)"
created: 2026-07-06
status: implemented
---

# Research: building the basis factor from free ccxt data

## Key Findings
- Iteration 10 established basis as the strongest documented cross-sectional crypto
  predictor. Data path: ccxt exposes perpetual swaps as unified `swap` markets
  (symbol form `BTC/USDT:USDT`) with the same `fetch_ohlcv` contract as spot — no
  key needed on bybit/okx public endpoints.
- Simplest robust daily basis: perp daily close vs same-exchange spot daily close,
  same midnight-UTC bar: basis_D = perp_close/spot_close − 1. Mark/index-price
  candles exist per exchange but are exchange-specific extensions; close-vs-close
  is uniform, and at daily horizon the difference is noise.
- Direction: persistent positive basis (perp premium / contango) marks crowded
  longs → negative expected return; carry logic shorts high-basis, longs low-basis.

## Decision → implementation
- New loader `ccxt_perp_basis.py` (dataset "basis"): fetches spot + swap daily bars
  per symbol (bybit primary, okx fallback), emits [obs_date, instrument_id, basis,
  +mandatory], availability = bar close + 24h (same rule as ccxt prices).
- New signal `basis_carry` (carry.py): −(trailing 7d mean basis), crypto sleeve,
  horizon 5, min_history 7 — mirrors carry_funding's shape; candidate in
  factors.yaml (n_trials bumps at gate, not registration).
- conftest gains a basis panel fixture; corruption harness corrupt-columns gains
  "basis" — the new signal is auto-covered on registration.

## Open Questions
- US-IP geo-blocks on bybit/okx public data are possible (same class as the
  binanceusdm funding issue) — loader degrades per-symbol with audit warnings;
  kraken's futures venue is a further fallback if needed at real-ingest time.

## Sources
- ccxt manual (unified swap markets, fetch_ohlcv) — docs.ccxt.com; iteration 10
  sources (BIS 1087, CMU carry paper) for the economic rationale.
