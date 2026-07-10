---
type: source
title: "Kalshi historical data access — endpoint archaeology"
source_type: api-documentation + live probes
date_published: 2026-07-10 (probed)
url: https://docs.kalshi.com/api-reference/historical/get-historical-markets
confidence: high (every claim probed live 2026-07-10)
key_claims:
  - public v2 /markets only serves RECENT settled markets (~2 months)
  - full archive lives on external-api.kalshi.com /historical/* (no auth)
  - legacy events are aliased under KX-prefixed series names post-migration
  - archived markets have NO candlesticks; /historical/trades is the price history
---

# Kalshi historical data access (probed 2026-07-10)

What it took to reach the 2022+ settled-market history for the backfill
(backlog #16). Every claim below was verified with live probes; recorded so the
next data problem doesn't re-run the archaeology.

## The endpoint map

| need | endpoint | notes |
|---|---|---|
| settled markets, recent (~2mo) | `api.elections.kalshi.com/trade-api/v2/markets?series_ticker=&status=settled` | cursor, limit<=1000; returns status "finalized" |
| settled markets, FULL archive | `external-api.kalshi.com/trade-api/v2/historical/markets?series_ticker=` | no auth; KXCPIYOY reaches 2022-12; ~2-3 month archival lag |
| trades, archived | `external-api.kalshi.com/trade-api/v2/historical/trades?ticker=` | works for legacy markets; price/count/taker_side per print |
| trades, recent | `api.elections.kalshi.com/trade-api/v2/markets/trades?ticker=` | fallback when archive hasn't caught up |
| candlesticks (ACTIVE markets only) | `.../series/{series}/markets/{ticker}/candlesticks` | the old unscoped path 404s since ~2026-07; archived markets 404 on every host/path combo |
| series list | `.../series/?category=Economics` | 607 series; legacy + KX duplicates both listed |
| events incl. legacy | `.../events?series_ticker=KX*&status=settled` | returns legacy-named events (CPIYOY-22NOV) under the KX series |

## Traps
- **Query by the KX series name only.** `series_ticker=CPIYOY` returns nothing
  anywhere; the 2025 ticker migration aliased legacy events under `KXCPIYOY`.
- **`/events/{legacy}?with_nested_markets=true` returns zero markets** even though
  the event exists — market records for legacy events are only on the historical
  host.
- **Schema generation shift**: prices are now dollar-strings (`"0.0300"`), sizes
  float-strings (`"68.00"`, `*_fp` suffix); the older integer-cents shape still
  appears in cached/legacy contexts. Parse both (the live loader now does).
- **`expiration_value` on settled markets is the actual released print** (e.g.
  "3.00" for June-2023 CPI YoY) — a free ground-truth column for macro diagnostics.
- Daily candles anchor at 04:00 UTC (midnight ET), not 00:00 UTC.

## Fees (for net-of-fee tests)
Taker: ceil-to-cent(0.07 × C × P × (1−P)) dollars per fill; maker = 25% of taker
([fee schedule](https://kalshi.com/fee-schedule) effective 2026-07-07). Whelan
imputation: per-contract fee from a 100-lot → 1.75c at 50c, 0.34c at 5c.

## Third-party alternatives (not needed, recorded)
Lychee Data (36GB paid dump), kalshibacktest.com (freemium API), pmxt.dev
(hourly orderbook parquet snapshots, free) — all superseded by the official
historical host for our use.

## Used by
[[questions/research-kalshi-mechanism-diagnostics]] — the backfill loader
(`production/data/loaders/kalshi_history.py`, dataset `event_markets_hist`).
