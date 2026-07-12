---
type: synthesis
title: "Research: data remediation — pre-2021 crypto, delisted equities, nowcast vintages, ETF adjustments"
created: 2026-07-07
status: researched — source/coverage/price/effort matrix filed, nothing implemented (research-only session)
related:
  - "[[sources/etf-adjustment-methodology]]"
  - "[[questions/research-rejected-factor-forensics]]"
---

# Research: closing the four data gaps from the first live-data run

## (a) Pre-2021 daily crypto OHLCV

| Source | Coverage | Price | Integration effort | Notes |
|---|---|---|---|---|
| **Coin Metrics community reference rates** | 550+ assets, FULL history at daily frequency (BTC to 2010-ish), includes many dead assets | free (HTTP API, 1000 req/10min; `format=csv`; GitHub archive `coinmetrics/data`) | ~0.5-1 day (one BaseLoader) | **Best documented depth**: transparent volume-weighted-median methodology across vetted constituent exchanges — arguably better than any single-exchange close for daily factors. Rate only (no per-exchange OHLC/volume) |
| CryptoCompare (CoinDesk) `histoday` | full daily history incl. CCCAGG aggregate; retains delisted coins | ~~free~~ **key-gated as of 2026-07-11** (keyless probe → 401) | ~1 day | needs a signup (owner action) before it can be evaluated |
| Binance public dumps (data.binance.vision) | per-pair klines from listing (SOL 2020-08, ATOM 2019-04), 1s-1mo | free, official, no key | ~1 day | **delisted-pair retention VERIFIED 2026-07-11** (BCCUSDT, delisted 2018-11, still serves all 13 monthly zips) — loader in progress as `binance_hist`; closes the newer-asset gaps CM community data cannot (its CSVs are gutted to ~7d for post-2020 assets) |
| CryptoDataDownload / Bitstamp CSVs | Bitstamp BTC from 2011; Gemini/Bitfinex archives | free, no login | ~0.5 day | pre-2017 BTC/ETH depth; per-exchange |

**Recommendation**: Coin Metrics community daily reference rates as the pre-2021
primary (documented methodology + dead-asset coverage), Bitstamp CSVs for
pre-2014 BTC depth, keep Coinbase as the post-2021 tradable-venue series. The
perp-basis start (2021-07) is a market-structure fact, not a vendor gap — perps
barely existed cross-sectionally before ~2020; no remediation exists.

> **CORRECTION (2026-07-10, live-probed during implementation)**: the community
> REST API (`community-api.coinmetrics.io/v4`) now serves only a **~7-day
> window** and silently returns `{"data": []}` when `start_time`/`end_time` are
> passed — it cannot backfill anything. The GitHub CSVs
> (`raw.githubusercontent.com/coinmetrics/data/master/csv/{asset}.csv`) are the
> free full-history route (per-asset, gzip, updated with days-to-weeks lag —
> fine for backfill, not a live feed). Also verified in-file:
> `ReferenceRate(D+1) == PriceUSD(D)` — ReferenceRate rows stamp the 00:00 UTC
> day START; use `PriceUSD(time=D)` for day-D-close alignment with the ccxt
> convention. Implemented as `--dataset cm_rates`, **backfill-only by
> construction** (emits strictly before each instrument's existing ccxt
> coverage): the "stamp availability earlier so coinbase stays primary" idea in
> the OPUS queue is impossible to do honestly under `asof_panel`'s
> latest-visible-wins tie-break, so structural non-overlap replaces it.

## (b) Delisted US equity dailies (184/874 PIT names missing)

| Source | Coverage | Price | Integration effort | Notes |
|---|---|---|---|---|
| **Norgate Platinum** | delisted + OTC-migrated + historical index constituents, to 1990 | US$630/yr (Diamond to 1950: $787.50/yr) | ~1-2 days (Windows NDU updater + `norgatedata` python reads local DB) | Best coverage-per-dollar; ALSO ships PIT index membership → independent cross-check of our Wikipedia-derived universe |
| Sharadar SEP (Nasdaq Data Link) | 21,000+ active+delisted tickers, 1998+, "nearly completely survivorship-free" | pricing gated behind NDL login (historically low-hundreds $/yr for individuals) | ~0.5 day (REST API) | API-first; shorter history than Norgate |
| Polygon | delisted metadata documented as SPOTTY (missing names/dates) | $200/mo advanced | — | NOT recommended for survivorship remediation |

**Recommendation**: Norgate Platinum — it closes the missing-prices gap AND
provides a second PIT-membership source, which doubles as a check on the
ticker-reuse quarantine class (backlog #18). Sharadar if a Windows updater
dependency is unacceptable.

## (c) Cleveland Fed inflation-nowcast VINTAGE history (backlog #16)

- Daily CPI/PCE nowcasts published each business day ~10:00 ET since early 2014.
- **No public as-published vintage archive was found** (the nowcasting page
  serves the current-quarter table; direct fetch 403'd; searches surface no
  archive file). Third parties (MacroMicro) chart a historical nowcast series,
  implying a recorded history exists outside the Fed, but their recording
  timestamps are undocumented — usable only with a caveat flag.
- The Cleveland Fed's own Economic Commentary 2023-06 ("A Real-Time Assessment of
  Inflation Nowcasting") evaluates as-published accuracy — an internal vintage
  archive exists.

**Recommendation (three tracks, cheapest first)**:
1. Start a daily 10:05-ET archiver NOW (cron + snapshot to the lake with
   `available_from` = capture time). Zero cost, PIT-perfect going forward. ~0.5 day.
2. Email Cleveland Fed research for the EC-2023-06 replication/vintage dataset —
   the only route to a PIT-clean backfill. (Owning-session action; free.)
3. If (2) fails, MacroMicro-recorded history as approximate vintages with an
   explicit `vintage_quality=approx` flag — acceptable for the mechanism-#2
   drift-regression DIAGNOSTIC, not for a gated factor.
   Note: forward-archiving alone delays the backlog-#16 pre-registered test by
   ~6-12 months of accumulation.

## (d) Cross-vendor distribution adjustments (bond/FX ETF quarantine class)

Fully diagnosed in [[sources/etf-adjustment-methodology]]: four documented
mechanisms (adjustment-anchor choice, factor rounding, retrieval-date-dependent
backward adjustment, proportional-vs-absolute). The month-end ~0-correlation
signature of our 66-instrument quarantine's FX/bond-ETF class is the textbook
symptom. **The instruments are healthy; the comparison of vendor adjusted closes
was the bug.** Practitioner normalization: cross-check unadjusted closes +
distribution-event tables separately; build total-return in-house with one
convention. Measurement-layer change in the vendor cross-check, ~1 day, owning
session. (The ticker-reuse subclass — APC/MI — is a different, real bug class
already handled by the hygiene blocklist.)

## Falsifiable checks

- CM community vs Coinbase daily closes on the overlap (2021+): correlation of
  log returns > 0.995 for BTC/ETH — validates CM as pre-2021 primary. If not,
  the constituent-market mix diverges from our tradable venue and CM should be
  demoted to cross-check-only.
- Norgate PIT index membership vs our Wikipedia-derived membership: disagreement
  rate < 2% of name-months; every disagreement resolvable to a documented
  corporate action. Larger → our PIT universe has a systematic bias.
- Re-running the vendor cross-check on UNADJUSTED closes should clear ≥ the
  FX/bond-ETF majority of the 66 quarantined instruments; those that remain
  discrepant are genuine data faults.

## Open Questions

- Binance data.binance.vision delisted-pair retention (README silent) — verify
  empirically by requesting a known-delisted symbol's kline archive.
- Sharadar SEP exact current pricing (gated behind Nasdaq Data Link login).
- Whether Cleveland Fed will share the vintage archive (track 2) — single
  highest-leverage unknown for backlog #16.

## Sources

- [[sources/etf-adjustment-methodology]] — FirstRate Data, Portfolio Optimizer,
  Alvarez Quant Trading, EODHD/Koyfin
- Coin Metrics community docs (gitbook-docs.coinmetrics.io; github.com/coinmetrics/data)
- Norgate prices/content pages (norgatedata.com/prices.php, data-content-tables.php)
- Sharadar SEP (data.nasdaq.com/databases/SEP); QuantRocket Sharadar docs
- Polygon knowledge base ("What does Massive do with delisted tickers?") — spotty
- Cleveland Fed inflation-nowcasting page + Economic Commentary 2023-06
