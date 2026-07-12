# Stooq secondary-feed verdict (2026-07-07 live ingest)

**Verdict: DEAD for programmatic access. Do not spend more time on stooq; the
cross-check needs a different secondary vendor.**

## Evidence

1. Historical CSV endpoint (`https://stooq.com/q/d/l/?s=aapl.us&i=d&d1=...&d2=...`)
   returns a JavaScript browser-verification interstitial — even with full Chrome
   User-Agent / Accept / Accept-Language / Referer headers. The wall is server-side
   bot protection, not header sniffing:

   ```
   <!DOCTYPE html>...<noscript>This site requires JavaScript to verify your
   browser. Please enable JavaScript and reload.</noscript>...
   ```

2. Current-quote endpoint (`https://stooq.com/q/l/?s=aapl.us&f=sd2t2ohlcv&h&e=csv`)
   returns "The page you requested does not exist or has been moved".

3. `pandas_datareader` in this venv raises `NotImplementedError: data_source='stooq'`.

4. The suggested wrapper repo **github.com/bratfizyk/stooq-api is a Haskell library**
   (not Python / no PyPI package), wraps only the current-quote endpoint (#2 above,
   itself now dead), and adds no authentication mechanism. A client wrapper cannot
   bypass a server-side JS challenge, so it does not help regardless of language.

5. Corroborating: a sibling project documented on 2026-04-25 that stooq's free CSV
   endpoint had already begun returning "Get your apikey:" prose instead of CSV.
   The lockdown has since escalated to the JS wall.

## Alternatives for the secondary price feed (cross-check unblocking)

| Vendor | Cost | Constraint |
|---|---|---|
| Tiingo | free tier w/ API key | 500 symbols/hr, 50 unique symbols/hr on free — slow for 874 names but fine as a weekly cross-check |
| Alpaca Market Data (IEX feed) | free w/ existing APCA keys | IEX-only consolidated data; owner already has Alpaca keys (not present in this machine's env) |
| Polygon flat files | paid | best quality incl. delisted |

Owner action needed: provision a Tiingo key (`TIINGO_API_KEY`) or export the APCA
keys into the environment, then wire a `TiingoPricesLoader`/`AlpacaPricesLoader`
with `available_from` stamped like the yfinance loader (post-close, later than
yfinance's 21:30 so the feed-priority algebra keeps yfinance primary — or earlier
to make the new feed primary; decide at implementation time).

`StooqPricesLoader` stays in the tree untouched (its unit tests are canned-payload
based and still pin the transform contract) in case stooq ever reopens.
