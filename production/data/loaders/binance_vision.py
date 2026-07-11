"""Binance's public bulk archive (data.binance.vision) — crypto daily-price backfill.

Closes the pre-coinbase depth gap that :mod:`coinmetrics_rates` could NOT reach:
sol/avax/atom/fil/near/grt/matic/shib have no ``PriceUSD`` in Coin Metrics'
community CSVs and their ``ReferenceRate*`` columns carry only the last ~7 days
(coinmetrics_rates.py docstring). Binance listed most of these pairs at launch, so
the venue's own archive is full-depth. It also carries DELISTED pairs (their zips
stay published after the pair is pulled from the live API), giving dead-asset
coverage the live ccxt feed structurally cannot.

ARCHIVE SHAPE (live-probed 2026-07-11, orchestrator + independently re-verified in
this build session)::

    listing: GET https://s3-ap-northeast-1.amazonaws.com/data.binance.vision
             ?list-type=2&delimiter=/&prefix=data/spot/monthly/klines/{PAIR}/1d/
    zip:     GET https://data.binance.vision/{key}

``{PAIR}`` = ``{SYMBOL}USDT`` (Binance's spot-quote convention; USDT is treated as
USD for our daily-close purposes — basis is <10bp except during depegs, an
acceptable approximation for a historical backfill, NOT for anything basis-trading
related). The listing XML interleaves each month's ``...zip`` key with a sibling
``...zip.CHECKSUM`` key; only ``.zip`` keys are kept. Re-verified live: BTCUSDT's
FULL history (2017-08 through 2026-06, 107 months = 214 keys including checksums)
still comes back with ``IsTruncated=false`` against the default ``MaxKeys=1000`` —
a single pair's monthly-1d listing has never been observed to need a second page —
but continuation-token pagination is implemented anyway rather than assumed. A
pair with zero archived months (never listed, or listed only against a different
quote) returns ``KeyCount=0`` with HTTP 200 (empty, not an error); a missing ZIP
KEY (bad month, bad pair) 404s. Both degrade the same way: warn and move on.

Each zip holds exactly one CSV, klines schema (12 columns, comma-separated, no
column names on the wire in every file this loader has ever fetched — verified on
both a 2020-08 and a 2026-06 BTCUSDT/SOLUSDT monthly file)::

    open_time, open, high, low, close, volume, close_time, quote_asset_volume,
    n_trades, taker_buy_base, taker_buy_quote, ignore

A header row is tolerated defensively per the build brief anyway: row 0 is dropped
whenever its ``open_time`` field fails numeric coercion (a header's first field
never parses as a number; a real row's always does), whatever the header's actual
column names turn out to be on some future file.

TIMESTAMP-PRECISION MIGRATION (found live during THIS build, 2026-07-11 — not
something the loader could have been briefed on in advance): older monthly files
carry a 13-digit MILLISECOND ``open_time``/``close_time`` (e.g. BTCUSDT 2020-08:
``1597104000000`` -> 2020-08-11 00:00:00 UTC); a CURRENT file carries a 16-digit
MICROSECOND epoch instead (BTCUSDT 2026-06: ``1780272000000000`` -> 2026-06-01
00:00:00 UTC, probed the same session) — every other field (price, volume, trade
count) is unaffected, only the two timestamp columns changed scale. Millisecond
values stay under 1e14 for any realistic date (1e13 ms ~= year 2286); live
microsecond values are ~1.7-1.8e15, so per-row magnitude alone disambiguates with
no file-level flag needed (``_open_time_to_obs_date``). This matters here
specifically because dead-asset rows (no ccxt coverage at all -> full history kept,
see BACKFILL-ONLY below) can run all the way to the present and would silently
mis-date every row by ~20 days if the loader assumed millisecond-only.

DAY-BOUNDARY SEMANTICS (verified in the same probe): ``open_time`` is exactly
00:00:00.000 UTC of day D and ``close_time`` is exactly 23:59:59.999 UTC of day D
— so ``obs_date = UTC date of open_time = day D`` IS day D's close under our ccxt
convention, with no shift needed (unlike coinmetrics_rates's ReferenceRate case,
which is day-START-stamped and needs a -1d correction). ``close`` = the kline close
price; ``volume`` = base-asset volume; ``dollar_volume`` = quote_asset_volume
(USDT-denominated, treated as dollars per the approximation above).

CURRENT-MONTH SKIP: this loader never touches the ``data/spot/daily/klines/...``
path. The live ccxt feed already covers recent dates (that is the entire point of
the backfill-only design below — this loader stops before ccxt's coverage begins),
so the archive's most recent, still-incomplete month buys nothing and is skipped
entirely to keep fetch() simple. Only ``data/spot/monthly/klines/...`` is read.

BACKFILL-ONLY BY CONSTRUCTION: identical rationale and mechanism to
:class:`~production.data.loaders.coinmetrics_rates.CoinMetricsRatesLoader` — no
honest availability stamp for THIS artifact can ever be earlier than ccxt's
obs+24h stamp without claiming the value knowable before it existed, so priority
cannot be encoded in the availability algebra. Instead, ``transform`` reads the
lake's existing ccxt-SOURCED ``prices``/``crypto`` rows and, per instrument, keeps
only ``obs_date`` strictly before that instrument's earliest ccxt row (minus 1
day). The ``source`` filter on the coverage floor is load-bearing for the exact
ratchet-bug reason documented in coinmetrics_rates.py (a prior binance:vision
backfill's own earlier rows must never pull the floor backward on re-run).
Instruments with no ccxt coverage at all (fully dead pairs, e.g. BCCUSDT) get
their full archived history. An empty lake -> emit nothing and warn loudly; run
the ccxt prices ingest first.

Availability: ``obs_date + 35 days``. The monthly zip for month M is published by
Binance "a few days" after M closes (probed behavior, not a documented SLA); a
day-1-of-M row is the worst case — it isn't public until the WHOLE month's zip
lands, i.e. up to ~30 days of the month plus a several-day publication lag. 35
days is a conservative, honest ceiling for that worst case, the same
trade-off :class:`CoinMetricsRatesLoader` makes with its own obs+48h stamp. Since
emitted keys never overlap ccxt's (backfill-only, above), the stamp never competes
for priority under the latest-vintage tie-break.

Hygiene: NO sub-$0.10 price floor, deliberately, matching ccxt_prices and
coinmetrics_rates (hygiene.py's floor is an EQUITY-universe device; SHIB-scale
crypto prices are legitimate and already live in the lake under ccxt). Ticker
reuse cannot occur here either: instrument ids resolve through the PIT instrument
master, never through raw vendor symbols.

Dataset/source convention: a THIRD secondary feed sharing the ``prices`` curated
dataset alongside ccxt (primary) and coinmetrics:community-csv (secondary),
distinguished by ``source="binance:vision"``. Shared-watermark dataset: a first
backfill needs ``--full`` (OPUS.md traps) — see coinmetrics_rates.py's identical
note; ``fetch`` ignores ``start``/``end`` entirely for the same reason (the
archive listing has no date-range parameter and returns full history per pair;
``transform``'s ccxt-coverage cutoff is what actually bounds the emitted rows).
"""
from __future__ import annotations

import io
import time
import xml.etree.ElementTree as ET
import zipfile

import pandas as pd

from production.data.base import (AvailabilityRule, BaseLoader, IngestError,
                                  asset_class_from_instrument_id)

LISTING_URL = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
ZIP_BASE_URL = "https://data.binance.vision"
_S3_NS = "http://s3.amazonaws.com/doc/2006-03-01/"

# Wire order of the no-header klines CSV (module docstring).
KLINE_COLUMNS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
                 "quote_asset_volume", "n_trades", "taker_buy_base", "taker_buy_quote",
                 "ignore"]

# ms values stay below this for any realistic date (1e13 ms ~= year 2286); live
# microsecond epoch values are ~1.7-1.8e15 (module docstring: TIMESTAMP-PRECISION
# MIGRATION). Anything at or above this magnitude is treated as microseconds.
_MICROSECOND_THRESHOLD = 1e14


def _parse_klines_csv(text: str) -> pd.DataFrame | None:
    """One monthly zip's CSV -> DataFrame with :data:`KLINE_COLUMNS`, header dropped.

    No file this loader has fetched carries a header row, but one is tolerated
    defensively: row 0 is dropped whenever its ``open_time`` fails numeric
    coercion (a header's first field never parses as a number; a real data row's
    always does), regardless of what the header's literal column names are.
    """
    try:
        df = pd.read_csv(io.StringIO(text), header=None, names=KLINE_COLUMNS)
    except (ValueError, pd.errors.ParserError):
        return None
    if df.empty:
        return df
    if pd.isna(pd.to_numeric(df.iloc[0]["open_time"], errors="coerce")):
        df = df.iloc[1:].reset_index(drop=True)
    return df


def _open_time_to_obs_date(raw: pd.Series) -> pd.Series:
    """UTC calendar day of kline ``open_time`` (module docstring: DAY-BOUNDARY
    SEMANTICS + TIMESTAMP-PRECISION MIGRATION for the ms/us disambiguation)."""
    ms = pd.to_numeric(raw, errors="coerce")
    ms = ms.where(ms < _MICROSECOND_THRESHOLD, ms / 1000.0)
    return pd.to_datetime(ms, unit="ms", errors="coerce").dt.normalize()


class BinanceVisionLoader(BaseLoader):
    dataset = "prices"
    vendor = "binance"
    source = "binance:vision"
    asset_classes = ["crypto"]
    default_asset_class = "crypto"
    # obs_date + 35 days (module docstring: worst-case monthly-archive publication
    # lag). Emitted rows never overlap ccxt's, so the stamp never competes for
    # priority under the latest-vintage tie-break.
    availability_rule = AvailabilityRule("obs_offset", {"offset": pd.Timedelta(days=35)})
    expectations = {
        "columns": ["close", "volume", "dollar_volume"],
        "ranges": {"close": (0, None), "volume": (0, None)},
        "max_null_frac": 0.02,
        "min_rows": 1,
    }

    def __init__(self, lake=None, instruments=None, assets=None, quote="USDT",
                 pause_s=0.25, max_months_per_asset=240):
        super().__init__(lake, instruments)
        self.assets = [str(a).upper() for a in (assets or [])]
        self.quote = str(quote).upper()
        self.pause_s = pause_s
        self.max_months_per_asset = max_months_per_asset

    # ------------------------------------------------------------------ fetch
    def _list_month_keys(self, session, pair: str) -> list[str]:
        """Monthly 1d kline zip keys for ``pair``, oldest first.

        Continuation-token pagination is implemented defensively; live-probed
        2026-07-11, even BTCUSDT's full ~9-year history (107 months, 214 keys
        including the sibling ``.CHECKSUM`` entries) came back on one page
        against the default ``MaxKeys=1000`` — a second page has never been
        observed for a single pair's monthly-1d listing (module docstring).
        """
        prefix = f"data/spot/monthly/klines/{pair}/1d/"
        keys: list[str] = []
        token = None
        for _ in range(20):  # hard cap: a pathological loop must not run forever
            params = {"list-type": "2", "delimiter": "/", "prefix": prefix}
            if token:
                params["continuation-token"] = token
            resp = session.get(LISTING_URL, params=params, timeout=60)
            resp.raise_for_status()
            root = ET.fromstring(resp.text)
            for c in root.findall(f"{{{_S3_NS}}}Contents"):
                key = c.findtext(f"{{{_S3_NS}}}Key") or ""
                if key.endswith(".zip"):          # excludes the sibling .CHECKSUM key
                    keys.append(key)
            truncated = (root.findtext(f"{{{_S3_NS}}}IsTruncated") or "false").strip().lower()
            if truncated != "true":
                break
            token = root.findtext(f"{{{_S3_NS}}}NextContinuationToken")
            if not token:
                break
        return sorted(keys)[: self.max_months_per_asset]

    def fetch(self, start, end) -> dict[str, list[str]]:
        """Return ``{asset: [csv_text, ...]}``, one CSV per monthly zip, oldest first.

        ``start``/``end`` are ignored (module docstring: the archive has no
        date-range listing parameter; ``transform``'s ccxt-coverage cutoff is what
        actually bounds the emitted rows). Per-asset failures — a listing error, a
        pair with zero archived months, or every zip in a pair's history failing
        to download — warn and skip rather than aborting the whole run; a single
        month's zip 404ing (bad key) also just warns and the rest of that asset's
        months still come through.
        """
        import requests

        if not self.assets:
            raise IngestError(
                "binance-vision: no assets configured — pass assets=[...] (the "
                "crypto sleeve symbols; see scripts/ingest.py's binance_hist wiring).")

        session = requests.Session()
        out: dict[str, list[str]] = {}
        for asset in self.assets:
            pair = f"{asset}{self.quote}"
            time.sleep(self.pause_s)
            try:
                keys = self._list_month_keys(session, pair)
            except Exception as exc:
                self.warnings.append(
                    f"binance-vision: listing failed for {pair!r}: {exc!r:.200}")
                continue
            if not keys:
                self.warnings.append(f"binance-vision: no monthly 1d history for {pair!r}")
                continue

            texts: list[str] = []
            for key in keys:
                time.sleep(self.pause_s)
                try:
                    resp = session.get(f"{ZIP_BASE_URL}/{key}", timeout=60)
                    resp.raise_for_status()
                    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                        member = zf.namelist()[0]
                        texts.append(zf.read(member).decode("utf-8"))
                except Exception as exc:
                    self.warnings.append(
                        f"binance-vision: download failed for {key!r}: {exc!r:.200}")
            if texts:
                out[asset] = texts
            else:
                self.warnings.append(
                    f"binance-vision: all monthly downloads failed for {pair!r}")
        return out

    # -------------------------------------------------------------- transform
    def _ccxt_coverage_start(self) -> pd.Series:
        """Earliest CCXT-SOURCED lake prices/crypto obs_date per instrument_id.

        Mirrors CoinMetricsRatesLoader._ccxt_coverage_start exactly (same
        ratchet-bug rationale: the ``source`` filter is load-bearing, or a prior
        binance:vision backfill's own rows would pull the floor backward on every
        re-run — see coinmetrics_rates.py's module docstring for the incident).
        """
        try:
            cur = self.lake.read_curated("prices", "crypto")
        except Exception:
            return pd.Series(dtype="datetime64[ns]")
        if cur is None or cur.empty or "source" not in cur.columns:
            return pd.Series(dtype="datetime64[ns]")
        ccxt = cur[cur["source"].astype(str).str.contains("ccxt")]
        if ccxt.empty:
            return pd.Series(dtype="datetime64[ns]")
        return pd.to_datetime(ccxt.groupby("instrument_id")["obs_date"].min())

    def transform(self, raw) -> pd.DataFrame:
        cols = ["obs_date", "instrument_id", "close", "volume", "dollar_volume",
                "asset_class"]
        coverage = self._ccxt_coverage_start()
        if coverage.empty:
            # Same refusal as CoinMetricsRatesLoader: emitting anything here could
            # later collide with (and, under latest-vintage tie-break, displace)
            # the tradable-venue series. Refuse loudly instead.
            self.warnings.append(
                "binance-vision: no existing prices/crypto coverage found in the "
                "lake — run the ccxt prices ingest first; emitting NOTHING "
                "(backfill-only contract, see module docstring)")
            return pd.DataFrame(columns=cols)

        frames = []
        for asset, texts in (raw or {}).items():
            iid = self.resolve(asset, "binance")
            if iid is None:
                continue
            parsed = [d for d in (_parse_klines_csv(t) for t in (texts or []))
                     if d is not None and not d.empty]
            if not parsed:
                continue
            klines = pd.concat(parsed, ignore_index=True)

            f = pd.DataFrame({
                "obs_date": _open_time_to_obs_date(klines["open_time"]),
                "instrument_id": iid,
                "close": pd.to_numeric(klines["close"], errors="coerce"),
                "volume": pd.to_numeric(klines["volume"], errors="coerce"),
                "dollar_volume": pd.to_numeric(klines["quote_asset_volume"], errors="coerce"),
            })
            f = f.dropna(subset=["obs_date", "close"])
            # Incremental re-runs can re-download an overlapping month; keep the
            # last-seen row per day (harmless — the lake's own upsert dedupes on
            # (obs_date, instrument_id, available_from) again downstream anyway).
            f = f.drop_duplicates(subset=["obs_date"], keep="last")

            cutoff = coverage.get(iid)
            if cutoff is not None and pd.notna(cutoff):
                f = f[f["obs_date"] < cutoff - pd.Timedelta(days=1)]
            # no ccxt coverage for this instrument (e.g. a delisted pair): keep all
            if f.empty:
                continue
            f["asset_class"] = asset_class_from_instrument_id(iid)
            frames.append(f)

        if not frames:
            return pd.DataFrame(columns=cols)
        # NO hygiene backstop here, deliberately (module docstring): the sub-$0.10
        # price floor is an equity-universe device, not applied by ccxt_prices or
        # coinmetrics_rates either, and ticker reuse cannot occur (ids resolve
        # through the PIT instrument master).
        return pd.concat(frames, ignore_index=True)
