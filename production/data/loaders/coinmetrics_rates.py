"""Coin Metrics community CSVs — pre-ccxt crypto price depth (backfill-only).

Backfills the depth gap the tradable-venue feed cannot reach: coinbase/kraken
(behind :mod:`production.data.loaders.ccxt_prices`) list most of this universe
only from ~2016-2021 on, while Coin Metrics' community data carries daily USD
prices with full history (BTC back to 2009) and dead-asset coverage.
Recommendation history: research/wiki/questions/research-data-remediation.md (a).

SOURCE CHOICE (probed live 2026-07-10 — supersedes the research page's claim):
the community REST API (``community-api.coinmetrics.io/v4``) now serves only a
~7-day window regardless of parameters, and SILENTLY returns ``{"data": []}``
when ``start_time``/``end_time`` are passed — it cannot backfill anything. The
free full-history artifact is the community GitHub repo::

    https://raw.githubusercontent.com/coinmetrics/data/master/csv/{asset}.csv

one CSV per asset id (lowercase), gzip-served, updated by CM with a lag of days
to weeks (btc's tail was 2026-05-24 on the probe date) — fine for a historical
backfill, NOT a live feed.

DAY-BOUNDARY SEMANTICS (verified in-file, not assumed): the CSV shows
``ReferenceRate(D+1) == PriceUSD(D)`` exactly — CM's ReferenceRate rows are
stamped at the 00:00 UTC day START (= the previous day's close), while
``PriceUSD(time=D)`` is day D's close measured at D+1 00:00 UTC. ``PriceUSD``
therefore aligns 1:1 with our ccxt convention (obs_date D = day-D close) and is
the column used; ReferenceRateUSD also starts years later on most assets.

BACKFILL-ONLY BY CONSTRUCTION (the feed-priority answer): the sanctioned PIT
read path breaks (obs_date, instrument_id) ties by LATEST visible
``available_from`` (production/core/pit.py — sort by [available_from,
ingested_at], keep last). No honest CM stamp can be both >= the close time
(obs+24h, when the value first exists) and earlier than ccxt's obs+24h stamp,
so priority CANNOT be encoded in the availability algebra without claiming the
value knowable before it exists. Instead of competing, this loader never emits
a row the ccxt feed already covers: ``transform`` reads the lake's existing
crypto prices and, per instrument, keeps only ``obs_date`` strictly before that
instrument's earliest ccxt row (minus 1 day). Instruments with no ccxt coverage
at all (dead assets) get their full history. Consequence: run the ccxt prices
ingest BEFORE this backfill — an empty ccxt slice makes this loader emit
nothing rather than risk displacing the tradable-venue series, and it warns
loudly in that case.

Availability: ``obs_date + 48h`` — the value exists at obs+24h and the repo
publishes with at least a further day of lag; 48h is a conservative, honest
floor for when THIS artifact was actually fetchable. Since emitted keys never
overlap ccxt's, the stamp never competes with the primary feed.

Dataset/source convention: like tiingo/alpaca, a SECONDARY feed sharing the
``prices`` curated dataset, distinguished by ``source="coinmetrics:community-csv"``.
Shared-watermark dataset: a first backfill needs ``--full`` (OPUS.md traps).
"""
from __future__ import annotations

import io
import time

import numpy as np
import pandas as pd

from production.data.base import (AvailabilityRule, BaseLoader, IngestError,
                                  asset_class_from_instrument_id)
from production.reference.hygiene import apply_hygiene

CSV_URL_TEMPLATE = "https://raw.githubusercontent.com/coinmetrics/data/master/csv/{asset}.csv"
PRICE_COLUMN = "PriceUSD"


class CoinMetricsRatesLoader(BaseLoader):
    dataset = "prices"
    vendor = "coinmetrics"
    source = "coinmetrics:community-csv"
    asset_classes = ["crypto"]
    default_asset_class = "crypto"
    # Value exists at obs+24h (day close); the GitHub artifact publishes with >=
    # another day of lag. 48h is honest and, because emitted keys never overlap
    # the ccxt feed's (see module docstring), never competes for priority.
    availability_rule = AvailabilityRule("obs_offset", {"offset": pd.Timedelta(hours=48)})
    expectations = {
        "columns": ["close", "volume", "dollar_volume"],
        "ranges": {"close": (0, None)},
        # volume/dollar_volume are ~100% null by construction (the community CSV
        # price columns carry no traded-volume figure).
        "max_null_frac": 0.99,
        "min_rows": 1,
    }

    def __init__(self, lake=None, instruments=None, assets=None, pause_s=0.5):
        super().__init__(lake, instruments)
        # CM asset ids are lowercase on the wire; accept either casing in.
        self.assets = [str(a).lower() for a in (assets or [])]
        self.pause_s = pause_s

    # ------------------------------------------------------------------ fetch
    def fetch(self, start, end) -> dict[str, str]:
        """Return ``{asset_id: csv_text}`` — one GitHub CSV per asset.

        Per-asset failures (no CSV for the id, transient error) warn and skip;
        if configured with no assets at all, that is a wiring bug and raises.
        """
        import requests

        if not self.assets:
            raise IngestError(
                "coinmetrics: no assets configured — pass assets=[...] (the crypto "
                "sleeve symbols; see scripts/ingest.py's cm_rates wiring).")

        out: dict[str, str] = {}
        for asset in self.assets:
            time.sleep(self.pause_s)     # raw.githubusercontent is tolerant; stay polite
            try:
                resp = requests.get(CSV_URL_TEMPLATE.format(asset=asset), timeout=120)
                resp.raise_for_status()
                out[asset] = resp.text
            except Exception as exc:
                self.warnings.append(f"coinmetrics: fetch failed for {asset!r}: {exc!r:.200}")
        return out

    # -------------------------------------------------------------- transform
    def _ccxt_coverage_start(self) -> pd.Series:
        """Earliest existing lake prices/crypto obs_date per instrument_id.

        This is what makes the loader backfill-only: emitted rows stop strictly
        before each instrument's existing coverage. Missing dataset -> empty
        Series (and transform warns + emits nothing for covered-universe safety).
        """
        try:
            cur = self.lake.read_curated("prices", "crypto")
        except Exception:
            return pd.Series(dtype="datetime64[ns]")
        if cur is None or cur.empty:
            return pd.Series(dtype="datetime64[ns]")
        return pd.to_datetime(cur.groupby("instrument_id")["obs_date"].min())

    def transform(self, raw) -> pd.DataFrame:
        cols = ["obs_date", "instrument_id", "close", "volume", "dollar_volume",
                "asset_class"]
        coverage = self._ccxt_coverage_start()
        if coverage.empty:
            # No existing crypto prices to bound the backfill: emitting anything
            # could later collide with (and, under latest-vintage tie-break,
            # displace) the tradable-venue series. Refuse loudly instead.
            self.warnings.append(
                "coinmetrics: no existing prices/crypto coverage found in the lake — "
                "run the ccxt prices ingest first; emitting NOTHING (backfill-only "
                "contract, see module docstring)")
            return pd.DataFrame(columns=cols)

        frames = []
        for asset, text in (raw or {}).items():
            iid = self.resolve(asset.upper(), "coinmetrics")
            if iid is None:
                continue
            try:
                df = pd.read_csv(io.StringIO(text), usecols=lambda c: c in ("time", PRICE_COLUMN),
                                 low_memory=False)
            except (ValueError, pd.errors.ParserError) as exc:
                self.warnings.append(f"coinmetrics: unparseable CSV for {asset!r}: {exc!r:.120}")
                continue
            if "time" not in df.columns or PRICE_COLUMN not in df.columns:
                self.warnings.append(
                    f"coinmetrics: {asset!r} CSV lacks time/{PRICE_COLUMN} — skipped")
                continue
            obs = pd.to_datetime(df["time"], errors="coerce")
            close = pd.to_numeric(df[PRICE_COLUMN], errors="coerce")
            f = pd.DataFrame({"obs_date": obs, "instrument_id": iid, "close": close})
            f = f.dropna(subset=["obs_date", "close"])

            cutoff = coverage.get(iid)
            if cutoff is not None and pd.notna(cutoff):
                f = f[f["obs_date"] < cutoff - pd.Timedelta(days=1)]
            # no ccxt coverage for this instrument (e.g. a dead asset): keep all
            if f.empty:
                continue
            f["volume"] = np.nan       # no traded-volume figure in the community CSV
            f["dollar_volume"] = np.nan
            f["asset_class"] = asset_class_from_instrument_id(iid)
            frames.append(f)

        if not frames:
            return pd.DataFrame(columns=cols)
        long = pd.concat(frames, ignore_index=True)
        return self._apply_price_hygiene(long)

    def _apply_price_hygiene(self, long: pd.DataFrame) -> pd.DataFrame:
        """Same backstop + blocklist enforcement as the other price loaders."""
        if long is not None and not long.empty and "symbol" not in long.columns \
                and "instrument_id" in long.columns:
            long = long.copy()
            long["symbol"] = long["instrument_id"].map(
                lambda i: str(i).split(":")[1] if pd.notna(i) and ":" in str(i) else i)
        clean, drops = apply_hygiene(long, symbol_col="symbol")
        clean = clean.drop(columns=["symbol"], errors="ignore")
        self.hygiene_drops = drops
        if any(drops.values()):
            self.warnings.append(
                f"hygiene: dropped {drops['backstop_dropped']} sub-floor price row(s), "
                f"{drops.get('flap_dropped', 0)} flap-outlier row(s), "
                f"and {drops['blocklist_dropped']} blocklisted-ticker row(s)")
        return clean
