"""Kalshi prediction-market loader (curated dataset ``event_markets``).

Kalshi is a CFTC-regulated binary event exchange. Its public v2 REST API needs no key
for market metadata or historical candlesticks:

  * list markets (paginated by ``cursor``):
        GET https://api.elections.kalshi.com/trade-api/v2/markets
        params: ``limit`` (<=1000), ``status`` (active/…​), ``cursor``
        -> {"markets": [ {ticker, event_ticker, title, close_time, status, ...}, ... ],
            "cursor": "<next|empty>"}
  * daily candlesticks for one market (the series-scoped path — the old
    ``/markets/{ticker}/candlesticks`` form 404s vendor-side since ~2026-07;
    the series ticker is the event ticker's prefix before the first ``-``):
        GET https://api.elections.kalshi.com/trade-api/v2/series/{series}/markets/{ticker}/candlesticks
        params: ``start_ts``, ``end_ts`` (epoch seconds), ``period_interval`` (1440=day)
        -> {"candlesticks": [ {end_period_ts, price:{...}, volume_fp,
                               open_interest_fp}, ... ]}

Prices historically came as **cents** (an integer 0..100 = probability * 100) and are
normalized by dividing by 100; the current API generation instead quotes dollar
strings (``price.close_dollars = "0.0300"``), already probabilities — both shapes are
parsed. A prediction-market snapshot has
no natural observation lag — the candle values are current as of the pull — so
``available_from = ingested_at`` (the ``ingest_time`` rule).

Every market is fetched independently and degrades on its own: a candlestick pull that
raises or returns garbage is skipped with a recorded warning rather than aborting the
whole ingest. The BaseLoader machinery gives us watermark-incremental pulls for free
(``run`` trims the fetch start to the lake watermark minus the overlap window).
"""
from __future__ import annotations

import pandas as pd

from production.data.base import AvailabilityRule, BaseLoader

KALSHI_BASE = "https://api.elections.kalshi.com/trade-api/v2"


class KalshiLoader(BaseLoader):
    dataset = "event_markets"
    vendor = "kalshi"
    source = "kalshi:markets"
    asset_classes = ["events"]
    default_asset_class = "events"
    # Snapshot data: knowable exactly when we pulled it.
    availability_rule = AvailabilityRule("ingest_time")
    expectations = {
        "columns": ["yes_price", "volume", "open_interest"],
        "ranges": {"yes_price": (0.0, 1.0), "volume": (0.0, None),
                   "open_interest": (0.0, None)},
        "min_rows": 1,
    }

    def __init__(self, lake=None, instruments=None, status="active", max_markets=500,
                 page_limit=1000):
        super().__init__(lake, instruments)
        self.status = status
        self.max_markets = max_markets
        self.page_limit = page_limit

    # ------------------------------------------------------------------ fetch
    def fetch(self, start, end) -> dict:
        """Return ``{"markets": [...], "candles": {ticker: [candle, ...]}}``.

        Market metadata is paginated via the opaque ``cursor``; each listed market's
        daily candlesticks are pulled between ``start`` and ``end``. Per-market failures
        are swallowed with a warning so one bad ticker never sinks the batch.
        """
        import requests

        start_ts = int(pd.Timestamp(start).timestamp())
        end_ts = int(pd.Timestamp(end).timestamp())

        markets: list[dict] = []
        cursor = None
        while len(markets) < self.max_markets:
            params = {"limit": self.page_limit, "status": self.status}
            if cursor:
                params["cursor"] = cursor
            resp = requests.get(f"{KALSHI_BASE}/markets", params=params, timeout=30)
            resp.raise_for_status()
            body = resp.json()
            page = body.get("markets", [])
            markets.extend(page)
            cursor = body.get("cursor")
            if not cursor or not page:
                break
        markets = markets[: self.max_markets]

        candles: dict[str, list] = {}
        for m in markets:
            ticker = m.get("ticker")
            if not ticker:
                continue
            # Series-scoped candlestick path (the unscoped one 404s vendor-side).
            series = str(m.get("event_ticker") or ticker).split("-", 1)[0]
            try:
                cresp = requests.get(
                    f"{KALSHI_BASE}/series/{series}/markets/{ticker}/candlesticks",
                    params={"start_ts": start_ts, "end_ts": end_ts,
                            "period_interval": 1440},
                    timeout=30)
                cresp.raise_for_status()
                candles[ticker] = cresp.json().get("candlesticks", [])
            except Exception as exc:  # network / shape / rate-limit — degrade this market
                self.warnings.append(f"kalshi candlesticks failed for {ticker}: {exc!r}")
                continue
        return {"markets": markets, "candles": candles}

    # -------------------------------------------------------------- transform
    def transform(self, raw) -> pd.DataFrame:
        markets = {m.get("ticker"): m for m in raw.get("markets", []) if m.get("ticker")}
        candles = raw.get("candles", {})
        rows: list[dict] = []
        for ticker, series in candles.items():
            meta = markets.get(ticker, {})
            # Stored as an ISO string (not a tz-aware Timestamp): the curated audit's
            # numeric summary cannot introspect tz-aware datetime columns. Downstream
            # (markets/signals) parses it back with pd.to_datetime(..., utc=True).
            close_time = meta.get("close_time")
            status = meta.get("status")
            # event_ticker groups all markets of one event -> the correlated-resolution key.
            event_key = meta.get("event_ticker") or ticker
            question = meta.get("title") or meta.get("subtitle") or ticker
            instrument_id = f"EV:kalshi:{ticker}"
            for candle in series or []:
                try:
                    ts = candle.get("end_period_ts")
                    obs_date = pd.Timestamp(int(ts), unit="s", tz="UTC").tz_localize(None).normalize()
                    price = candle.get("price") or {}
                    close_cents = price.get("close")
                    if close_cents is not None:
                        yes_price = float(close_cents) / 100.0  # cents -> probability
                    elif price.get("close_dollars") is not None:
                        yes_price = float(price["close_dollars"])  # already probability
                    else:
                        continue
                    rows.append({
                        "obs_date": obs_date,
                        "instrument_id": instrument_id,
                        "yes_price": yes_price,
                        "volume": float(candle.get("volume")
                                        or candle.get("volume_fp") or 0.0),
                        "open_interest": float(candle.get("open_interest")
                                               or candle.get("open_interest_fp") or 0.0),
                        "close_time": close_time,
                        "status": status,
                        "event_key": event_key,
                        "question": question,
                        "venue": "kalshi",
                    })
                except (TypeError, ValueError) as exc:
                    self.warnings.append(f"kalshi candle parse failed for {ticker}: {exc!r}")
                    continue
        return _object_cast(pd.DataFrame(rows))


def _object_cast(df: pd.DataFrame) -> pd.DataFrame:
    """Force text columns to numpy ``object`` dtype.

    pandas 3.x infers python ``str`` columns as ``StringDtype``, which the curated
    audit's numeric summary (``np.issubdtype``) cannot introspect. object dtype keeps
    the audit happy while remaining fully round-trippable through parquet.
    """
    for col in ("close_time", "status", "event_key", "question", "venue"):
        if col in df.columns:
            df[col] = df[col].astype(object)
    return df
