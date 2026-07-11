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

Category stamping (promotion check 3 of the political-favorite-tilt study, see
``research/wiki/questions/research-political-underconfidence.md``): one extra
paginated call enumerates the Politics category::

    GET https://api.elections.kalshi.com/trade-api/v2/series/?category=Politics
        -> {"series": [{ticker, category, ...}, ...], "cursor": "<next|empty>"}

matching the single-page-in-practice behavior documented on this same endpoint by
``kalshi_history.py``'s ``_enumerate_category`` (Politics: ~2,083 series, one page,
no ``cursor`` key at all — but pagination is still driven generically off ``cursor``
in case that changes). Each market's series (its ``event_ticker`` prefix before the
first ``-``) is checked against the resulting frozenset and every row is stamped
``category="politics"`` or ``category="other"``.

This category label gates :func:`production.events.signals.political_favorite_tilt`.
Degrade path: if the series listing itself fails, every row is stamped
``category="unknown"`` (with a recorded warning) rather than guessing — and the
signal treats anything other than ``"politics"`` as *no tilt*. This is
**fail-CLOSED**: an ambiguous category suppresses a new, narrowly-scoped edge. It is
the opposite polarity of :func:`production.events.signals.longshot_bias`'s macro
exclusion, which is fail-OPEN (a missing/ambiguous column there leaves the older,
already-validated fade active elsewhere) — deliberately so, since that signal's
default state is "trade" and this one's default state is "don't".
"""
from __future__ import annotations

import time

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

    # status="open" — the live-market enum value. "active" 400s as of 2026-07-11
    # ("invalid status filter"; probed: open/unopened/settled accepted) — the
    # third vendor-side drift on this API this week (candlestick path, dollar
    # schema, now the status enum).
    def __init__(self, lake=None, instruments=None, status="open", max_markets=500,
                 page_limit=1000, min_volume_contracts=100.0, max_list_pages=30,
                 pause_s=0.15):
        super().__init__(lake, instruments)
        self.status = status
        self.max_markets = max_markets
        self.min_volume_contracts = min_volume_contracts
        self.max_list_pages = max_list_pages
        # Candle + listing pacing: 500 unpaced candlestick calls 429 (2026-07-11).
        self.pause_s = pause_s
        self.page_limit = page_limit

    # ------------------------------------------------------------------ fetch
    def _fetch_politics_series(self) -> frozenset | None:
        """Enumerate every series ticker in Kalshi's Politics category.

        One extra paginated call: ``GET {KALSHI_BASE}/series/?category=Politics``.
        Probed live to return the whole category in a single page (~2,083 series;
        see ``kalshi_history.py``'s ``_enumerate_category`` docstring for the same
        finding on this exact endpoint — no ``cursor`` key at all), but pagination
        is still driven generically off ``cursor`` in case the vendor starts
        paginating it later.

        Returns ``None`` — never an empty set — on any failure, so the caller can
        tell "listing failed, category is unknown for every row" apart from
        "listing succeeded, this series legitimately isn't Politics". Only the
        ``None`` case degrades to ``category="unknown"`` in :meth:`transform`.
        """
        import requests

        tickers: list[str] = []
        cursor = None
        try:
            while True:
                params = {"category": "Politics"}
                if cursor:
                    params["cursor"] = cursor
                resp = requests.get(f"{KALSHI_BASE}/series/", params=params, timeout=30)
                resp.raise_for_status()
                body = resp.json()
                page = body.get("series") or []
                tickers.extend(s.get("ticker") for s in page if s.get("ticker"))
                cursor = body.get("cursor")
                if not cursor or not page:
                    break
        except Exception as exc:  # network / shape / rate-limit — degrade, don't guess
            self.warnings.append(f"kalshi politics series listing failed: {exc!r}")
            return None
        return frozenset(tickers)

    def fetch(self, start, end) -> dict:
        """Return ``{"markets": [...], "candles": {ticker: [candle, ...]},
        "politics_series": frozenset[str] | None}``.

        Market metadata is paginated via the opaque ``cursor``; each listed market's
        daily candlesticks are pulled between ``start`` and ``end``. Per-market failures
        are swallowed with a warning so one bad ticker never sinks the batch. One extra
        call (:meth:`_fetch_politics_series`) enumerates the Politics category for the
        category stamp applied in :meth:`transform`.
        """
        import requests

        start_ts = int(pd.Timestamp(start).timestamp())
        end_ts = int(pd.Timestamp(end).timestamp())

        # Rank by volume, never take the raw listing head: as of 2026-07-11 the
        # "open" listing leads with THOUSANDS of esports parlay micro-markets
        # (KXMVESPORTSMULTIGAMEEXTENDED-*), mostly zero-volume but some traded —
        # a head slice yields no usable batch (zero candles everywhere -> schema-
        # failed empty ingest) and a bare volume floor still fills up with traded
        # parlays. Scan up to max_list_pages pages, keep markets clearing the
        # volume floor, then take the TOP max_markets BY VOLUME — the sleeve's
        # targets (elections/macro/financials, 10^4-10^6 contracts) dominate that
        # ranking; downstream liquid_universe/dedupe handles the rest.
        def _vol(m: dict) -> float:
            try:
                return float(m.get("volume_fp", m.get("volume", 0)) or 0)
            except (TypeError, ValueError):
                return 0.0

        candidates: list[dict] = []
        cursor = None
        pages = 0
        while pages < self.max_list_pages:
            params = {"limit": self.page_limit, "status": self.status}
            if cursor:
                params["cursor"] = cursor
            resp = requests.get(f"{KALSHI_BASE}/markets", params=params, timeout=30)
            resp.raise_for_status()
            body = resp.json()
            page = body.get("markets", [])
            pages += 1
            candidates.extend(m for m in page if _vol(m) >= self.min_volume_contracts)
            cursor = body.get("cursor")
            if not cursor or not page:
                break
            time.sleep(self.pause_s)
        markets = sorted(candidates, key=_vol, reverse=True)[: self.max_markets]
        if not markets:
            self.warnings.append(
                f"kalshi: no market cleared min_volume_contracts="
                f"{self.min_volume_contracts} across {pages} listing page(s)")

        politics_series = self._fetch_politics_series()

        candles: dict[str, list] = {}
        for m in markets:
            ticker = m.get("ticker")
            if not ticker:
                continue
            time.sleep(self.pause_s)   # unpaced candle calls 429 (2026-07-11)
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
        return {"markets": markets, "candles": candles, "politics_series": politics_series}

    # -------------------------------------------------------------- transform
    def transform(self, raw) -> pd.DataFrame:
        markets = {m.get("ticker"): m for m in raw.get("markets", []) if m.get("ticker")}
        candles = raw.get("candles", {})
        # None (listing failed upstream, or the payload predates this key entirely)
        # -> every row degrades to "unknown", never guessed as "other".
        politics_series = raw.get("politics_series")
        rows: list[dict] = []
        for ticker, candle_series in candles.items():
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
            # Market's series = the event_ticker's own prefix before the first "-"
            # (same rule the candlestick fetch and signals._series_of use).
            market_series = str(event_key).split("-", 1)[0]
            if politics_series is None:
                category = "unknown"
            elif market_series in politics_series:
                category = "politics"
            else:
                category = "other"
            for candle in candle_series or []:
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
                        "category": category,
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
    for col in ("close_time", "status", "event_key", "question", "venue", "category"):
        if col in df.columns:
            df[col] = df[col].astype(object)
    return df
