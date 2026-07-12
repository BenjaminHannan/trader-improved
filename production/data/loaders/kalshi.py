"""Kalshi prediction-market loader (curated dataset ``event_markets``).

Kalshi is a CFTC-regulated binary event exchange. Its public v2 REST API needs no key
for market metadata or historical candlesticks:

  * enumerate every series ticker in a category (one call per configured category;
    see :meth:`KalshiLoader._fetch_category_series`):
        GET https://api.elections.kalshi.com/trade-api/v2/series/?category={category}
        -> {"series": [{ticker, category, ...}, ...], "cursor": "<next|empty>"}
  * list a SINGLE series' open markets (paginated by ``cursor``):
        GET https://api.elections.kalshi.com/trade-api/v2/markets
        params: ``series_ticker``, ``limit`` (<=1000), ``status`` (open/…​), ``cursor``
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

**Category-scoped discovery (2026-07-11 rewrite)**: as of that date Kalshi's whole-
universe ``/markets?status=open`` listing is flooded with hundreds of thousands of
same-day esports/cross-category parlay micro-markets (``KXMVESPORTSMULTIGAME*``,
``KXMVECROSSCATEGORY*``). They dominate BOTH the cursor head and the volume ranking
(the single top "open" market was a 51k-contract parlay) and serve NO daily
candlesticks at all (same-day lifetimes) — so head-slicing OR volume-ranking that
listing both yield a zero-candle batch and a schema-failed empty ingest.
Whole-universe scanning is therefore dead: instead, for each configured category
(:attr:`KalshiLoader.categories`, default ``("Politics", "Economics", "Financials")``)
every series ticker is enumerated and each series' open markets are listed directly
via ``/markets?series_ticker={S}&status=open`` — a listing path the parlay flood never
appears on, because the sleeve's targets never share a series with it. See
:meth:`KalshiLoader.fetch` for the full stage-by-stage contract.

Category stamping (promotion check 3 of the political-favorite-tilt study, see
``research/wiki/questions/research-political-underconfidence.md``): the SAME
per-category series enumeration that drives market discovery also drives the
``category`` column. Each market's series (its ``event_ticker`` prefix before the
first ``-``) is looked up against the ``category_series`` map built from every
configured category and stamped with THAT category's own name, lowercased
(``"politics"``, ``"economics"``, ``"financials"``, ...) — not the old politics/other
binary. A series that was enumerated under none of the configured categories (e.g. one
category's listing failed while others succeeded, and the series belongs to the failed
one) stamps ``category="other"``.

This category label gates :func:`production.events.signals.political_favorite_tilt`,
which only acts on ``category == "politics"`` — unaffected by the wider category set.
Degrade path: if EVERY configured category's series listing fails, every row is
stamped ``category="unknown"`` (with a recorded warning per category) rather than
guessing — and the signal treats anything other than ``"politics"`` as *no tilt*. This
is **fail-CLOSED**: an ambiguous category suppresses a new, narrowly-scoped edge. It is
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
                 pause_s=0.15, categories=("Politics", "Economics", "Financials")):
        super().__init__(lake, instruments)
        self.status = status
        self.max_markets = max_markets
        self.min_volume_contracts = min_volume_contracts
        # Per-series page cap (most series are single-page; this only guards a
        # pathological series from spinning forever).
        self.max_list_pages = max_list_pages
        # Series-listing + candle pacing: 500 unpaced candlestick calls 429
        # (2026-07-11); the ~2,700-series-listing rewrite reuses the same pacer.
        self.pause_s = pause_s
        self.page_limit = page_limit
        self.categories = tuple(categories)

    # ------------------------------------------------------------------ fetch
    def _fetch_category_series(self, category: str) -> frozenset | None:
        """Enumerate every series ticker in one Kalshi category.

        One (in practice single-page) call: ``GET {KALSHI_BASE}/series/?category=``
        + ``category``. Probed live to return the whole category in one page — no
        ``cursor`` key at all; see ``kalshi_history.py``'s ``_enumerate_category``
        docstring for the same finding on this exact endpoint — but pagination is
        still driven generically off ``cursor`` in case the vendor starts paginating
        it later.

        Returns ``None`` — never an empty set — on any failure, so :meth:`fetch` can
        tell "this category's listing failed, treat it as unlisted" apart from
        "listing succeeded, this category legitimately has zero series". A ``None``
        here makes :meth:`fetch` warn and skip just this category; only when EVERY
        configured category returns ``None`` does the whole run degrade to
        ``category="unknown"`` stamping in :meth:`transform`.
        """
        import requests

        tickers: list[str] = []
        cursor = None
        try:
            while True:
                params = {"category": category}
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
            self.warnings.append(f"kalshi {category} series listing failed: {exc!r}")
            return None
        return frozenset(tickers)

    def fetch(self, start, end) -> dict:
        """Return ``{"markets": [...], "candles": {ticker: [candle, ...]},
        "category_series": {category_lower: frozenset[str]} | None}``.

        Three stages:

        1. **Category enumeration.** Every series ticker in each configured category
           (:attr:`categories`) is listed via :meth:`_fetch_category_series`. A
           category whose own listing fails is warned-and-skipped (the warning is
           recorded inside that method); the run proceeds with whichever categories
           succeeded. Only if EVERY category fails does the resulting map come back
           ``None`` here (never an empty dict), which is the fail-closed signal
           :meth:`transform` reads to stamp every row ``"unknown"``.

        2. **Market listing, category-scoped.** For every series enumerated in step
           1, its open markets are listed directly via
           ``GET {KALSHI_BASE}/markets?series_ticker={S}&status={status}``
           (cursor-paginated, ``page_limit`` per page, paced with ``pause_s`` between
           calls) — this listing path is never touched by the whole-universe esports-
           parlay flood documented at module level, because the sleeve's target
           series never collide with the parlay series. A series listing call that
           itself fails (network blip, 5xx) is warned and skipped — one bad series
           among ~2,700 must not sink the whole batch — and a series with zero open
           markets still costs exactly one call before moving on. Every returned
           market is filtered to ``volume >= min_volume_contracts`` as it's
           collected; the surviving candidates are then ranked by volume descending
           and only the top ``max_markets`` are kept.

        3. **Candlesticks, top slice only.** Exactly the markets that survived step
           2's rank-and-cap proceed to the (paced) per-market candlestick pull —
           candles are never fetched for anything outside that slice. Unchanged from
           the prior single-listing design: a candlestick pull that raises or returns
           garbage is skipped with a recorded warning rather than aborting the batch.
        """
        import requests

        start_ts = int(pd.Timestamp(start).timestamp())
        end_ts = int(pd.Timestamp(end).timestamp())

        def _vol(m: dict) -> float:
            try:
                return float(m.get("volume_fp", m.get("volume", 0)) or 0)
            except (TypeError, ValueError):
                return 0.0

        # Stage 1: enumerate series per configured category. A category's own
        # listing failure warns (inside _fetch_category_series) and is dropped; only
        # if every category fails does category_series stay empty here -> None below.
        category_series: dict[str, frozenset] = {}
        for category in self.categories:
            series = self._fetch_category_series(category)
            if series is not None:
                category_series[category.lower()] = series
        category_series_out = category_series if category_series else None

        # Stage 2: list OPEN markets per enumerated series — category-scoped, never
        # the flooded whole-universe listing (see module docstring). Filter to the
        # volume floor as candidates are collected; rank + cap happens once, after.
        all_series = sorted({s for series in category_series.values() for s in series})
        candidates: list[dict] = []
        for series in all_series:
            cursor = None
            pages = 0
            while pages < self.max_list_pages:
                params = {"series_ticker": series, "status": self.status,
                          "limit": self.page_limit}
                if cursor:
                    params["cursor"] = cursor
                try:
                    resp = requests.get(f"{KALSHI_BASE}/markets", params=params, timeout=30)
                    resp.raise_for_status()
                    body = resp.json()
                except Exception as exc:  # one bad series must not sink ~2,700 others
                    self.warnings.append(
                        f"kalshi markets listing failed for series {series}: {exc!r}")
                    break
                page = body.get("markets", [])
                pages += 1
                candidates.extend(m for m in page if _vol(m) >= self.min_volume_contracts)
                cursor = body.get("cursor")
                time.sleep(self.pause_s)
                if not cursor or not page:
                    break
        markets = sorted(candidates, key=_vol, reverse=True)[: self.max_markets]
        if not markets:
            self.warnings.append(
                f"kalshi: no market cleared min_volume_contracts="
                f"{self.min_volume_contracts} across {len(all_series)} series "
                f"in categories {sorted(category_series)}")

        # Stage 3: candlesticks for the top slice only.
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
        return {"markets": markets, "candles": candles,
                "category_series": category_series_out}

    # -------------------------------------------------------------- transform
    def transform(self, raw) -> pd.DataFrame:
        markets = {m.get("ticker"): m for m in raw.get("markets", []) if m.get("ticker")}
        candles = raw.get("candles", {})
        # None (every category's listing failed upstream, or the payload predates
        # this key/generation entirely) -> every row degrades to "unknown", never
        # guessed as "other".
        category_series = raw.get("category_series")
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
            if category_series is None:
                category = "unknown"
            else:
                category = next(
                    (cat for cat, series in category_series.items()
                     if market_series in series),
                    "other")
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
