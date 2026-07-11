"""Event (prediction-market) subsystem tests — NO NETWORK.

Both loaders' ``fetch`` is monkeypatched with small canned payloads shaped like the
real Kalshi / Polymarket public responses (documented inline). We exercise the
transform -> stamp -> audit -> curated pipeline, yes_price normalization and the
[0,1] range audit, the liquid-universe and event-dedup logic, the two signals
(including a look-ahead corruption check on the convergence signal), and the
Kelly / book-sizing math.
"""
from __future__ import annotations

import sys
import types

import numpy as np
import pandas as pd
import pytest

from production.data.audit import audit
from production.data.loaders.kalshi import KALSHI_BASE, KalshiLoader
from production.data.loaders.polymarket import PolymarketLoader
from production.events.markets import EventMarket, dedupe_related, liquid_universe
from production.events.signals import (
    longshot_bias,
    political_favorite_tilt,
    resolution_convergence,
)
from production.events.sizing import (
    FEE_BPS,
    bernoulli_variance,
    kelly_fraction,
    size_event_book,
)

UTC = "UTC"


# --------------------------------------------------------------- canned payloads
def _kalshi_payload():
    """Minimal Kalshi shape: /markets list + per-ticker /candlesticks.

    Two markets on one event (event_ticker "EVT-A"). T1 uses the CURRENT API
    generation (dollar-string prices + ``*_fp`` sizes); T2 the legacy CENTS shape
    (integer 10 -> 0.10) — the loader must parse both.
    """
    markets = [
        # volume_fp: the listing-level volume the fetch ranks/filters on (the
        # 2026-07-11 esports-spam fix drops sub-floor listings before candles).
        {"ticker": "T1", "event_ticker": "EVT-A", "title": "Will A resolve YES?",
         "close_time": "2026-08-01T00:00:00Z", "status": "active",
         "volume_fp": "5000.00"},
        {"ticker": "T2", "event_ticker": "EVT-A", "title": "Will A resolve by July?",
         "close_time": "2026-08-01T00:00:00Z", "status": "active",
         "volume_fp": "4000.00"},
    ]
    ts = int(pd.Timestamp("2026-07-01", tz=UTC).timestamp())
    candles = {
        "T1": [{"end_period_ts": ts, "price": {"close_dollars": "0.1000"},
                "volume_fp": "5000.00", "open_interest_fp": "2000.00"}],
        "T2": [{"end_period_ts": ts, "price": {"close": 90}, "volume": 4000,
                "open_interest": 1500}],
    }
    return {"markets": markets, "candles": candles}


def _polymarket_payload():
    """Minimal Polymarket shape: gamma /markets + CLOB /prices-history.

    YES price already a probability in [0,1]; conditionId is the shared-event key.
    """
    markets = [{
        "id": "111", "conditionId": "0xabc", "question": "Will B happen?",
        "endDate": "2026-09-01T00:00:00Z", "closed": False,
        "volume": 25000, "liquidity": 8000,
        "clobTokenIds": "[\"tokYes\", \"tokNo\"]",
    }]
    ts = int(pd.Timestamp("2026-07-02", tz=UTC).timestamp())
    history = {"111": [{"t": ts, "p": 0.62}]}
    return {"markets": markets, "history": history}


# ============================================================ (1) loader transforms
def test_kalshi_transform_shape_and_normalization():
    loader = KalshiLoader()
    long = loader.transform(_kalshi_payload())
    assert set(long["instrument_id"]) == {"EV:kalshi:T1", "EV:kalshi:T2"}
    t1 = long[long["instrument_id"] == "EV:kalshi:T1"].iloc[0]
    assert t1["yes_price"] == pytest.approx(0.10)     # 10 cents -> 0.10 probability
    assert t1["volume"] == 5000 and t1["open_interest"] == 2000
    assert t1["event_key"] == "EVT-A" and t1["venue"] == "kalshi"
    assert pd.Timestamp(t1["close_time"]) == pd.Timestamp("2026-08-01", tz=UTC)


def test_polymarket_transform_shape_and_normalization():
    loader = PolymarketLoader()
    long = loader.transform(_polymarket_payload())
    assert list(long["instrument_id"]) == ["EV:polymarket:111"]
    row = long.iloc[0]
    assert row["yes_price"] == pytest.approx(0.62)    # already a probability
    assert row["volume"] == 25000 and row["open_interest"] == 8000
    assert row["event_key"] == "0xabc" and row["venue"] == "polymarket"


# ================================================ (2) normalization + range audit
_EXPECT = KalshiLoader.expectations


def _mandatory(df):
    """Attach mandatory columns so a transformed frame is audit-ready."""
    df = df.copy()
    ts = pd.Timestamp("2026-07-03 12:00", tz=UTC)
    df["available_from"] = ts
    df["source"] = "test"
    df["ingested_at"] = ts
    return df


def test_yes_price_in_range_passes_audit():
    long = _mandatory(KalshiLoader().transform(_kalshi_payload()))
    assert (long["yes_price"].between(0.0, 1.0)).all()
    assert audit(long, _EXPECT).fatal is False


def test_yes_price_out_of_range_is_fatal():
    long = _mandatory(KalshiLoader().transform(_kalshi_payload()))
    long.loc[long.index[0], "yes_price"] = 1.5          # impossible probability
    rep = audit(long, _EXPECT)
    assert rep.fatal is True
    assert rep.checks["ranges"]["ok"] is False


def test_kalshi_run_end_to_end_writes_curated(tmp_lake, monkeypatch):
    loader = KalshiLoader(tmp_lake)
    monkeypatch.setattr(loader, "fetch", lambda start, end: _kalshi_payload())
    res = loader.run("2026-06-01", "2026-07-31", incremental=False)
    assert res.rows == 2 and res.audit["fatal"] is False
    cur = tmp_lake.read_curated("event_markets", "events")
    assert set(cur["instrument_id"]) == {"EV:kalshi:T1", "EV:kalshi:T2"}
    # snapshot availability == ingest time
    assert (cur["available_from"] == cur["ingested_at"]).all()


# ================================================== (2b) kalshi category stamping
# T1/T2 (from _kalshi_payload) both carry event_ticker "EVT-A" -> series "EVT".
# category_series maps LOWERED category name -> its series set; transform() now
# stamps the matched category's own name (actual category, not a politics/other
# binary).
def test_kalshi_transform_stamps_category_politics_when_series_matches_politics():
    raw = dict(_kalshi_payload(), category_series={"politics": frozenset({"EVT"})})
    long = KalshiLoader().transform(raw)
    assert set(long["category"]) == {"politics"}


def test_kalshi_transform_stamps_actual_category_not_binary():
    # Same series, a DIFFERENT configured category -> the label is that category's
    # own name (economics), proving this isn't a politics/other binary.
    raw = dict(_kalshi_payload(), category_series={"economics": frozenset({"EVT"})})
    long = KalshiLoader().transform(raw)
    assert set(long["category"]) == {"economics"}


def test_kalshi_transform_stamps_category_other_when_series_in_no_configured_category():
    raw = dict(_kalshi_payload(), category_series={
        "politics": frozenset({"SOMEOTHERSERIES"}),
        "financials": frozenset({"YETANOTHER"}),
    })
    long = KalshiLoader().transform(raw)
    assert set(long["category"]) == {"other"}


def test_kalshi_transform_stamps_category_unknown_when_listing_absent():
    # No "category_series" key at all (e.g. a payload from before this feature, or a
    # caller that never ran fetch()) -> fail-CLOSED "unknown", not "other".
    long = KalshiLoader().transform(_kalshi_payload())
    assert set(long["category"]) == {"unknown"}


def test_kalshi_transform_stamps_category_unknown_when_all_categories_failed():
    # fetch()'s own all-categories-failed degrade: category_series is explicitly
    # None (not merely absent).
    raw = dict(_kalshi_payload(), category_series=None)
    long = KalshiLoader().transform(raw)
    assert set(long["category"]) == {"unknown"}


class _FakeResp:
    def __init__(self, body):
        self._body = body

    def raise_for_status(self):
        pass

    def json(self):
        return self._body


def _kalshi_market(ticker, event_ticker, volume,
                   close_time="2026-08-01T00:00:00Z", status="active"):
    return {"ticker": ticker, "event_ticker": event_ticker, "title": ticker,
            "close_time": close_time, "status": status, "volume_fp": str(volume)}


def _kalshi_candle(ts, close_dollars="0.5000", volume=100, open_interest=50):
    return {"end_period_ts": ts, "price": {"close_dollars": close_dollars},
            "volume_fp": str(volume), "open_interest_fp": str(open_interest)}


def _install_kalshi_fake_requests(monkeypatch, fake_get):
    fake_requests = types.ModuleType("requests")
    fake_requests.get = fake_get
    monkeypatch.setitem(sys.modules, "requests", fake_requests)
    monkeypatch.setattr("time.sleep", lambda s: None)


def test_kalshi_fetch_lists_markets_per_series_across_categories_and_stamps_category(monkeypatch):
    """Category-scoped discovery (2026-07-11 rewrite): each configured category's
    series are enumerated, then EACH series is listed directly via
    ``/markets?series_ticker=...&status=open`` -- never the flooded whole-universe
    listing. Category stamping threads all the way through fetch() -> transform()
    to actual per-category labels, not a politics/other binary."""
    ts = int(pd.Timestamp("2026-07-01", tz=UTC).timestamp())
    series_by_category = {"Politics": ["POLSER"], "Economics": ["ECOSER"]}
    markets_by_series = {
        "POLSER": [_kalshi_market("P1", "POLSER-A", 5000)],
        "ECOSER": [_kalshi_market("E1", "ECOSER-A", 4000)],
    }
    candle_calls: list[str] = []

    def fake_get(url, params=None, timeout=None):
        if url == f"{KALSHI_BASE}/series/":
            cat = params["category"]
            return _FakeResp({"series": [{"ticker": t}
                                         for t in series_by_category.get(cat, [])]})
        if url == f"{KALSHI_BASE}/markets":
            assert params["status"] == "open"
            series = params["series_ticker"]
            return _FakeResp({"markets": markets_by_series.get(series, []), "cursor": None})
        if url.endswith("/candlesticks"):
            ticker = url.rsplit("/", 2)[1]
            candle_calls.append(ticker)
            return _FakeResp({"candlesticks": [_kalshi_candle(ts)]})
        raise AssertionError(f"unexpected url {url}")

    _install_kalshi_fake_requests(monkeypatch, fake_get)

    loader = KalshiLoader(categories=("Politics", "Economics"))
    raw = loader.fetch("2026-06-01", "2026-07-31")

    assert raw["category_series"] == {"politics": frozenset({"POLSER"}),
                                      "economics": frozenset({"ECOSER"})}
    assert {m["ticker"] for m in raw["markets"]} == {"P1", "E1"}
    assert set(candle_calls) == {"P1", "E1"}

    long = loader.transform(raw)
    got = dict(zip(long["instrument_id"], long["category"]))
    assert got == {"EV:kalshi:P1": "politics", "EV:kalshi:E1": "economics"}


def test_kalshi_fetch_filters_volume_ranks_and_caps_before_candles(monkeypatch):
    """Candidates below min_volume_contracts never reach the ranking; only the top
    max_markets by volume proceed to a candlestick call -- candles are never fetched
    for anything outside that slice."""
    ts = int(pd.Timestamp("2026-07-01", tz=UTC).timestamp())
    markets = [
        _kalshi_market("LOW", "SER-A", 50),    # below the 100-contract floor -> dropped
        _kalshi_market("MID", "SER-B", 200),
        _kalshi_market("TOP", "SER-C", 300),   # highest volume -> the only survivor (max_markets=1)
    ]

    candle_calls: list[str] = []

    def fake_get(url, params=None, timeout=None):
        if url == f"{KALSHI_BASE}/series/":
            return _FakeResp({"series": [{"ticker": "SER"}]})
        if url == f"{KALSHI_BASE}/markets":
            assert params["series_ticker"] == "SER"
            return _FakeResp({"markets": markets, "cursor": None})
        if url.endswith("/candlesticks"):
            ticker = url.rsplit("/", 2)[1]
            candle_calls.append(ticker)
            return _FakeResp({"candlesticks": [_kalshi_candle(ts)]})
        raise AssertionError(f"unexpected url {url}")

    _install_kalshi_fake_requests(monkeypatch, fake_get)

    loader = KalshiLoader(categories=("Politics",), max_markets=1,
                          min_volume_contracts=100.0)
    raw = loader.fetch("2026-06-01", "2026-07-31")

    assert [m["ticker"] for m in raw["markets"]] == ["TOP"]
    assert candle_calls == ["TOP"]


def test_kalshi_fetch_single_series_listing_failure_does_not_sink_others(monkeypatch):
    """One bad series among (in production) ~2,700 must not sink the whole batch:
    its own /markets listing call is warned-and-skipped, the rest proceed."""
    ts = int(pd.Timestamp("2026-07-01", tz=UTC).timestamp())

    def fake_get(url, params=None, timeout=None):
        if url == f"{KALSHI_BASE}/series/":
            return _FakeResp({"series": [{"ticker": "BADSER"}, {"ticker": "GOODSER"}]})
        if url == f"{KALSHI_BASE}/markets":
            if params["series_ticker"] == "BADSER":
                raise RuntimeError("boom: 500 from vendor")
            return _FakeResp({"markets": [_kalshi_market("G1", "GOODSER-A", 5000)],
                              "cursor": None})
        if url.endswith("/candlesticks"):
            return _FakeResp({"candlesticks": [_kalshi_candle(ts)]})
        raise AssertionError(f"unexpected url {url}")

    _install_kalshi_fake_requests(monkeypatch, fake_get)

    loader = KalshiLoader(categories=("Politics",))
    raw = loader.fetch("2026-06-01", "2026-07-31")

    assert {m["ticker"] for m in raw["markets"]} == {"G1"}
    assert any("BADSER" in w for w in loader.warnings)


def test_kalshi_fetch_category_listing_failure_warns_and_continues_with_others(monkeypatch):
    """A single category's series listing failing is additive: warn, skip that
    category, and proceed with whichever others succeeded -- only ALL-categories-
    failed degrades category_series to None (module docstring, item 3)."""
    ts = int(pd.Timestamp("2026-07-01", tz=UTC).timestamp())

    def fake_get(url, params=None, timeout=None):
        if url == f"{KALSHI_BASE}/series/":
            if params["category"] == "Politics":
                raise RuntimeError("boom: politics series listing unreachable")
            return _FakeResp({"series": [{"ticker": "ECOSER"}]})
        if url == f"{KALSHI_BASE}/markets":
            assert params["series_ticker"] == "ECOSER"
            return _FakeResp({"markets": [_kalshi_market("E1", "ECOSER-A", 5000)],
                              "cursor": None})
        if url.endswith("/candlesticks"):
            return _FakeResp({"candlesticks": [_kalshi_candle(ts)]})
        raise AssertionError(f"unexpected url {url}")

    _install_kalshi_fake_requests(monkeypatch, fake_get)

    loader = KalshiLoader(categories=("Politics", "Economics"))
    raw = loader.fetch("2026-06-01", "2026-07-31")

    assert raw["category_series"] == {"economics": frozenset({"ECOSER"})}
    assert any("Politics series listing failed" in w for w in loader.warnings)
    assert {m["ticker"] for m in raw["markets"]} == {"E1"}

    long = loader.transform(raw)
    assert set(long["category"]) == {"economics"}


def test_kalshi_fetch_all_categories_fail_degrades_category_series_to_none(monkeypatch):
    """Every configured category's series listing failing is the only case that
    degrades category_series to None (and, structurally, leaves no series to list
    markets for at all -- the resulting "unknown" transform() stamping is proven
    directly, against a payload WITH rows, by
    test_kalshi_transform_stamps_category_unknown_when_all_categories_failed)."""
    def fake_get(url, params=None, timeout=None):
        if url == f"{KALSHI_BASE}/series/":
            raise RuntimeError(f"boom: {params['category']} series listing unreachable")
        raise AssertionError(f"unexpected url {url}")   # no series -> no markets call

    _install_kalshi_fake_requests(monkeypatch, fake_get)

    loader = KalshiLoader(categories=("Politics", "Economics", "Financials"))
    raw = loader.fetch("2026-06-01", "2026-07-31")

    assert raw["category_series"] is None
    assert raw["markets"] == [] and raw["candles"] == {}
    # one warning per failed category, plus the "no market cleared the volume floor"
    # summary warning from the (necessarily empty) market-listing stage
    failure_warnings = [w for w in loader.warnings if "series listing failed" in w]
    assert len(failure_warnings) == 3


# ===================================================== helpers: curated-shape panel
def _panel(rows) -> pd.DataFrame:
    """Build an event_markets-shaped panel from (obs_date, iid, yes, vol, oi, close,
    event_key) tuples."""
    df = pd.DataFrame(rows, columns=[
        "obs_date", "instrument_id", "yes_price", "volume", "open_interest",
        "close_time", "event_key"])
    df["obs_date"] = pd.to_datetime(df["obs_date"])
    df["close_time"] = pd.to_datetime(df["close_time"], utc=True)
    df["question"] = df["instrument_id"]
    df["venue"] = "kalshi"
    df["status"] = "active"
    return df


# ============================================================== (3) liquid_universe
def test_liquid_universe_filters_volume_and_nearness():
    obs = "2026-07-01"
    rows = [
        # liquid, 31 days to close -> kept
        (obs, "EV:kalshi:OK", 0.5, 5000, 100, "2026-08-01", "E1"),
        # thin volume -> dropped
        (obs, "EV:kalshi:THIN", 0.5, 100, 100, "2026-08-01", "E2"),
        # closes tomorrow-ish but < min_days -> dropped (0 days to close)
        (obs, "EV:kalshi:SOON", 0.5, 5000, 100, "2026-07-01", "E3"),
        # closes >120 days out -> dropped
        (obs, "EV:kalshi:FAR", 0.5, 5000, 100, "2027-01-01", "E4"),
    ]
    universe = liquid_universe(_panel(rows), min_volume_usd=1000,
                               min_days_to_close=1, max_days_to_close=120)
    assert [m.id for m in universe] == ["EV:kalshi:OK"]
    assert isinstance(universe[0], EventMarket)


def test_liquid_universe_carries_category_into_eventmarket():
    obs = "2026-07-01"
    df = _panel([(obs, "EV:kalshi:POL", 0.5, 5000, 100, "2026-08-01", "E1")])
    df["category"] = "politics"
    universe = liquid_universe(df, min_volume_usd=1000, min_days_to_close=1,
                               max_days_to_close=120)
    assert universe[0].category == "politics"


def test_liquid_universe_category_absent_defaults_to_none():
    # _panel() rows carry no "category" column at all (e.g. a Polymarket-only slice).
    obs = "2026-07-01"
    rows = [(obs, "EV:kalshi:OK", 0.5, 5000, 100, "2026-08-01", "E1")]
    universe = liquid_universe(_panel(rows), min_volume_usd=1000,
                               min_days_to_close=1, max_days_to_close=120)
    assert universe[0].category is None


# ================================================================ (4) dedupe_related
def test_dedupe_by_event_key_groups_related():
    m1 = EventMarket("EV:kalshi:T1", "Will A?", pd.Timestamp("2026-08-01", tz=UTC),
                     0.1, 5000, 100, "kalshi", event_key="EVT-A")
    m2 = EventMarket("EV:kalshi:T2", "Will A by July?", pd.Timestamp("2026-08-01", tz=UTC),
                     0.9, 4000, 100, "kalshi", event_key="EVT-A")
    m3 = EventMarket("EV:poly:9", "Will C?", pd.Timestamp("2026-09-01", tz=UTC),
                     0.4, 9000, 100, "polymarket", event_key="0xzzz")
    groups = dedupe_related([m1, m2, m3])
    assert len(groups) == 2                       # {T1,T2} share EVT-A; C alone
    sizes = sorted(len(g) for g in groups)
    assert sizes == [1, 2]


def test_dedupe_fallback_question_prefix():
    # No event_key -> exact question-prefix match groups them.
    a = EventMarket("EV:x:1", "Fed cuts rates in 2026 meeting", None, 0.3, 1, 1, "x")
    b = EventMarket("EV:x:2", "Fed cuts rates in 2026 meeting?", None, 0.4, 1, 1, "x")
    c = EventMarket("EV:x:3", "Bitcoin above 200k", None, 0.2, 1, 1, "x")
    groups = dedupe_related([a, b, c], prefix_len=20)
    assert sorted(len(g) for g in groups) == [1, 2]


# ============================================== (4b) political same-close-date grouping
def test_dedupe_political_same_close_date_unions_across_event_keys():
    # Two DIFFERENT political events (different event_key) closing the same UTC date
    # -> one correlated group (election-night co-resolution).
    m1 = EventMarket("EV:kalshi:P1", "Will X win state A?",
                     pd.Timestamp("2024-11-05T23:00:00Z"), 0.80, 5000, 100, "kalshi",
                     event_key="EVT-A", category="politics")
    m2 = EventMarket("EV:kalshi:P2", "Will Y win state B?",
                     pd.Timestamp("2024-11-05T23:30:00Z"), 0.75, 5000, 100, "kalshi",
                     event_key="EVT-B", category="politics")
    groups = dedupe_related([m1, m2])
    assert len(groups) == 1
    assert {m.id for m in groups[0]} == {"EV:kalshi:P1", "EV:kalshi:P2"}


def test_dedupe_political_different_close_dates_stay_separate():
    m1 = EventMarket("EV:kalshi:P1", "Will X win?", pd.Timestamp("2024-11-05T23:00:00Z"),
                     0.80, 5000, 100, "kalshi", event_key="EVT-A", category="politics")
    m2 = EventMarket("EV:kalshi:P3", "Will Z win runoff?", pd.Timestamp("2024-12-06T23:00:00Z"),
                     0.60, 5000, 100, "kalshi", event_key="EVT-C", category="politics")
    groups = dedupe_related([m1, m2])
    assert sorted(len(g) for g in groups) == [1, 1]


def test_dedupe_political_grouping_does_not_sweep_in_non_political():
    # A non-political market closing on the SAME date as two political ones must
    # stay out of the politics group entirely -- behavior for non-political markets
    # is unchanged (ordinary event_key/question-prefix grouping).
    m1 = EventMarket("EV:kalshi:P1", "Will X win state A?",
                     pd.Timestamp("2024-11-05T23:00:00Z"), 0.80, 5000, 100, "kalshi",
                     event_key="EVT-A", category="politics")
    m2 = EventMarket("EV:kalshi:P2", "Will Y win state B?",
                     pd.Timestamp("2024-11-05T23:30:00Z"), 0.75, 5000, 100, "kalshi",
                     event_key="EVT-B", category="politics")
    m3 = EventMarket("EV:kalshi:N1", "Will it rain in NYC?",
                     pd.Timestamp("2024-11-05T12:00:00Z"), 0.50, 5000, 100, "kalshi",
                     event_key="EVT-D", category="other")
    groups = dedupe_related([m1, m2, m3])
    sizes = sorted(len(g) for g in groups)
    assert sizes == [1, 2]
    political_group = next(g for g in groups if len(g) == 2)
    assert {m.id for m in political_group} == {"EV:kalshi:P1", "EV:kalshi:P2"}


def test_dedupe_political_without_category_falls_back_to_event_key():
    # category=None (e.g. Polymarket, or a pre-feature panel) -> ordinary grouping,
    # unaffected by same close date.
    m1 = EventMarket("EV:poly:P1", "Will X win?", pd.Timestamp("2024-11-05T23:00:00Z"),
                     0.80, 5000, 100, "polymarket", event_key="0xaaa")
    m2 = EventMarket("EV:poly:P2", "Will Y win?", pd.Timestamp("2024-11-05T23:30:00Z"),
                     0.75, 5000, 100, "polymarket", event_key="0xbbb")
    groups = dedupe_related([m1, m2])
    assert sorted(len(g) for g in groups) == [1, 1]


# ================================================================ (5) longshot_bias
def test_longshot_known_answers_and_boundaries():
    rows = [
        ("2026-07-01", "EV:k:LOW", 0.05, 1, 1, "2026-08-01", "E"),   # in (0.03,0.15) -> -1
        ("2026-07-01", "EV:k:HIGH", 0.95, 1, 1, "2026-08-01", "E"),  # in (0.85,0.97) -> +1
        ("2026-07-01", "EV:k:MID", 0.50, 1, 1, "2026-08-01", "E"),   # absent
        ("2026-07-01", "EV:k:EDGE", 0.15, 1, 1, "2026-08-01", "E"),  # boundary -> absent
        ("2026-07-01", "EV:k:TAIL", 0.01, 1, 1, "2026-08-01", "E"),  # below low -> absent
    ]
    out = longshot_bias(_panel(rows), low=0.03, high=0.15)
    vals = dict(zip(out["instrument_id"], out["value"]))
    assert vals == {"EV:k:LOW": -1.0, "EV:k:HIGH": 1.0}


# ============================================ (5b) longshot macro re-spec (backlog #20)
def _venue_panel(rows) -> pd.DataFrame:
    """Panel builder like ``_panel`` but with an explicit per-row ``venue`` (rows carry
    (obs_date, iid, yes, vol, oi, close, event_key, venue))."""
    df = pd.DataFrame(rows, columns=[
        "obs_date", "instrument_id", "yes_price", "volume", "open_interest",
        "close_time", "event_key", "venue"])
    df["obs_date"] = pd.to_datetime(df["obs_date"])
    df["close_time"] = pd.to_datetime(df["close_time"], utc=True)
    df["question"] = df["instrument_id"]
    df["status"] = "active"
    return df


def test_longshot_excludes_measured_macro_series_on_kalshi_only():
    # yes_price=0.05 is squarely in the (0.03, 0.15) longshot band -> would score -1
    # everywhere if not for the macro exclusion.
    rows = [
        ("2026-07-01", "EV:kalshi:KXCPIYOY-26MAY-T1", 0.05, 1, 1, "2026-08-01",
         "KXCPIYOY-26MAY", "kalshi"),                                     # measured macro -> excluded
        ("2026-07-01", "EV:kalshi:NFLSB-26-T1", 0.05, 1, 1, "2026-08-01",
         "NFLSB-26", "kalshi"),                                           # non-macro kalshi -> scored
        ("2026-07-01", "EV:polymarket:1", 0.05, 1, 1, "2026-08-01",
         "KXCPIYOY-26MAY", "polymarket"),                                 # same series, not kalshi -> scored
    ]
    out = longshot_bias(_venue_panel(rows))
    ids = set(out["instrument_id"])
    assert "EV:kalshi:KXCPIYOY-26MAY-T1" not in ids
    assert ids == {"EV:kalshi:NFLSB-26-T1", "EV:polymarket:1"}


def test_longshot_excludes_legacy_unprefixed_macro_series():
    rows = [
        ("2026-07-01", "EV:kalshi:CPIYOY-23JUN-T1", 0.05, 1, 1, "2026-08-01",
         "CPIYOY-23JUN", "kalshi"),
    ]
    out = longshot_bias(_venue_panel(rows))
    assert out.empty


def test_longshot_unmeasured_macro_ish_prefix_is_not_excluded():
    # "CPIDELAY" and "GDPUSMIN" merely start with measured names; exact-match must NOT
    # treat them as excluded.
    rows = [
        ("2026-07-01", "EV:kalshi:CPIDELAY-24X-T1", 0.05, 1, 1, "2026-08-01",
         "CPIDELAY-24X", "kalshi"),
        ("2026-07-01", "EV:kalshi:GDPUSMIN-24-T1", 0.05, 1, 1, "2026-08-01",
         "GDPUSMIN-24", "kalshi"),
    ]
    out = longshot_bias(_venue_panel(rows))
    assert set(out["instrument_id"]) == {"EV:kalshi:CPIDELAY-24X-T1", "EV:kalshi:GDPUSMIN-24-T1"}


def test_resolution_convergence_unaffected_by_macro_exclusion():
    # Same price path, only the series differs (one measured-macro, one not): the
    # convergence signal must be byte-identical either way -- the re-spec only touches
    # longshot_bias.
    prices = [0.40, 0.42, 0.44, 0.46, 0.48, 0.50, 0.53, 0.56, 0.60, 0.63, 0.66, 0.70]
    dates = pd.date_range("2026-07-01", periods=len(prices), freq="D")
    close = dates[-1] + pd.Timedelta(days=8)
    macro_rows = [(d, "EV:kalshi:M", p, 1, 1, close, "KXCPIYOY-26MAY", "kalshi")
                  for d, p in zip(dates, prices)]
    plain_rows = [(d, "EV:kalshi:M", p, 1, 1, close, "NOTMACRO-26MAY", "kalshi")
                  for d, p in zip(dates, prices)]
    macro_out = resolution_convergence(_venue_panel(macro_rows), window=5)
    plain_out = resolution_convergence(_venue_panel(plain_rows), window=5)
    assert not macro_out.empty
    pd.testing.assert_frame_equal(macro_out, plain_out)


# ========================================== (5c) political_favorite_tilt (promotion)
def _political_panel(rows) -> pd.DataFrame:
    """Panel builder with explicit per-row ``venue`` and ``category`` (rows carry
    (obs_date, iid, yes, vol, oi, close, event_key, venue, category))."""
    df = pd.DataFrame(rows, columns=[
        "obs_date", "instrument_id", "yes_price", "volume", "open_interest",
        "close_time", "event_key", "venue", "category"])
    df["obs_date"] = pd.to_datetime(df["obs_date"])
    df["close_time"] = pd.to_datetime(df["close_time"], utc=True)
    df["question"] = df["instrument_id"]
    df["status"] = "active"
    return df


def test_political_tilt_emits_only_in_bucket_for_politics_kalshi_rows():
    rows = [
        # politics + in-bucket (closed interval) -> p_hat = yes + 0.03
        ("2026-07-01", "EV:kalshi:P1", 0.80, 1, 1, "2026-08-01", "E1", "kalshi", "politics"),
        # inclusive lower boundary
        ("2026-07-01", "EV:kalshi:P2", 0.70, 1, 1, "2026-08-01", "E2", "kalshi", "politics"),
        # inclusive upper boundary
        ("2026-07-01", "EV:kalshi:P3", 0.95, 1, 1, "2026-08-01", "E3", "kalshi", "politics"),
        # below bucket -> no emission
        ("2026-07-01", "EV:kalshi:P4", 0.50, 1, 1, "2026-08-01", "E4", "kalshi", "politics"),
        # above bucket -> no emission
        ("2026-07-01", "EV:kalshi:P5", 0.97, 1, 1, "2026-08-01", "E5", "kalshi", "politics"),
        # non-political kalshi, in bucket -> no emission
        ("2026-07-01", "EV:kalshi:M1", 0.80, 1, 1, "2026-08-01", "E6", "kalshi", "macro"),
        # unknown category (loader listing-failure degrade), in bucket -> no emission
        ("2026-07-01", "EV:kalshi:U1", 0.80, 1, 1, "2026-08-01", "E7", "kalshi", "unknown"),
        # politics on Polymarket (venue guard), in bucket -> no emission
        ("2026-07-01", "EV:polymarket:P6", 0.80, 1, 1, "2026-08-01", "E8", "polymarket", "politics"),
    ]
    out = political_favorite_tilt(_political_panel(rows))
    got = dict(zip(out["instrument_id"], out["value"]))
    assert set(got) == {"EV:kalshi:P1", "EV:kalshi:P2", "EV:kalshi:P3"}
    # value is the EDGE (p_hat - p): the haircut constant everywhere the 0.99
    # cap doesn't bind — the contract size_event_book sums across signals.
    assert got["EV:kalshi:P1"] == pytest.approx(0.03)
    assert got["EV:kalshi:P2"] == pytest.approx(0.03)
    assert got["EV:kalshi:P3"] == pytest.approx(0.03)


def test_political_tilt_edge_shrinks_where_cap_binds():
    rows = [("2026-07-01", "EV:kalshi:P7", 0.98, 1, 1, "2026-08-01", "E1", "kalshi", "politics")]
    # widen `high` so a price whose +haircut would exceed the cap is reachable
    out = political_favorite_tilt(_political_panel(rows), high=0.99)
    # 0.98 + 0.03 = 1.01 -> p_hat clipped to 0.99 -> edge 0.01, not 0.03
    assert out.iloc[0]["value"] == pytest.approx(0.01)


def test_political_tilt_missing_category_column_is_empty():
    # A Polymarket-only panel: the loader never stamps "category" at all -> the
    # whole column is absent (not just NaN) -> fail-closed, no emission anywhere.
    rows = [("2026-07-01", "EV:polymarket:X", 0.80, 1, 1, "2026-08-01", "E1")]
    df = pd.DataFrame(rows, columns=[
        "obs_date", "instrument_id", "yes_price", "volume", "open_interest",
        "close_time", "event_key"])
    df["obs_date"] = pd.to_datetime(df["obs_date"])
    df["close_time"] = pd.to_datetime(df["close_time"], utc=True)
    df["venue"] = "polymarket"
    assert political_favorite_tilt(df).empty


def test_political_tilt_missing_venue_column_is_empty():
    df = _political_panel(
        [("2026-07-01", "EV:kalshi:P1", 0.80, 1, 1, "2026-08-01", "E1", "kalshi", "politics")]
    ).drop(columns=["venue"])
    assert political_favorite_tilt(df).empty


def test_political_tilt_empty_panel_returns_empty_frame():
    assert political_favorite_tilt(pd.DataFrame()).empty


# ==================================================== (6) convergence signal + PIT
def _convergence_panel(prices):
    dates = pd.date_range("2026-07-01", periods=len(prices), freq="D")
    close = dates[-1] + pd.Timedelta(days=8)          # all rows within 30d of close
    rows = [(d, "EV:k:M", p, 1, 1, close, "E") for d, p in zip(dates, prices)]
    return _panel(rows)


def test_convergence_pit_future_corruption_leaves_history():
    prices = [0.40, 0.42, 0.44, 0.46, 0.48, 0.50, 0.53, 0.56, 0.60, 0.63, 0.66, 0.70]
    base = resolution_convergence(_convergence_panel(prices), window=5)

    corrupted = list(prices)
    for i in range(8, len(corrupted)):
        corrupted[i] = 0.01                            # slam the future tail toward 0
    after = resolution_convergence(_convergence_panel(corrupted), window=5)

    # values on dates strictly before the corruption (needs price_D and price_{D-5},
    # both < day 8) must be byte-identical.
    cut = pd.Timestamp("2026-07-01") + pd.Timedelta(days=8)
    b = base[base["obs_date"] < cut].reset_index(drop=True)
    a = after[after["obs_date"] < cut].reset_index(drop=True)
    assert len(b) > 0
    pd.testing.assert_frame_equal(a, b)


def test_convergence_only_near_close():
    prices = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
    dates = pd.date_range("2026-07-01", periods=len(prices), freq="D")
    far_close = dates[-1] + pd.Timedelta(days=90)      # >30d out -> nothing emitted
    rows = [(d, "EV:k:M", p, 1, 1, far_close, "E") for d, p in zip(dates, prices)]
    assert resolution_convergence(_panel(rows), window=5).empty


# ============================================================== (7) kelly math
def test_bernoulli_variance():
    assert bernoulli_variance(0.5) == pytest.approx(0.25)
    assert bernoulli_variance(0.1) == pytest.approx(0.09)
    assert bernoulli_variance(1.0) == pytest.approx(0.0)


def test_kelly_zero_edge():
    assert kelly_fraction(0.5, 0.5) == pytest.approx(0.0)


def test_kelly_cap_binds_both_sides():
    assert kelly_fraction(0.9, 0.5, cap=0.05) == pytest.approx(0.05)     # long YES capped
    assert kelly_fraction(0.1, 0.5, cap=0.05) == pytest.approx(-0.05)    # NO side capped


def test_kelly_formula_uncapped():
    # f* = (p_model - p_market)/(1 - p_market) = (0.6-0.5)/0.5 = 0.2
    assert kelly_fraction(0.6, 0.5, cap=1.0) == pytest.approx(0.2)


# ========================================================= (8) size_event_book
def _sizing_panel(price_map):
    rows = [("2026-07-01", iid, p, 1, 1, "2026-08-01", "E") for iid, p in price_map.items()]
    return _panel(rows)


def _signals(value_map):
    return pd.DataFrame({
        "obs_date": pd.Timestamp("2026-07-01"),
        "instrument_id": list(value_map),
        "value": list(value_map.values()),
    })


def test_size_book_edge_below_fee_haircut_no_trade():
    haircut = FEE_BPS / 1e4
    sig = _signals({"EV:k:A": haircut * 0.5})           # edge below the 2% haircut
    book = size_event_book(sig, _sizing_panel({"EV:k:A": 0.5}))
    assert book.empty


def test_size_book_per_group_cap_binds():
    sig = _signals({"EV:k:A": 0.5})                     # huge edge
    book = size_event_book(sig, _sizing_panel({"EV:k:A": 0.5}),
                           per_group_cap=0.02, capital_frac=0.10)
    assert len(book) == 1
    assert book.iloc[0]["weight"] == pytest.approx(0.02)
    assert book.iloc[0]["side"] == "YES"


def test_size_book_group_netting_one_position_per_group():
    sig = _signals({"EV:k:A": 0.10, "EV:k:B": 0.30})    # both in one event group
    panel = _sizing_panel({"EV:k:A": 0.5, "EV:k:B": 0.5})
    m1 = EventMarket("EV:k:A", "q", None, 0.5, 1, 1, "kalshi", event_key="EVT")
    m2 = EventMarket("EV:k:B", "q", None, 0.5, 1, 1, "kalshi", event_key="EVT")
    groups = dedupe_related([m1, m2])
    book = size_event_book(sig, panel, groups=groups)
    assert list(book["instrument_id"]) == ["EV:k:B"]    # larger |signal| wins the group


def test_size_book_gross_cap_scales_down():
    price_map = {f"EV:k:{i}": 0.5 for i in range(10)}
    sig = _signals({iid: 0.5 for iid in price_map})     # each maxes its per-group cap
    book = size_event_book(sig, _sizing_panel(price_map),
                           per_group_cap=0.02, capital_frac=0.10)
    assert len(book) == 10
    assert book["weight"].sum() == pytest.approx(0.10)  # scaled to the gross cap
    # equal scaling: each 0.02 -> 0.01
    assert np.allclose(book["weight"], 0.01)
