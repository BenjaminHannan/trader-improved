"""KalshiHistoryLoader tests — NO NETWORK.

``fetch`` is monkeypatched with a canned payload shaped like the real
``/historical/markets`` + ``/historical/trades`` responses (dollar-string prices,
float-string sizes, taker sides — documented in the loader). We pin: daily-bar
aggregation arithmetic (last/VWAP/volume/taker split), the settlement-row PIT
contract (outcome knowable only at settlement_ts), the two-row-type schema, the
[0,1] range audit, and the end-to-end curated write.

A second group of tests below exercises ``fetch`` itself (not monkeypatched) against
a fake ``requests`` module, HTTP-level, to pin the category-enumeration cost control
and the trade-truncation direction — both probed live against the real API (module
docstring) and reproduced here with canned pages so the tests never touch the network.
"""
from __future__ import annotations

import sys
import types

import pandas as pd
import pytest

from production.data.audit import audit
from production.data.loaders.kalshi_history import HIST_BASE, LIVE_BASE, KalshiHistoryLoader

UTC = "UTC"


def _payload():
    """One series, two markets.

    M1 (result yes): 3 trades across 2 UTC days — day 1 has two trades (prices
    0.04 then 0.06, sizes 100/300, takers yes/no), day 2 one trade (0.10, 50, yes).
    M2 (result no): a single thin trade. Shapes mirror the probed API exactly.
    """
    markets = [
        {"ticker": "CPIYOY-23JUN-T2.7", "event_ticker": "CPIYOY-23JUN",
         "title": "Will CPI YoY exceed 2.7%?", "status": "finalized",
         "result": "yes", "settlement_ts": "2023-07-12T13:30:00Z",
         "close_time": "2023-07-12T12:25:00Z", "expiration_value": "3.00",
         "strike_type": "greater", "floor_strike": 2.7},
        {"ticker": "CPIYOY-23JUN-T4.0", "event_ticker": "CPIYOY-23JUN",
         "title": "Will CPI YoY exceed 4.0%?", "status": "finalized",
         "result": "no", "settlement_ts": "2023-07-12T13:30:00Z",
         "close_time": "2023-07-12T12:25:00Z", "expiration_value": "3.00",
         "strike_type": "greater", "floor_strike": 4.0},
    ]
    trades = {
        "CPIYOY-23JUN-T2.7": [
            {"created_time": "2023-07-10T14:00:00Z", "yes_price_dollars": "0.0400",
             "no_price_dollars": "0.9600", "count_fp": "100.00", "taker_side": "yes"},
            {"created_time": "2023-07-10T20:52:15Z", "yes_price_dollars": "0.0600",
             "no_price_dollars": "0.9400", "count_fp": "300.00", "taker_side": "no"},
            {"created_time": "2023-07-11T09:00:00Z", "yes_price_dollars": "0.1000",
             "no_price_dollars": "0.9000", "count_fp": "50.00", "taker_side": "yes"},
        ],
        "CPIYOY-23JUN-T4.0": [
            {"created_time": "2023-07-11T10:00:00Z", "yes_price_dollars": "0.0200",
             "no_price_dollars": "0.9800", "count_fp": "68.00", "taker_side": "no"},
        ],
    }
    return {"KXCPIYOY": {"markets": markets, "trades": trades}}


def test_daily_bar_aggregation_arithmetic():
    long = KalshiHistoryLoader().transform(_payload())
    bars = long[(long["instrument_id"] == "EV:kalshi:CPIYOY-23JUN-T2.7")
                & (long["row_type"] == "bar")].sort_values("obs_date")
    assert len(bars) == 2
    d1 = bars.iloc[0]
    assert d1["obs_date"] == pd.Timestamp("2023-07-10")
    assert d1["yes_price"] == pytest.approx(0.06)          # last trade of the day
    # VWAP = (0.04*100 + 0.06*300) / 400
    assert d1["yes_vwap"] == pytest.approx((0.04 * 100 + 0.06 * 300) / 400)
    assert d1["volume"] == pytest.approx(400.0)
    assert d1["n_trades"] == 2
    assert d1["taker_yes_frac"] == pytest.approx(100 / 400)  # contract-weighted
    # knowable the moment the last trade printed
    assert pd.Timestamp(d1["knowable_at"]) == pd.Timestamp("2023-07-10T20:52:15Z")
    d2 = bars.iloc[1]
    assert d2["yes_price"] == pytest.approx(0.10) and d2["n_trades"] == 1


def test_settlement_rows_carry_outcome_at_settlement_time():
    long = KalshiHistoryLoader().transform(_payload())
    st = long[long["row_type"] == "settlement"].set_index("instrument_id")
    assert st.loc["EV:kalshi:CPIYOY-23JUN-T2.7", "yes_price"] == 1.0   # result yes
    assert st.loc["EV:kalshi:CPIYOY-23JUN-T4.0", "yes_price"] == 0.0   # result no
    row = st.loc["EV:kalshi:CPIYOY-23JUN-T2.7"]
    assert pd.Timestamp(row["knowable_at"]) == pd.Timestamp("2023-07-12T13:30:00Z")
    assert row["obs_date"] == pd.Timestamp("2023-07-12")
    assert row["result"] == "yes" and row["expiration_value"] == "3.00"
    # outcome columns never leak onto bar rows
    bars = long[long["row_type"] == "bar"]
    assert bars["result"].isna().all() and bars["expiration_value"].isna().all()


def test_metadata_and_strikes_on_every_row():
    long = KalshiHistoryLoader().transform(_payload())
    assert (long["series_ticker"] == "KXCPIYOY").all()
    assert (long["event_key"] == "CPIYOY-23JUN").all()
    assert (long["venue"] == "kalshi").all()
    t27 = long[long["instrument_id"] == "EV:kalshi:CPIYOY-23JUN-T2.7"]
    assert (t27["floor_strike"] == 2.7).all()
    assert (t27["strike_type"] == "greater").all()


def test_transform_stamps_category_unknown_when_payload_has_no_category_key():
    """A payload shaped like the pre-category-fix ``fetch()`` output (no "category"
    key per series at all — exactly ``_payload()`` above) degrades every row to
    "unknown" rather than raising or guessing "politics"."""
    long = KalshiHistoryLoader().transform(_payload())
    assert set(long["category"]) == {"unknown"}


def test_transform_stamps_category_from_payload():
    """``transform`` passes each series' resolved category straight through from the
    ``fetch()``-produced payload onto every row of that series."""
    payload = _payload()
    payload["KXCPIYOY"]["category"] = "economics"
    long = KalshiHistoryLoader().transform(payload)
    assert set(long["category"]) == {"economics"}


def test_range_audit_and_end_to_end_curated_write(tmp_lake, monkeypatch):
    loader = KalshiHistoryLoader(tmp_lake)
    monkeypatch.setattr(loader, "fetch", lambda start, end: _payload())
    res = loader.run("2023-01-01", "2023-12-31", incremental=False)
    assert res.audit["fatal"] is False
    cur = tmp_lake.read_curated("event_markets_hist", "events")
    # 2 bars + settlement for M1, 1 bar + settlement for M2
    assert len(cur) == 5
    # PIT: available_from equals each row's own knowable_at, tz-aware UTC
    av = pd.to_datetime(cur["available_from"], utc=True)
    kn = pd.to_datetime(cur["knowable_at"], utc=True)
    assert (av == kn).all()
    rep = audit(cur, KalshiHistoryLoader.expectations)
    assert rep.fatal is False


def test_bad_result_and_bad_trades_degrade_not_raise():
    payload = _payload()
    payload["KXCPIYOY"]["markets"].append(
        {"ticker": "CPIYOY-23JUN-XSCALAR", "event_ticker": "CPIYOY-23JUN",
         "title": "scalar oddity", "result": "3.00",   # non-binary result
         "settlement_ts": "2023-07-12T13:30:00Z",
         "close_time": "2023-07-12T12:25:00Z"})
    payload["KXCPIYOY"]["trades"]["CPIYOY-23JUN-XSCALAR"] = [
        {"created_time": None, "yes_price_dollars": "9.99", "count_fp": "1.00"},
    ]
    loader = KalshiHistoryLoader()
    long = loader.transform(payload)
    # no settlement row for the unmapped result, warning recorded, garbage trade dropped
    assert "EV:kalshi:CPIYOY-23JUN-XSCALAR" not in set(
        long[long["row_type"] == "settlement"]["instrument_id"])
    assert any("unmapped result" in w for w in loader.warnings)
    assert long["yes_price"].between(0, 1).all()


# ======================================================= fetch()-level, no network
class _FakeResp:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def _install_fake_requests(monkeypatch, responder):
    """Replace ``requests`` (imported inside ``fetch``) with a fake whose
    ``Session().get`` routes through ``responder(url, params) -> dict``."""
    calls: list[tuple[str, dict]] = []

    class _FakeSession:
        def get(self, url, params=None, timeout=None):
            p = dict(params or {})
            calls.append((url, p))
            return _FakeResp(responder(url, p))

    mod = types.ModuleType("requests")
    mod.Session = _FakeSession
    monkeypatch.setitem(sys.modules, "requests", mod)
    monkeypatch.setattr("time.sleep", lambda s: None)
    return calls


def _market(ticker, event_ticker, close_time, result="yes"):
    return {"ticker": ticker, "event_ticker": event_ticker, "title": ticker,
            "result": result, "settlement_ts": close_time, "close_time": close_time}


def test_category_enumeration_skips_thin_series_with_summary_warning(monkeypatch):
    """Category mode enumerates via GET {LIVE_BASE}/series?category=... (probed live:
    a single page, no ``cursor`` key at all), then drops any series whose settled
    market count is below ``min_settled_markets`` BEFORE fetching any trades."""
    series_listing = {"series": [       # no "cursor" key, matching the live probe
        {"ticker": "KXTHIN1", "category": "Politics"},
        {"ticker": "KXTHIN2", "category": "Politics"},
        {"ticker": "KXBIG1", "category": "Politics"},
    ]}
    markets_by_series = {
        "KXTHIN1": [_market("KXTHIN1-A", "KXTHIN1-EVT", "2024-01-05T00:00:00Z")],
        "KXTHIN2": [],
        "KXBIG1": [_market(f"KXBIG1-{i}", "KXBIG1-EVT", "2024-01-05T00:00:00Z")
                  for i in range(3)],
    }
    trade_calls: list[str] = []

    def responder(url, params):
        if url == f"{LIVE_BASE}/series":
            assert params.get("category") == "Politics"
            return series_listing
        if url == f"{HIST_BASE}/historical/markets":
            return {"markets": markets_by_series.get(params.get("series_ticker"), []),
                    "cursor": None}
        if url == f"{LIVE_BASE}/markets":
            return {"markets": [], "cursor": None}
        if url in (f"{HIST_BASE}/historical/trades", f"{LIVE_BASE}/markets/trades"):
            trade_calls.append(params.get("ticker"))
            return {"trades": [], "cursor": None}
        raise AssertionError(f"unexpected url {url}")

    _install_fake_requests(monkeypatch, responder)

    loader = KalshiHistoryLoader(category="Politics", min_settled_markets=2)
    raw = loader.fetch("2024-01-01", "2024-01-31")

    # KXTHIN1 (1 settled market) and KXTHIN2 (0) fall below min_settled_markets=2 and
    # are skipped entirely; only KXBIG1 (3 settled markets) is fetched.
    assert set(raw.keys()) == {"KXBIG1"}
    assert len(raw["KXBIG1"]["markets"]) == 3
    # the cost control: no trade fetch ever touched a skipped series' tickers
    assert trade_calls and all(t.startswith("KXBIG1") for t in trade_calls)
    assert any(
        w == "category Politics: enumerated 3 series, fetched 1 with >= 2 settled markets"
        for w in loader.warnings)


def test_series_param_takes_precedence_over_category(monkeypatch):
    """An explicit `series=` list must skip category enumeration entirely — no call
    to the series-listing endpoint, no category summary warning. The explicit
    `category=` still stamps every row (free — no extra call needed since the
    caller already told us the category)."""
    def responder(url, params):
        if url == f"{LIVE_BASE}/series":
            raise AssertionError("category enumeration must not run when series is explicit")
        if url == f"{HIST_BASE}/historical/markets":
            return {"markets": [_market("KXFOO-A", "KXFOO-EVT", "2024-01-05T00:00:00Z")],
                    "cursor": None}
        if url == f"{LIVE_BASE}/markets":
            return {"markets": [], "cursor": None}
        if url in (f"{HIST_BASE}/historical/trades", f"{LIVE_BASE}/markets/trades"):
            return {"trades": [], "cursor": None}
        raise AssertionError(f"unexpected url {url}")

    calls = _install_fake_requests(monkeypatch, responder)
    loader = KalshiHistoryLoader(series=["KXFOO"], category="Politics",
                                 min_settled_markets=99)  # would skip everything if active
    raw = loader.fetch("2024-01-01", "2024-01-31")

    assert set(raw.keys()) == {"KXFOO"}
    assert len(raw["KXFOO"]["markets"]) == 1     # not dropped by min_settled_markets
    assert not any(url == f"{LIVE_BASE}/series" for url, _ in calls)
    assert not any("category Politics" in w for w in loader.warnings)
    assert raw["KXFOO"]["category"] == "politics"


def test_fetch_category_mode_stamps_self_category_on_every_row(monkeypatch):
    """Category-scoped run (self.category set, no explicit series): every fetched
    series stamps `self.category.lower()` directly — the series were enumerated
    FROM that category, so no separate resolution call is made."""
    series_listing = {"series": [{"ticker": "KXBIG1", "category": "Politics"}]}

    def responder(url, params):
        if url == f"{LIVE_BASE}/series":
            assert params.get("category") == "Politics"
            return series_listing
        if url == f"{HIST_BASE}/historical/markets":
            return {"markets": [_market("KXBIG1-A", "KXBIG1-EVT", "2024-01-05T00:00:00Z")],
                    "cursor": None}
        if url == f"{LIVE_BASE}/markets":
            return {"markets": [], "cursor": None}
        if url in (f"{HIST_BASE}/historical/trades", f"{LIVE_BASE}/markets/trades"):
            return {"trades": [], "cursor": None}
        raise AssertionError(f"unexpected url {url}")

    _install_fake_requests(monkeypatch, responder)
    loader = KalshiHistoryLoader(category="Politics", min_settled_markets=1)
    raw = loader.fetch("2024-01-01", "2024-01-31")

    assert raw["KXBIG1"]["category"] == "politics"


def test_fetch_explicit_series_resolves_category_via_listing(monkeypatch):
    """Explicit series, no category hint: category is resolved via the SAME
    category-listing endpoint kalshi.py uses, one call per CATEGORIES entry. A
    series found under exactly one configured category stamps that category's
    lowercase name."""
    listings = {
        "Politics": {"series": [{"ticker": "KXOTHER"}]},
        "Economics": {"series": [{"ticker": "KXFOO"}]},
        "Financials": {"series": []},
    }
    series_calls: list[str] = []

    def responder(url, params):
        if url == f"{LIVE_BASE}/series":
            series_calls.append(params["category"])
            return listings[params["category"]]
        if url == f"{HIST_BASE}/historical/markets":
            return {"markets": [_market("KXFOO-A", "KXFOO-EVT", "2024-01-05T00:00:00Z")],
                    "cursor": None}
        if url == f"{LIVE_BASE}/markets":
            return {"markets": [], "cursor": None}
        if url in (f"{HIST_BASE}/historical/trades", f"{LIVE_BASE}/markets/trades"):
            return {"trades": [], "cursor": None}
        raise AssertionError(f"unexpected url {url}")

    _install_fake_requests(monkeypatch, responder)
    loader = KalshiHistoryLoader(series=["KXFOO"])
    raw = loader.fetch("2024-01-01", "2024-01-31")

    assert series_calls == ["Politics", "Economics", "Financials"]
    assert raw["KXFOO"]["category"] == "economics"


def test_fetch_explicit_series_unmatched_category_stamps_other(monkeypatch):
    """A series found under NONE of the successfully-listed categories stamps
    "other" — mirroring kalshi.py's transform, never guessed as "politics"."""
    def responder(url, params):
        if url == f"{LIVE_BASE}/series":
            return {"series": [{"ticker": "KXSOMETHINGELSE"}]}
        if url == f"{HIST_BASE}/historical/markets":
            return {"markets": [_market("KXFOO-A", "KXFOO-EVT", "2024-01-05T00:00:00Z")],
                    "cursor": None}
        if url == f"{LIVE_BASE}/markets":
            return {"markets": [], "cursor": None}
        if url in (f"{HIST_BASE}/historical/trades", f"{LIVE_BASE}/markets/trades"):
            return {"trades": [], "cursor": None}
        raise AssertionError(f"unexpected url {url}")

    _install_fake_requests(monkeypatch, responder)
    loader = KalshiHistoryLoader(series=["KXFOO"])
    raw = loader.fetch("2024-01-01", "2024-01-31")

    assert raw["KXFOO"]["category"] == "other"


def test_fetch_explicit_series_all_category_listings_fail_stamps_unknown(monkeypatch):
    """Every configured category's listing failing degrades every series to
    "unknown" rather than guessing — the same fail-closed posture kalshi.py uses
    for its own all-categories-failed case."""
    def responder(url, params):
        if url == f"{LIVE_BASE}/series":
            raise RuntimeError(f"boom: {params['category']} series listing unreachable")
        if url == f"{HIST_BASE}/historical/markets":
            return {"markets": [_market("KXFOO-A", "KXFOO-EVT", "2024-01-05T00:00:00Z")],
                    "cursor": None}
        if url == f"{LIVE_BASE}/markets":
            return {"markets": [], "cursor": None}
        if url in (f"{HIST_BASE}/historical/trades", f"{LIVE_BASE}/markets/trades"):
            return {"trades": [], "cursor": None}
        raise AssertionError(f"unexpected url {url}")

    _install_fake_requests(monkeypatch, responder)
    loader = KalshiHistoryLoader(series=["KXFOO"])
    raw = loader.fetch("2024-01-01", "2024-01-31")

    assert raw["KXFOO"]["category"] == "unknown"
    assert sum("category listing failed" in w for w in loader.warnings) == 3


def test_trade_truncation_keeps_newest_survives_oldest_dropped(monkeypatch):
    """Empirically (module docstring, live-probed on KXFEDDECISION-25SEP-C25),
    /historical/trades pages NEWEST-first: page N+1's head continues just before
    page N's tail. So capping at max_trades keeps the head of the sequence (the
    trades nearest settlement) and the pages never fetched are the OLDEST ones.
    Reuses the real probed timestamps for documentation fidelity.
    """
    ticker = "KXFOO-A"
    market = _market(ticker, "KXFOO-EVT", "2025-09-17T17:55:00Z")
    # 4 single-trade pages, strictly decreasing timestamps (newest first), exactly
    # as observed live: 17:54:54 -> 16:23:22 -> 16:23:21 -> 14:27:28.
    pages = {
        None: {"trades": [{"created_time": "2025-09-17T17:54:54.988913Z",
                           "yes_price_dollars": "0.9500", "count_fp": "500.00",
                           "taker_side": "no"}], "cursor": "c1"},
        "c1": {"trades": [{"created_time": "2025-09-17T16:23:22.504277Z",
                           "yes_price_dollars": "0.9200", "count_fp": "300.00",
                           "taker_side": "yes"}], "cursor": "c2"},
        "c2": {"trades": [{"created_time": "2025-09-17T16:23:21.445338Z",
                           "yes_price_dollars": "0.9100", "count_fp": "200.00",
                           "taker_side": "yes"}], "cursor": "c3"},
        "c3": {"trades": [{"created_time": "2025-09-17T14:27:28.176588Z",  # oldest
                           "yes_price_dollars": "0.9000", "count_fp": "100.00",
                           "taker_side": "no"}], "cursor": None},
    }
    trade_page_calls = {"n": 0}

    def responder(url, params):
        if url == f"{HIST_BASE}/historical/markets":
            return {"markets": [market], "cursor": None}
        if url == f"{LIVE_BASE}/markets":
            return {"markets": [], "cursor": None}
        if url == f"{HIST_BASE}/historical/trades":
            trade_page_calls["n"] += 1
            return pages[params.get("cursor")]
        if url == f"{LIVE_BASE}/series":
            # category=None here -> the loader resolves KXFOO's category via this
            # endpoint (one call per CATEGORIES entry); empty listings are fine, the
            # test below doesn't assert on category, only trade-truncation direction.
            return {"series": [], "cursor": None}
        raise AssertionError(f"unexpected url {url}")

    _install_fake_requests(monkeypatch, responder)

    # page_limit=1, max_trade_pages_per_market=3 -> max_trades=3: stops once 3 of the
    # 4 pages have been walked, i.e. exactly the oldest (4th) page is never fetched.
    loader = KalshiHistoryLoader(series=["KXFOO"], page_limit=1,
                                 max_trade_pages_per_market=3)
    raw = loader.fetch("2025-09-01", "2025-09-30")

    rows = raw["KXFOO"]["trades"][ticker]
    assert trade_page_calls["n"] == 3        # oldest page never even requested
    ts = sorted(pd.Timestamp(r["created_time"]) for r in rows)
    assert len(rows) == 3
    assert ts[0] == pd.Timestamp("2025-09-17T16:23:21.445338Z")   # oldest survivor
    assert ts[-1] == pd.Timestamp("2025-09-17T17:54:54.988913Z")  # newest survivor
    assert pd.Timestamp("2025-09-17T14:27:28.176588Z") not in set(ts)  # truncated
    assert any("oldest trades truncated" in w for w in loader.warnings)
