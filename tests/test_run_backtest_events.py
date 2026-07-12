"""scripts/run_backtest.py's events-sleeve lake wiring: config gate + graceful degrade.

production/backtest/engine.py already folds a populated ``event_markets`` panel into the
book as the ``events`` sleeve (see tests/test_backtest_engine.py's events-sleeve
integration section, and its config-gate twin ``test_events_sleeve_disabled_via_config_
is_skipped``). This module tests the layer ABOVE the engine -- the CLI's lake loader,
which is what actually gets that panel out of the curated lake and into the engine's
input bundle. It reuses the canned Kalshi payload shape from tests/test_events.py
(KalshiLoader with a monkeypatched ``fetch``, no network) to write a small curated
event_markets dataset via the loader's real transform/audit/write path, then exercises
``scripts.run_backtest._load_event_markets`` / ``_load_lake_bundle`` directly.

``_load_event_markets`` reads the UNION of two curated datasets -- the historical
settled-market backfill (``event_markets_hist``, written by
``production.data.loaders.kalshi_history.KalshiHistoryLoader``) and the live snapshot
(``event_markets``, written by ``KalshiLoader``) -- deduped on any overlapping
``(obs_date, instrument_id)`` key. The "present data" section below covers all three
shapes: snapshot-only, hist-only, and both-present-with-overlap.

NO NETWORK: everything is written to a tmp lake (the shared ``tmp_lake`` fixture from
tests/conftest.py).
"""
from __future__ import annotations

import pandas as pd
import pytest

from production.data.loaders.kalshi import KalshiLoader
from production.data.loaders.kalshi_history import KalshiHistoryLoader
from scripts.run_backtest import _load_event_markets, _load_lake_bundle, _synthetic_bundle

UTC = "UTC"


# --------------------------------------------------------------- canned payload (reused
# shape from tests/test_events.py::_kalshi_payload -- two markets on one event, current +
# legacy price encodings).
def _kalshi_payload():
    markets = [
        {"ticker": "T1", "event_ticker": "EVT-A", "title": "Will A resolve YES?",
         "close_time": "2026-08-01T00:00:00Z", "status": "active", "volume_fp": "5000.00"},
        {"ticker": "T2", "event_ticker": "EVT-A", "title": "Will A resolve by July?",
         "close_time": "2026-08-01T00:00:00Z", "status": "active", "volume_fp": "4000.00"},
    ]
    ts = int(pd.Timestamp("2026-07-01", tz=UTC).timestamp())
    candles = {
        "T1": [{"end_period_ts": ts, "price": {"close_dollars": "0.1000"},
                "volume_fp": "5000.00", "open_interest_fp": "2000.00"}],
        "T2": [{"end_period_ts": ts, "price": {"close": 90}, "volume": 4000,
                "open_interest": 1500}],
    }
    return {"markets": markets, "candles": candles}


def _write_canned_event_markets(lake, monkeypatch):
    """Run the real KalshiLoader (fetch monkeypatched) so the curated dataset it writes is
    exactly the shape production ingestion produces -- not a hand-rolled frame."""
    loader = KalshiLoader(lake)
    monkeypatch.setattr(loader, "fetch", lambda start, end: _kalshi_payload())
    res = loader.run("2026-06-01", "2026-07-31", incremental=False)
    assert res.rows == 2 and res.audit["fatal"] is False
    return res


# --------------------------------------------------------- canned hist payload (reused
# shape from tests/test_kalshi_history.py's _payload() -- one series, one settled market
# with a single day of trades).
def _kalshi_hist_payload():
    markets = [
        {"ticker": "T3", "event_ticker": "EVT-B", "title": "Will B resolve YES?",
         "close_time": "2021-08-01T00:00:00Z"},
    ]
    trades = {
        "T3": [
            {"created_time": "2021-07-15T14:00:00Z", "yes_price_dollars": "0.4000",
             "no_price_dollars": "0.6000", "count_fp": "100.00", "taker_side": "yes"},
        ],
    }
    return {"KXFOO": {"markets": markets, "trades": trades}}


def _write_canned_event_markets_hist(lake, monkeypatch):
    """Run the real KalshiHistoryLoader (fetch monkeypatched) so the curated
    event_markets_hist dataset it writes is exactly the shape production backfill
    ingestion produces."""
    loader = KalshiHistoryLoader(lake)
    monkeypatch.setattr(loader, "fetch", lambda start, end: _kalshi_hist_payload())
    res = loader.run("2021-07-01", "2021-08-31", incremental=False)
    assert res.rows == 1 and res.audit["fatal"] is False
    return res


def _write_minimal_event_markets_hist(lake, obs_date, instrument_id, yes_price,
                                      available_from, ingested_at):
    """Hand-rolled single-row event_markets_hist write with fully controlled
    available_from/ingested_at, for deterministic union/dedupe-ordering tests (real
    loader timestamps are wall-clock-derived and unsuitable for pinning an ordering)."""
    df = pd.DataFrame({
        "obs_date": [pd.Timestamp(obs_date)],
        "instrument_id": [instrument_id],
        "yes_price": [yes_price],
        "volume": [10.0],
        "row_type": ["bar"],
        "venue": ["kalshi"],
        "available_from": [pd.Timestamp(available_from, tz=UTC)],
        "source": ["test"],
        "ingested_at": [pd.Timestamp(ingested_at, tz=UTC)],
    })
    lake.write_curated(df, "event_markets_hist", "events")


def _write_minimal_event_markets(lake, obs_date, instrument_id, yes_price,
                                 available_from, ingested_at):
    """Hand-rolled single-row event_markets (live-snapshot-shaped) write; see
    ``_write_minimal_event_markets_hist``."""
    df = pd.DataFrame({
        "obs_date": [pd.Timestamp(obs_date)],
        "instrument_id": [instrument_id],
        "yes_price": [yes_price],
        "volume": [10.0],
        "venue": ["kalshi"],
        "available_from": [pd.Timestamp(available_from, tz=UTC)],
        "source": ["test"],
        "ingested_at": [pd.Timestamp(ingested_at, tz=UTC)],
    })
    lake.write_curated(df, "event_markets", "events")


def _write_minimal_prices(lake):
    dates = pd.date_range("2020-01-02", periods=3, freq="D")
    avail = dates.tz_localize(UTC) + pd.Timedelta(hours=24)
    df = pd.DataFrame({"obs_date": dates, "instrument_id": "EQ:AAA:2000-01-03",
                       "close": 100.0, "available_from": avail, "source": "test",
                       "ingested_at": avail})
    lake.write_curated(df, "prices", "equity")


# =================================================== (a) data present + config-enabled
def test_load_event_markets_snapshot_only_populates_frame(tmp_lake, monkeypatch):
    """Only the live snapshot (event_markets) is curated -- event_markets_hist absent
    entirely -- and the union degrades gracefully to just the one present dataset."""
    _write_canned_event_markets(tmp_lake, monkeypatch)
    df = _load_event_markets(tmp_lake, "2026-06-01", "2026-07-31", events_enabled=True)
    assert df is not None
    assert set(df["instrument_id"]) == {"EV:kalshi:T1", "EV:kalshi:T2"}
    # exactly what production/events/backtest.py::event_sleeve_returns needs downstream.
    assert {"obs_date", "yes_price", "available_from", "close_time"}.issubset(df.columns)


def test_load_event_markets_hist_only_populates_frame(tmp_lake, monkeypatch):
    """Only the historical backfill (event_markets_hist) is curated -- event_markets
    (live snapshot) absent entirely -- and the union degrades gracefully to just the
    one present dataset."""
    _write_canned_event_markets_hist(tmp_lake, monkeypatch)
    df = _load_event_markets(tmp_lake, "2021-07-01", "2021-08-31", events_enabled=True)
    assert df is not None
    assert set(df["instrument_id"]) == {"EV:kalshi:T3"}
    assert {"obs_date", "yes_price", "available_from"}.issubset(df.columns)


def test_load_event_markets_unions_hist_and_snapshot_dedupe_keeps_latest(tmp_lake):
    """Both curated datasets present with an overlapping (obs_date, instrument_id) key:
    the union keeps exactly one row per key -- the LATER one by
    [available_from, ingested_at], the repo's vendor-dedupe convention (see
    scripts/remediate_illiquid_equity.py's _deduped_view for the same sort-then-
    drop_duplicates(keep='last') pattern applied to a single vendor's overlap). A
    non-overlapping row from either side survives untouched."""
    iid = "EV:kalshi:T1"
    _write_minimal_event_markets_hist(
        tmp_lake, "2026-07-01", iid, yes_price=0.80,
        available_from="2026-07-01T10:00:00Z", ingested_at="2026-07-01T10:00:00Z")
    _write_minimal_event_markets(
        tmp_lake, "2026-07-01", iid, yes_price=0.10,
        available_from="2026-07-05T00:00:00Z", ingested_at="2026-07-05T00:00:00Z")
    _write_minimal_event_markets_hist(
        tmp_lake, "2026-06-15", "EV:kalshi:T2", yes_price=0.55,
        available_from="2026-06-15T09:00:00Z", ingested_at="2026-06-15T09:00:00Z")

    df = _load_event_markets(tmp_lake, "2026-06-01", "2026-07-31", events_enabled=True)
    assert df is not None
    assert len(df) == 2   # the overlapping key collapsed to exactly one row
    assert set(df["instrument_id"]) == {iid, "EV:kalshi:T2"}
    overlapped = df[(df["obs_date"] == pd.Timestamp("2026-07-01"))
                    & (df["instrument_id"] == iid)]
    assert len(overlapped) == 1
    assert overlapped.iloc[0]["yes_price"] == pytest.approx(0.10)   # the later row wins
    survivor = df[df["instrument_id"] == "EV:kalshi:T2"].iloc[0]
    assert survivor["yes_price"] == pytest.approx(0.55)


def test_load_lake_bundle_includes_event_markets_when_present(tmp_lake, monkeypatch, capsys):
    _write_minimal_prices(tmp_lake)
    _write_canned_event_markets(tmp_lake, monkeypatch)
    bundle = _load_lake_bundle(tmp_lake, None, None, events_enabled=True)
    assert "event_markets" in bundle
    assert set(bundle["event_markets"]["instrument_id"]) == {"EV:kalshi:T1", "EV:kalshi:T2"}
    out = capsys.readouterr().out
    # data WAS found -> neither degrade note fires.
    assert "'event_markets' absent" not in out
    assert "events sleeve disabled" not in out


# ============================================================ (b) no data -> degrade
def test_load_event_markets_absent_degrades_with_note_not_crash(tmp_lake, capsys):
    """An empty lake (no curated event_markets dataset at all) must not raise -- it prints
    the same style of note the optional SIGNAL_BUNDLE_DATASETS entries (tvl, basis, ...)
    use and returns None."""
    df = _load_event_markets(tmp_lake, "2026-06-01", "2026-07-31", events_enabled=True)
    assert df is None
    out = capsys.readouterr().out
    assert "optional dataset 'event_markets' absent" in out


def test_load_lake_bundle_no_events_data_runs_without_crashing(tmp_lake, capsys):
    _write_minimal_prices(tmp_lake)
    bundle = _load_lake_bundle(tmp_lake, None, None, events_enabled=True)
    assert "event_markets" not in bundle
    assert "prices" in bundle    # the rest of the bundle is unaffected by the degrade
    out = capsys.readouterr().out
    assert "optional dataset 'event_markets' absent" in out


# ======================================================= (c) config-disabled (events.enabled=false)
def test_load_event_markets_disabled_skips_lake_read_even_if_data_present(
        tmp_lake, monkeypatch, capsys):
    """events.enabled=false must suppress the sleeve even when curated data DOES exist --
    the config gate wins over data presence, mirroring the engine-level gate pinned by
    tests/test_backtest_engine.py::test_events_sleeve_disabled_via_config_is_skipped."""
    _write_canned_event_markets(tmp_lake, monkeypatch)
    df = _load_event_markets(tmp_lake, "2026-06-01", "2026-07-31", events_enabled=False)
    assert df is None
    out = capsys.readouterr().out
    assert "events sleeve disabled" in out
    assert "events.enabled=false" in out


def test_load_lake_bundle_disabled_omits_event_markets(tmp_lake, monkeypatch, capsys):
    _write_minimal_prices(tmp_lake)
    _write_canned_event_markets(tmp_lake, monkeypatch)
    bundle = _load_lake_bundle(tmp_lake, None, None, events_enabled=False)
    assert "event_markets" not in bundle
    assert "prices" in bundle


# ============================================================ (a) synthetic path unaffected
def test_synthetic_bundle_has_no_event_markets_key():
    """The --synthetic smoke path never touches the lake, so it must never carry an
    event_markets key -- the engine's events branch is a no-op on that path (data.get
    returns None), reproducing pre-events-sleeve behavior bit-identically."""
    bundle, instruments, sectors = _synthetic_bundle("2019-01-01", "2019-06-30")
    assert "event_markets" not in bundle
    assert "prices" in bundle
