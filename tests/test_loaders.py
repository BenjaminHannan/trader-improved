"""Loader tests — NO NETWORK.

Every loader's `fetch` is monkeypatched to return small hand-built payloads shaped
like the real vendor responses, so the pipeline (transform -> stamp -> audit ->
curated write), the availability rules, and the audit fatal/warning split are all
exercised without touching the network.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from production.data.audit import audit
from production.data.base import AvailabilityRule, IngestError, stamp_availability
from production.data.loaders.ccxt_funding import (DEFAULT_EXCHANGE_CHAIN,
                                                  CcxtFundingLoader)
from production.data.loaders.cftc_cot import CftcCotLoader
from production.data.loaders.fred_alfred import FredAlfredLoader
from production.data.loaders.ken_french import KenFrenchLoader, parse_french_csv
from production.data.loaders.yfinance_prices import YFinancePricesLoader

UTC = "UTC"


# --------------------------------------------------------------------- helpers
def _instruments() -> pd.DataFrame:
    def row(iid, sym, yf, stq):
        return dict(
            instrument_id=iid, asset_class="equity", sleeve="equity", symbol=sym,
            vendor_symbols=json.dumps({"yfinance": yf, "stooq": stq}),
            currency="USD", valid_from="1990-01-01", valid_to="2099-01-01",
            proxy_of=None, sector="tech", meta="{}",
        )
    return pd.DataFrame([
        row("EQ:AAA:2000-01-03", "AAA", "AAA", "aaa.us"),
        row("EQ:BBB:2000-01-03", "BBB", "BBB", "bbb.us"),
    ])


def _instruments_with_fb() -> pd.DataFrame:
    """Instrument master extended with FB, whose bare ticker is blocklisted from the
    2022-06-09 Meta rename onward (see reference.hygiene.REUSED_TICKER_BLOCKLIST)."""
    base = _instruments()
    fb = dict(
        instrument_id="EQ:FB:2012-05-18", asset_class="equity", sleeve="equity",
        symbol="FB", vendor_symbols=json.dumps({"yfinance": "FB", "stooq": "fb.us"}),
        currency="USD", valid_from="1990-01-01", valid_to="2099-01-01",
        proxy_of=None, sector="tech", meta="{}",
    )
    return pd.concat([base, pd.DataFrame([fb])], ignore_index=True)


def _yf_payload(dates, syms) -> pd.DataFrame:
    """yfinance-style DataFrame with MultiIndex (ticker, field) columns."""
    frames = {}
    for j, sym in enumerate(syms):
        base = 10.0 * (j + 1)
        frames[sym] = pd.DataFrame({
            "Open": base + np.arange(len(dates)),
            "High": base + np.arange(len(dates)) + 1,
            "Low": base + np.arange(len(dates)) - 1,
            "Close": base + np.arange(len(dates)),
            "Volume": 100.0 * (np.arange(len(dates)) + 1),
        }, index=dates)
    return pd.concat(frames, axis=1)  # -> columns MultiIndex level0=ticker


def _long(n=60, iid="EQ:AAA:2000-01-03") -> pd.DataFrame:
    """A valid curated-shape long price frame."""
    dates = pd.date_range("2020-01-02", periods=n, freq="B")
    df = pd.DataFrame({
        "obs_date": dates, "instrument_id": iid,
        "close": np.linspace(10, 20, n), "volume": np.linspace(100, 200, n),
    })
    df["dollar_volume"] = df["close"] * df["volume"]
    df["available_from"] = df["obs_date"].dt.tz_localize(UTC) + pd.Timedelta(hours=21, minutes=30)
    df["source"] = "test"
    df["ingested_at"] = df["available_from"] + pd.Timedelta(minutes=5)
    return df


PRICE_EXPECT = {"columns": ["close", "volume", "dollar_volume"],
                "ranges": {"close": (0, None)}, "max_null_frac": 0.02, "min_rows": 1}


# ============================================================ (1) availability
def test_obs_offset_rule():
    df = pd.DataFrame({"obs_date": [pd.Timestamp("2020-01-06")]})
    out = stamp_availability(df, AvailabilityRule("obs_offset",
                                                  {"offset": pd.Timedelta(hours=21, minutes=30)}))
    assert out["available_from"].iloc[0] == pd.Timestamp("2020-01-06 21:30", tz=UTC)


def test_cot_tuesday_to_friday_2030():
    # The canonical COT lag: Tuesday obs becomes knowable Friday 20:30 UTC same week.
    df = pd.DataFrame({"obs_date": [pd.Timestamp("2020-01-07")]})  # a Tuesday
    rule = AvailabilityRule("next_weekday_time", {"weekday": 4, "hour": 20, "minute": 30})
    out = stamp_availability(df, rule)
    assert out["available_from"].iloc[0] == pd.Timestamp("2020-01-10 20:30", tz=UTC)


def test_cot_wednesday_holiday_shift_to_next_friday():
    # A holiday-shifted Wednesday obs -> the coming Friday 20:30 UTC.
    df = pd.DataFrame({"obs_date": [pd.Timestamp("2020-01-08")]})  # a Wednesday
    rule = AvailabilityRule("next_weekday_time", {"weekday": 4, "hour": 20, "minute": 30})
    out = stamp_availability(df, rule)
    assert out["available_from"].iloc[0] == pd.Timestamp("2020-01-10 20:30", tz=UTC)


def test_ingest_time_rule():
    ts = pd.Timestamp("2021-06-01 12:00", tz=UTC)
    df = pd.DataFrame({"obs_date": [pd.Timestamp("2021-06-01")], "ingested_at": [ts]})
    out = stamp_availability(df, AvailabilityRule("ingest_time"))
    assert out["available_from"].iloc[0] == ts


def test_explicit_rule():
    df = pd.DataFrame({"obs_date": [pd.Timestamp("2021-06-01")],
                       "realtime_start": [pd.Timestamp("2021-06-03")]})
    out = stamp_availability(df, AvailabilityRule("explicit", {"column": "realtime_start"}))
    assert out["available_from"].iloc[0] == pd.Timestamp("2021-06-03", tz=UTC)


def test_explicit_rule_mixed_iso_precision():
    """Vendor timestamps of MIXED sub-second precision in one column (Kalshi
    politics settlements sometimes lack .%f — killed a 4h backfill at stamp time
    2026-07-11): the ISO8601 parser must accept the mix."""
    df = pd.DataFrame({
        "obs_date": pd.to_datetime(["2025-11-14", "2025-11-15"]),
        "knowable_at": ["2025-11-14T16:32:24+00:00",             # no fraction
                        "2025-11-15T13:36:24.520345Z"],          # microseconds
    })
    out = stamp_availability(df, AvailabilityRule("explicit", {"column": "knowable_at"}))
    assert out["available_from"].iloc[0] == pd.Timestamp("2025-11-14T16:32:24", tz=UTC)
    assert out["available_from"].iloc[1] == pd.Timestamp("2025-11-15T13:36:24.520345", tz=UTC)


# ================================================= (4) yfinance transform shape
def test_yfinance_multiindex_to_long_and_dollar_volume():
    dates = pd.date_range("2020-01-02", periods=3, freq="B")
    payload = _yf_payload(dates, ["AAA", "BBB"])
    loader = YFinancePricesLoader(instruments=_instruments(), symbols=["AAA", "BBB"])
    long = loader.transform(payload)

    assert set(long["instrument_id"]) == {"EQ:AAA:2000-01-03", "EQ:BBB:2000-01-03"}
    assert set(long.columns) >= {"obs_date", "instrument_id", "close", "volume",
                                 "dollar_volume", "asset_class"}
    assert len(long) == 6  # 2 symbols x 3 days
    np.testing.assert_allclose(long["dollar_volume"], long["close"] * long["volume"])


# ================================================ (4b) hygiene wired into loader
def test_transform_applies_hygiene_backstop_and_blocklist():
    # Dates straddle the FB blocklist boundary (2022-06-09, inclusive): 06-07, 06-08
    # are legit Facebook rows; 06-09, 06-10 are blocked (recycled ticker era).
    dates = pd.date_range("2022-06-07", periods=4, freq="B")  # Tue..Fri
    payload = _yf_payload(dates, ["AAA", "BBB", "FB"])
    # A single sub-$0.10 close on AAA's first day -> price backstop must drop it.
    payload[("AAA", "Close")] = [0.05, 11.0, 12.0, 13.0]

    loader = YFinancePricesLoader(instruments=_instruments_with_fb(),
                                  symbols=["AAA", "BBB", "FB"])
    long = loader.transform(payload)

    def days(iid):
        sub = long[long["instrument_id"] == iid]
        return set(pd.to_datetime(sub["obs_date"]).dt.strftime("%Y-%m-%d"))

    # AAA: the $0.05 row (2022-06-07) is gone; the three real rows remain.
    assert days("EQ:AAA:2000-01-03") == {"2022-06-08", "2022-06-09", "2022-06-10"}
    # FB: rows on/after the 2022-06-09 boundary dropped; earlier rows survive.
    assert days("EQ:FB:2012-05-18") == {"2022-06-07", "2022-06-08"}
    # BBB is untouched by either backstop.
    assert days("EQ:BBB:2000-01-03") == {"2022-06-07", "2022-06-08", "2022-06-09", "2022-06-10"}
    # No sub-floor close survives anywhere.
    assert (long["close"] >= 0.10).all()
    # Drop counts recorded (1 penny row, 2 blocklisted FB rows) and surfaced as a warning.
    assert loader.hygiene_drops == {"backstop_dropped": 1, "flap_dropped": 0,
                                    "blocklist_dropped": 2, "corrupt_series": [],
                                    "illiquid_series": []}
    assert any("hygiene" in w for w in loader.warnings)


def test_hygiene_drop_counts_land_in_audit_record(tmp_lake, monkeypatch):
    dates = pd.date_range("2022-06-07", periods=4, freq="B")
    payload = _yf_payload(dates, ["AAA", "FB"])
    payload[("AAA", "Close")] = [0.05, 11.0, 12.0, 13.0]
    loader = YFinancePricesLoader(tmp_lake, _instruments_with_fb(), symbols=["AAA", "FB"])
    monkeypatch.setattr(loader, "fetch", lambda start, end: payload)

    res = loader.run("2022-06-01", "2022-06-30", incremental=False)
    # 3 AAA rows kept + 2 FB rows kept (06-07, 06-08) = 5.
    assert res.rows == 5
    assert any("hygiene" in w for w in res.audit["warnings"])


# ==================================================== (2) run end-to-end on lake
def test_run_end_to_end_writes_raw_curated_and_audit(tmp_lake, monkeypatch):
    dates = pd.date_range("2020-01-02", periods=4, freq="B")
    payload = _yf_payload(dates, ["AAA", "BBB"])
    loader = YFinancePricesLoader(tmp_lake, _instruments(), symbols=["AAA", "BBB"])
    monkeypatch.setattr(loader, "fetch", lambda start, end: payload)

    res = loader.run("2020-01-01", "2020-01-31", incremental=False)
    assert res.rows == 8

    # raw zone snapshot written
    raw_files = list((tmp_lake.root / "raw" / "vendor=yfinance"
                      / "dataset=prices").glob("ingest_date=*/part.parquet"))
    assert raw_files, "raw snapshot not written"

    # curated rows carry all mandatory columns + correct available_from
    cur = tmp_lake.read_curated("prices", "equity")
    for col in ("obs_date", "available_from", "source", "ingested_at", "instrument_id"):
        assert col in cur.columns
    first = cur.sort_values("obs_date").iloc[0]
    assert first["available_from"] == pd.Timestamp("2020-01-02 21:30", tz=UTC)

    # audit record persisted
    audit_files = list((tmp_lake.root / "audit").glob("*.json"))
    assert audit_files
    rec = json.loads(audit_files[0].read_text())
    assert rec["dataset"] == "prices" and rec["fatal"] is False


# ================================================================ (3) audit()
def test_audit_passes_clean_frame():
    assert audit(_long(), PRICE_EXPECT).fatal is False


def test_audit_fatal_on_missing_mandatory_column():
    rep = audit(_long().drop(columns=["available_from"]), PRICE_EXPECT)
    assert rep.fatal is True
    assert rep.checks["schema"]["ok"] is False


def test_audit_fatal_on_duplicate_key():
    df = _long()
    dup = df.iloc[[0]].copy()
    dup["close"] = 999.0  # same (obs_date, instrument_id, available_from), conflicting value
    rep = audit(pd.concat([df, dup], ignore_index=True), PRICE_EXPECT)
    assert rep.fatal is True
    assert rep.checks["duplicates"]["count"] >= 1


def test_audit_fatal_on_negative_close():
    df = _long()
    df.loc[df.index[0], "close"] = -5.0
    rep = audit(df, PRICE_EXPECT)
    assert rep.fatal is True
    assert rep.checks["ranges"]["ok"] is False


def test_audit_warning_only_on_mild_null_fraction():
    df = _long(n=100)
    df.loc[df.index[:4], "volume"] = np.nan  # 4% nulls: over 2% threshold, under 5x (10%)
    rep = audit(df, PRICE_EXPECT)
    assert rep.fatal is False                       # warning, not fatal
    assert rep.checks["null_fraction"]["ok"] is False
    assert "volume" in rep.checks["null_fraction"]["over_threshold"]


def test_audit_summary_has_moments_and_span():
    rep = audit(_long(), PRICE_EXPECT)
    assert rep.summary["rows"] == 60
    assert "close" in rep.summary["columns"]
    assert set(rep.summary["columns"]["close"]) == {"mean", "std", "min", "max"}
    assert len(rep.summary["date_span"]) == 2


# ============================================================ (5) fred aliasing
def _fred_payload():
    dates = pd.date_range("2020-01-01", periods=5, freq="B")
    def mk(v):
        return pd.DataFrame({"date": dates.astype(str), "value": [v] * 5,
                             "realtime_start": dates.astype(str)})
    return {"DGS3MO": mk(2.0), "IR3TIB01EZM156N": mk(0.5)}


def test_fred_alias_mapping_applied():
    loader = FredAlfredLoader()
    long = loader.transform(_fred_payload())
    assert set(long["series_id"]) == {"DGS3MO_US", "RATE_EU"}
    # vintages present -> explicit realtime_start rule
    assert loader.availability_rule.kind == "explicit"
    assert loader.availability_rule.params["column"] == "realtime_start"


def test_fred_fallback_no_vintages_warns_and_uses_offset():
    dates = pd.date_range("2020-01-01", periods=5, freq="B")
    payload = {"VIXCLS": pd.DataFrame({"date": dates.astype(str), "value": [18.0] * 5})}
    loader = FredAlfredLoader()
    long = loader.transform(payload)
    assert set(long["series_id"]) == {"VIXCLS"}
    assert loader.availability_rule.kind == "obs_offset"
    assert any("fredgraph" in w for w in loader.warnings)


def test_fred_realtime_chunks_stay_under_vintage_cap():
    """Regression (live-ingest 2026-07-07): ALFRED 400s when the realtime window spans
    > 2000 vintage dates (10y of a daily series is ~2600). The window must be walked in
    <=4-year chunks, contiguous and non-overlapping, final chunk open at 9999-12-31."""
    from production.data.loaders.fred_alfred import _realtime_chunks

    chunks = _realtime_chunks("2016-01-01", "2026-07-07")
    assert len(chunks) >= 3
    assert str(chunks[0][0]) == "2016-01-01"
    assert chunks[-1][1] == "9999-12-31"
    for (cs, ce), (ns, _) in zip(chunks, chunks[1:]):
        assert pd.Timestamp(ns) == pd.Timestamp(ce) + pd.Timedelta(days=1)  # contiguous
        # each closed chunk spans <= 4 years => under the ~2000 daily-vintage cap
        assert (pd.Timestamp(ce) - pd.Timestamp(cs)).days <= 4 * 366


def test_fred_vintaged_fetch_stitches_chunks_and_keeps_earliest_vintage(monkeypatch):
    """Chunked ALFRED pulls re-report older observations clamped to each chunk start;
    the stitcher must keep the earliest realtime_start per (date, value) while a
    genuine revision (same date, new value) keeps its own later vintage."""
    import sys
    import types

    calls = []

    class _Resp:
        status_code = 200
        text = ""
        def __init__(self, obs):
            self._obs = obs
        def raise_for_status(self):
            pass
        def json(self):
            return {"observations": self._obs}

    def fake_get(url, params=None, timeout=None):
        calls.append(params)
        cs = params["realtime_start"]
        if cs == "2016-01-01":        # chunk 1: original print
            obs = [{"date": "2016-06-01", "value": "2.0", "realtime_start": "2016-06-02"}]
        else:                          # later chunk: clamped carry-in + a real revision
            obs = [{"date": "2016-06-01", "value": "2.0", "realtime_start": cs},
                   {"date": "2016-06-01", "value": "2.5", "realtime_start": "2021-03-01"}]
        return _Resp(obs)

    mod = types.ModuleType("requests")
    mod.get = fake_get
    monkeypatch.setitem(sys.modules, "requests", mod)

    loader = FredAlfredLoader(api_key="test-key", series=["DGS3MO"])
    df = loader._fetch_vintaged("DGS3MO", pd.Timestamp("2016-01-01").date(),
                                pd.Timestamp("2026-07-07").date())
    assert len(calls) >= 2                          # actually chunked
    got = {(r["value"], r["realtime_start"]) for _, r in df.iterrows()}
    assert ("2.0", "2016-06-02") in got             # earliest vintage wins
    assert ("2.5", "2021-03-01") in got             # revision kept
    assert len(df) == 2                             # clamped duplicates dropped


# ===================================================== (6) incremental watermark
def test_incremental_only_repulls_tail(tmp_lake, monkeypatch):
    # Seed the lake so a watermark exists, then confirm an incremental run adjusts
    # the fetch start to watermark - 5 days (small overlap re-pull).
    seed = _long(n=20)  # obs_date 2020-01-02 .. business days out
    seed["obs_date"] = pd.date_range("2020-01-06", periods=20, freq="B")
    seed["available_from"] = seed["obs_date"].dt.tz_localize(UTC) + pd.Timedelta(hours=21, minutes=30)
    seed["ingested_at"] = seed["available_from"]
    tmp_lake.write_curated(seed, "prices", "equity")
    wm = tmp_lake.watermark("prices")
    assert wm is not None

    loader = YFinancePricesLoader(tmp_lake, _instruments(), symbols=["AAA"])
    captured = {}

    def fake_fetch(start, end):
        captured["start"] = pd.Timestamp(start)
        return _yf_payload(pd.date_range("2020-02-01", periods=2, freq="B"), ["AAA"])

    monkeypatch.setattr(loader, "fetch", fake_fetch)
    loader.run("2019-01-01", "2020-02-05", incremental=True)
    assert captured["start"] == pd.Timestamp(wm) - pd.Timedelta(days=5)


# ============================================================= (7) idempotence
def test_double_run_row_count_stable(tmp_lake, monkeypatch):
    dates = pd.date_range("2020-01-02", periods=5, freq="B")
    payload = _yf_payload(dates, ["AAA", "BBB"])
    loader = YFinancePricesLoader(tmp_lake, _instruments(), symbols=["AAA", "BBB"])
    monkeypatch.setattr(loader, "fetch", lambda start, end: payload)

    loader.run("2020-01-01", "2020-01-31", incremental=False)
    n1 = len(tmp_lake.read_curated("prices", "equity"))
    loader.run("2020-01-01", "2020-01-31", incremental=False)  # identical re-pull
    n2 = len(tmp_lake.read_curated("prices", "equity"))
    assert n1 == n2 == 10


# ================================================================== extras
def test_fatal_audit_raises_ingest_error(tmp_lake, monkeypatch):
    # A negative volume must abort the curated write with IngestError. (A negative
    # *close* would be cleaned upstream by the sub-floor hygiene backstop before the
    # audit ever sees it, so we trip a fatal range that hygiene does not touch.)
    dates = pd.date_range("2020-01-02", periods=3, freq="B")
    payload = _yf_payload(dates, ["AAA"])
    payload[("AAA", "Volume")] = [-1.0, 5.0, 6.0]
    loader = YFinancePricesLoader(tmp_lake, _instruments(), symbols=["AAA"])
    monkeypatch.setattr(loader, "fetch", lambda start, end: payload)
    with pytest.raises(IngestError):
        loader.run("2020-01-01", "2020-01-31", incremental=False)
    # audit trail still written even though curated write was blocked
    assert list((tmp_lake.root / "audit").glob("*.json"))


def test_stage3_stubs_raise_not_implemented():
    from production.data.loaders.stage3_stubs.stubs import ENTSOELoader, JODILoader

    for cls in (ENTSOELoader, JODILoader):
        loader = cls()
        assert loader.availability_rule is not None
        with pytest.raises(NotImplementedError):
            loader.fetch("2020-01-01", "2020-12-31")


# ============================================ (8) yfinance payload-shape hardening
def _flat_payload(dates, base=10.0, adj_close=False, lower=False):
    """A single-ticker FLAT-column frame the way yfinance returns for one symbol."""
    cols = {
        "Open": base + np.arange(len(dates)),
        "High": base + np.arange(len(dates)) + 1,
        "Low": base + np.arange(len(dates)) - 1,
        "Close": base + np.arange(len(dates)),
        "Volume": 100.0 * (np.arange(len(dates)) + 1),
    }
    if adj_close:  # auto_adjust=False keeps a separate 'Adj Close'
        cols["Adj Close"] = cols["Close"]
    df = pd.DataFrame(cols, index=dates)
    if lower:
        df.columns = [c.lower() for c in df.columns]
    return df


def test_yfinance_field_ticker_multiindex_order():
    # yfinance's other layout: columns are (field, ticker), not (ticker, field).
    dates = pd.date_range("2020-01-02", periods=3, freq="B")
    payload = _yf_payload(dates, ["AAA", "BBB"]).swaplevel(0, 1, axis=1)
    loader = YFinancePricesLoader(instruments=_instruments(), symbols=["AAA", "BBB"])
    long = loader.transform(payload)
    assert set(long["instrument_id"]) == {"EQ:AAA:2000-01-03", "EQ:BBB:2000-01-03"}
    assert len(long) == 6


def test_yfinance_flat_single_ticker():
    dates = pd.date_range("2020-01-02", periods=4, freq="B")
    loader = YFinancePricesLoader(instruments=_instruments(), symbols=["AAA"])
    long = loader.transform(_flat_payload(dates))
    assert set(long["instrument_id"]) == {"EQ:AAA:2000-01-03"}
    assert len(long) == 4
    np.testing.assert_allclose(long["dollar_volume"], long["close"] * long["volume"])


def test_yfinance_flat_lowercase_and_adjclose():
    # Lower-cased field names AND an extra 'Adj Close' column both parse fine.
    dates = pd.date_range("2020-01-02", periods=3, freq="B")
    loader = YFinancePricesLoader(instruments=_instruments(), symbols=["AAA"])
    long = loader.transform(_flat_payload(dates, adj_close=True, lower=True))
    assert len(long) == 3
    assert (long["close"] > 0).all()


def test_yfinance_named_column_levels_parse():
    # yfinance 1.x stamps names on the levels ('Price','Ticker') + index name 'Date'.
    dates = pd.date_range("2020-01-02", periods=3, freq="B")
    payload = _yf_payload(dates, ["AAA", "BBB"])
    payload.columns = payload.columns.set_names(["Ticker", "Price"])
    payload.index = payload.index.set_names("Date")
    loader = YFinancePricesLoader(instruments=_instruments(), symbols=["AAA", "BBB"])
    long = loader.transform(payload)
    assert len(long) == 6


def test_yfinance_partial_empty_symbol_degrades_with_warning():
    # BBB comes back all-NaN (geo/rate-limited in the batch): warn + skip, don't raise.
    dates = pd.date_range("2020-01-02", periods=3, freq="B")
    payload = _yf_payload(dates, ["AAA", "BBB"])
    payload[("BBB", "Close")] = np.nan
    loader = YFinancePricesLoader(instruments=_instruments(), symbols=["AAA", "BBB"])
    long = loader.transform(payload)
    assert set(long["instrument_id"]) == {"EQ:AAA:2000-01-03"}
    assert any("BBB" in w and "no usable close" in w for w in loader.warnings)


def test_yfinance_unrecognized_shape_reports_columns():
    # Genuinely unparseable (non-OHLCV flat frame, ambiguous multi-symbol request):
    # the error must name the columns for one-glance diagnosis.
    bad = pd.DataFrame({"foo": [1, 2], "bar": [3, 4]})
    loader = YFinancePricesLoader(instruments=_instruments(), symbols=["AAA", "BBB"])
    with pytest.raises(IngestError) as ei:
        loader.transform(bad)
    msg = str(ei.value)
    assert "unrecognized OHLCV payload shape" in msg
    assert "foo" in msg and "bar" in msg


# ================================================= (9) ccxt funding exchange chain
def _crypto_instruments() -> pd.DataFrame:
    def row(iid, ccxt_sym):
        return dict(
            instrument_id=iid, asset_class="crypto", sleeve="crypto",
            symbol=ccxt_sym.split("/")[0],
            vendor_symbols=json.dumps({"ccxt": ccxt_sym}),
            currency="USD", valid_from="1990-01-01", valid_to="2099-01-01",
            proxy_of=None, sector=None, meta="{}",
        )
    return pd.DataFrame([
        row("CR:BTC:2013-09-01", "BTC/USDT:USDT"),
        row("CR:ETH:2015-08-07", "ETH/USDT:USDT"),
    ])


class _FakeExchange:
    def __init__(self, handler):
        self.handler = handler

    def fetch_funding_rate_history(self, sym, since=None):
        return self.handler(sym)  # returns a list, or raises


def _install_fake_ccxt(monkeypatch, handlers: dict):
    """Register a fake ``ccxt`` module whose exchanges use the given per-symbol handlers."""
    import sys
    import types

    mod = types.ModuleType("ccxt")
    for name, handler in handlers.items():
        mod.__dict__[name] = (lambda h: (lambda: _FakeExchange(h)))(handler)
    monkeypatch.setitem(sys.modules, "ccxt", mod)


def _funding_hist(rate):
    day = pd.Timestamp("2021-01-01", tz="UTC")
    return [{"timestamp": int((day + pd.Timedelta(hours=8 * k)).timestamp() * 1000),
             "fundingRate": rate} for k in range(3)]


def test_funding_default_chain_is_reachable_venues():
    loader = CcxtFundingLoader(symbols=["BTC/USDT:USDT"])
    assert loader.exchanges == DEFAULT_EXCHANGE_CHAIN == ["bybit", "okx", "kraken"]
    # An explicit exchange still pins the venue (overrides the chain).
    assert CcxtFundingLoader(symbols=[], exchange="binanceusdm").exchanges == ["binanceusdm"]


def test_funding_chain_falls_through_to_next_exchange(monkeypatch):
    # bybit serves BTC but not ETH; okx serves ETH -> both resolved across the chain.
    def bybit(sym):
        if sym == "BTC/USDT:USDT":
            return _funding_hist(0.0001)
        raise RuntimeError("not listed")

    def okx(sym):
        if sym == "ETH/USDT:USDT":
            return _funding_hist(0.0002)
        raise RuntimeError("not listed")

    _install_fake_ccxt(monkeypatch, {"bybit": bybit, "okx": okx,
                                     "kraken": lambda s: (_ for _ in ()).throw(RuntimeError())})
    loader = CcxtFundingLoader(instruments=_crypto_instruments(),
                              symbols=["BTC/USDT:USDT", "ETH/USDT:USDT"])
    raw = loader.fetch("2021-01-01", "2021-01-02")
    assert set(raw) == {"BTC/USDT:USDT", "ETH/USDT:USDT"}
    long = loader.transform(raw)
    assert set(long["instrument_id"]) == {"CR:BTC:2013-09-01", "CR:ETH:2015-08-07"}
    assert "funding_rate" in long.columns and "available_from" in long.columns


def test_funding_all_symbols_fail_raises_geoblock_message(monkeypatch):
    def geoblocked(sym):
        raise RuntimeError("451 geo-blocked")

    _install_fake_ccxt(monkeypatch, {"bybit": geoblocked, "okx": geoblocked,
                                     "kraken": geoblocked})
    loader = CcxtFundingLoader(instruments=_crypto_instruments(),
                              symbols=["BTC/USDT:USDT", "ETH/USDT:USDT"])
    with pytest.raises(IngestError) as ei:
        loader.fetch("2021-01-01", "2021-01-02")
    msg = str(ei.value).lower()
    assert "geo-block" in msg and "--help" in msg


def test_funding_perp_symbol_resolves_against_spot_master(monkeypatch):
    """Regression (live-ingest 2026-07-07): the REAL master keys ccxt on the spot pair
    ("BTC/USD" — see instruments._crypto_vendor_symbols), while funding fetches the
    perp ("BTC/USDT:USDT"). transform() must fall back to the base's spot pair or every
    row silently resolves to None and the schema audit fails on an empty frame."""
    def spot_row(iid, base):
        return dict(
            instrument_id=iid, asset_class="crypto", sleeve="crypto", symbol=base,
            vendor_symbols=json.dumps({"ccxt": f"{base}/USD"}),   # real master shape
            currency="USD", valid_from="1990-01-01", valid_to="2099-01-01",
            proxy_of=None, sector=None, meta="{}",
        )
    master = pd.DataFrame([spot_row("CR:BTC:2015-01-01", "BTC"),
                           spot_row("CR:ETH:2015-01-01", "ETH")])
    loader = CcxtFundingLoader(instruments=master,
                               symbols=["BTC/USDT:USDT", "ETH/USDT:USDT"])
    raw = {"BTC/USDT:USDT": _funding_hist(0.0001),
           "ETH/USDT:USDT": _funding_hist(0.0002)}
    long = loader.transform(raw)
    assert set(long["instrument_id"]) == {"CR:BTC:2015-01-01", "CR:ETH:2015-01-01"}
    assert (long["funding_rate"].abs() <= 0.05).all()


def test_funding_fetch_paginates_past_venue_page_cap(monkeypatch):
    """Regression (live-ingest 2026-07-07): one fetch_funding_rate_history call returns
    only the venue's most recent page (bybit ~200 cycles ≈ 66 days), so a 2016 start
    silently yielded ~93 days. fetch() must advance `since` past each page."""
    import sys
    import types

    day0 = pd.Timestamp("2021-01-01", tz="UTC")

    class _PagingExchange:
        # honors `since` like a real venue (the shared _FakeExchange drops it), and —
        # like bybit — returns EMPTY when `since` predates the listing, not a clamp.
        # 12 cycles spaced 30d apart (~1y of history, longer than the probe stride).
        def fetch_funding_rate_history(self, sym, since=None):
            rows = [{"timestamp": int((day0 + pd.Timedelta(days=30 * k)).timestamp() * 1000),
                     "fundingRate": 0.0001} for k in range(12)]
            if since is not None and since < rows[0]["timestamp"] - 1:
                return []                   # pre-listing: empty page, no clamping
            rows = [r for r in rows if since is None or r["timestamp"] >= since]
            return rows[:4]                 # pages of 4 cycles

    mod = types.ModuleType("ccxt")
    mod.bybit = _PagingExchange
    monkeypatch.setitem(sys.modules, "ccxt", mod)
    loader = CcxtFundingLoader(instruments=_crypto_instruments(),
                               symbols=["BTC/USDT:USDT"], exchanges=["bybit"])
    # In-history start: all 3 pages stitched, no duplicated boundary rows.
    raw = loader.fetch("2021-01-01", "2022-06-01")
    ts = [r["timestamp"] for r in raw["BTC/USDT:USDT"]]
    assert len(ts) == 12
    assert len(set(ts)) == 12

    # Pre-listing start: bybit-style venues return an EMPTY page rather than clamping,
    # which previously read as "venue has nothing" and fell through to okx's ~3-month
    # retention. The forward probe must find the series (it may skip at most one
    # ~180d stride past the listing — negligible against a multi-year live series).
    raw = loader.fetch("2020-12-01", "2022-06-01")
    ts = [r["timestamp"] for r in raw["BTC/USDT:USDT"]]
    assert len(ts) >= 6                     # found the series and kept paginating
    assert len(set(ts)) == len(ts)


# =================================================== (10) ken french direct zips
_FF_FACTORS_CSV = """This file was created by CMPT_ME_BEME_RETS using the 202001 CRSP database.

,Mkt-RF,SMB,HML,RF
20200102, 0.85,-0.19, 0.34, 0.006
20200103,-0.71, 0.12,-0.09, 0.006
20200106, 0.40, 0.05, 0.11, 0.006

  Copyright 2020 Kenneth R. French
"""

_FF_MOM_CSV = """This file was created using momentum returns.

,Mom
20200102, 0.10
20200103,-0.22
20200106, 0.33

  Copyright 2020 Kenneth R. French
"""


def test_parse_french_csv_skips_preamble_and_footer():
    df = parse_french_csv(_FF_FACTORS_CSV)
    assert list(df.columns) == ["Mkt-RF", "SMB", "HML", "RF"]
    assert len(df) == 3  # footer/copyright line excluded
    assert df.index[0] == pd.Timestamp("2020-01-02")
    np.testing.assert_allclose(df.loc["2020-01-02", "Mkt-RF"], 0.85)


def test_ken_french_transform_maps_and_scales():
    raw = {"factors": parse_french_csv(_FF_FACTORS_CSV),
           "momentum": parse_french_csv(_FF_MOM_CSV)}
    loader = KenFrenchLoader()
    long = loader.transform(raw)
    assert set(long["series_id"]) == {"FF_MKT_RF", "FF_SMB", "FF_HML", "FF_RF", "FF_MOM"}
    # percent -> decimal: 0.85% -> 0.0085
    mkt = long[(long["series_id"] == "FF_MKT_RF") & (long["obs_date"] == "2020-01-02")]
    np.testing.assert_allclose(mkt["value"].iloc[0], 0.0085)
    assert (long["value"].abs() < 1.0).all()


def test_ken_french_fetch_parses_zip_without_network(monkeypatch):
    # Drive fetch end-to-end against an in-memory zip (no network): confirms the
    # requests->zipfile->parse path and the date-window clip.
    import io
    import zipfile

    class _Resp:
        def __init__(self, content):
            self.content = content

        def raise_for_status(self):
            pass

    def _zip_bytes(name, text):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr(name, text)
        return buf.getvalue()

    payloads = {
        "F-F_Research_Data_Factors_daily.CSV": _FF_FACTORS_CSV,
        "F-F_Momentum_Factor_daily.CSV": _FF_MOM_CSV,
    }

    def fake_get(url, timeout=None):
        name, text = ("F-F_Research_Data_Factors_daily.CSV", _FF_FACTORS_CSV) \
            if "Momentum" not in url else ("F-F_Momentum_Factor_daily.CSV", _FF_MOM_CSV)
        return _Resp(_zip_bytes(name, text))

    import requests
    monkeypatch.setattr(requests, "get", fake_get)
    loader = KenFrenchLoader()
    raw = loader.fetch("2020-01-01", "2020-01-03")  # clip drops the 2020-01-06 row
    assert set(raw) == {"factors", "momentum"}
    assert len(raw["factors"]) == 2 and len(raw["momentum"]) == 2


# ====================================================== (11) cftc COT dedup on OI
def _cot_instruments() -> pd.DataFrame:
    def row(iid, sym):
        return dict(
            instrument_id=iid, asset_class="commodity", sleeve="commodity_etf",
            symbol=sym, vendor_symbols=json.dumps({"cftc": sym}),
            currency="USD", valid_from="1990-01-01", valid_to="2099-01-01",
            proxy_of=None, sector=None, meta="{}",
        )
    return pd.DataFrame([row("CO:GLD:2004-11-18", "GLD"),
                         row("CO:USO:2006-04-10", "USO")])


def _cot_rec(market, oi, nc_long, nc_short, date="2021-06-08"):
    return {"market_and_exchange_names": market,
            "report_date_as_yyyy_mm_dd": date,
            "noncomm_positions_long_all": nc_long,
            "noncomm_positions_short_all": nc_short,
            "open_interest_all": oi}


def test_cot_dedup_keeps_largest_open_interest():
    # Same Tuesday + GOLD mapped to GLD via three planted variants (report-type +
    # contract listings). Only the deepest (largest OI) survives per (date, proxy).
    raw = [
        _cot_rec("GOLD - COMMODITY EXCHANGE INC.", oi=100.0, nc_long=40, nc_short=10),
        _cot_rec("GOLD - COMMODITY EXCHANGE INC. (COMBINED)", oi=500.0, nc_long=90, nc_short=20),
        _cot_rec("GOLD - COMMODITY EXCHANGE INC. (MICRO)", oi=25.0, nc_long=5, nc_short=1),
        _cot_rec("CRUDE OIL, LIGHT SWEET - NEW YORK MERC", oi=300.0, nc_long=70, nc_short=30),
    ]
    loader = CftcCotLoader(instruments=_cot_instruments())
    out = loader.transform(raw)
    gold = out[out["instrument_id"] == "CO:GLD:2004-11-18"]
    assert len(gold) == 1
    assert gold["open_interest"].iloc[0] == 500.0     # deepest listing kept
    assert gold["noncomm_net"].iloc[0] == 70.0        # 90 - 20 from that same row
    # Distinct proxies coexist; the key is unique per (obs_date, instrument_id).
    assert set(out["instrument_id"]) == {"CO:GLD:2004-11-18", "CO:USO:2006-04-10"}
    assert not out.duplicated(subset=["obs_date", "instrument_id"]).any()


def test_cot_fetch_filters_markets_serverside_and_paginates(monkeypatch):
    """Regression (live-ingest 2026-07-07): an unfiltered, un-ordered $limit=50000
    socrata query silently truncated a decade of all-markets COT to an arbitrary 50k
    slice (surfaced as history 'starting in 2019'). fetch() must (a) filter to the
    mapped markets server-side, (b) order deterministically, (c) page past $limit."""
    import sys
    import types

    calls = []

    class _Resp:
        def __init__(self, rows):
            self._rows = rows
        def raise_for_status(self):
            pass
        def json(self):
            return self._rows

    def fake_get(url, params=None, timeout=None):
        calls.append(params)
        limit, offset = params["$limit"], params["$offset"]
        total = limit + 3            # force exactly one extra page
        rows = [{"i": k} for k in range(offset, min(offset + limit, total))]
        return _Resp(rows)

    mod = types.ModuleType("requests")
    mod.get = fake_get
    monkeypatch.setitem(sys.modules, "requests", mod)

    loader = CftcCotLoader(instruments=_cot_instruments())
    out = loader.fetch("2016-01-01", "2026-07-07")
    assert len(out) == loader.PAGE_LIMIT + 3          # both pages stitched
    where = calls[0]["$where"]
    assert "starts_with(market_and_exchange_names, 'GOLD')" in where
    assert "starts_with(market_and_exchange_names, 'EURO FX')" in where
    assert calls[0]["$order"] == "report_date_as_yyyy_mm_dd"
    assert [c["$offset"] for c in calls] == [0, loader.PAGE_LIMIT]


def test_cot_dedup_survives_audit_no_duplicates():
    raw = [
        _cot_rec("GOLD - COMMODITY EXCHANGE INC.", oi=100.0, nc_long=40, nc_short=10),
        _cot_rec("GOLD - COMMODITY EXCHANGE INC. (COMBINED)", oi=500.0, nc_long=90, nc_short=20),
    ]
    loader = CftcCotLoader(instruments=_cot_instruments())
    out = loader.transform(raw)
    stamped = stamp_availability(
        out.assign(source="test", ingested_at=pd.Timestamp.now(tz=UTC)),
        loader.availability_rule)
    rep = audit(stamped, loader.expectations)
    assert rep.checks["duplicates"]["ok"] is True


# ================================================== (11) raw-zone snapshot frames
def test_raw_frame_dict_of_json_payloads_is_row_per_key():
    """Regression (live-ingest 2026-07-07): a universe-scale {symbol: companyfacts}
    dict serialized into ONE parquet cell blew parquet's 2GB string cap. Dict payloads
    of non-DataFrame values must snapshot as one row per key."""
    from production.data.base import _to_raw_frame

    payload = {"AAPL": {"cik": 320193, "facts": {"x": 1}},
               "MSFT": {"cik": 789019, "facts": {"y": 2}}}
    df = _to_raw_frame(payload)
    assert list(df["__key__"]) == ["AAPL", "MSFT"]
    assert len(df) == 2
    # 64-bit offsets: a universe of multi-MB JSON blobs exceeds 2 GB per COLUMN chunk,
    # which regular (32-bit-offset) pyarrow strings cannot hold. Built directly as an
    # arrow large_string array — pandas astype routes through a 32-bit intermediate.
    import pyarrow as pa
    assert df["payload"].dtype == pd.ArrowDtype(pa.large_string())
    assert json.loads(df["payload"][0])["cik"] == 320193


def test_raw_snapshot_failure_degrades_to_key_index(tmp_path, monkeypatch):
    """Regression (live-ingest 2026-07-07): three consecutive 25-minute SEC pulls died
    AT THE RAW SNAPSHOT after the fetch succeeded. run() must degrade to a key-index
    snapshot with a loud warning, never discard the completed fetch."""
    from production.core.lake import Lake
    from production.data.loaders.yfinance_prices import YFinancePricesLoader

    lake = Lake(str(tmp_path))
    loader = YFinancePricesLoader(lake, _instruments(), symbols=["AAA"])
    dates = pd.date_range("2020-01-02", periods=3, freq="B")
    monkeypatch.setattr(loader, "fetch", lambda s, e: _yf_payload(dates, ["AAA"]))

    real_write_raw = lake.write_raw
    calls = {"n": 0}
    def flaky_write_raw(df, vendor, dataset, ts):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ValueError("array cannot contain more than 2147483646 bytes")
        return real_write_raw(df, vendor, dataset, ts)
    monkeypatch.setattr(lake, "write_raw", flaky_write_raw)

    res = loader.run("2020-01-01", "2020-01-10", incremental=False)
    assert res.rows == 3                               # curated write still landed
    assert any("raw snapshot failed" in w for w in res.audit["warnings"])
    assert calls["n"] == 2                             # fallback snapshot written


def test_raw_frame_list_with_mixed_type_column_is_parquet_safe(tmp_path):
    """Regression (live-ingest 2026-07-07): DefiLlama's chains list carries chainId as
    int for some rows and str for others; the raw snapshot write must not crash."""
    from production.core.lake import Lake
    from production.data.base import _to_raw_frame

    payload = [{"name": "Corn", "chainId": 21000000, "tvl": 0.0},
               {"name": "Unit", "chainId": "11235", "tvl": 1.5},
               {"name": "NoId", "chainId": None, "tvl": 2.0}]
    df = _to_raw_frame(payload)
    lake = Lake(str(tmp_path))
    path = lake.write_raw(df, "defillama", "defi_tvl", pd.Timestamp.now(tz=UTC))
    assert path.exists()


# ============================================= (12) ccxt OHLCV pagination + depth
def _mk_daily_bars(start: str, n: int) -> list:
    t0 = int(pd.Timestamp(start).tz_localize(UTC).timestamp() * 1000)
    day = 24 * 3600 * 1000
    return [[t0 + k * day, 1.0, 1.1, 0.9, 1.0 + k * 0.01, 100.0] for k in range(n)]


class _TruncatedVenue:
    """Kraken-style: returns its LAST `retention` daily bars regardless of `since`."""
    def __init__(self, bars, retention=720):
        self.bars, self.retention = bars, retention

    def fetch_ohlcv(self, sym, timeframe="1d", since=None):
        return self.bars[-self.retention:]


class _PagedVenue:
    """Coinbase-style: honors `since`, pages of `page` bars."""
    def __init__(self, bars, page=300):
        self.bars, self.page = bars, page

    def fetch_ohlcv(self, sym, timeframe="1d", since=None):
        rows = [b for b in self.bars if since is None or b[0] >= since]
        return rows[:self.page]


def test_fetch_ohlcv_paginated_stitches_pages(monkeypatch):
    from production.data.base import fetch_ohlcv_paginated
    monkeypatch.setattr("time.sleep", lambda s: None)

    bars = _mk_daily_bars("2016-01-01", 1000)
    got = fetch_ohlcv_paginated(_PagedVenue(bars), "BTC/USD",
                                since=bars[0][0], until=bars[-1][0])
    assert len(got) == 1000                       # 4 pages stitched
    assert len({b[0] for b in got}) == 1000       # no duplicated boundary bars


def test_crypto_prices_fallback_wins_when_primary_truncates(monkeypatch):
    """Regression (live-ingest 2026-07-07): kraken serves only its last ~720 daily
    bars, so crypto prices silently started 2024-07. The loader must consult the
    fallback venue and keep the deeper series."""
    import sys
    import types

    from production.data.loaders.ccxt_prices import CcxtPricesLoader

    monkeypatch.setattr("time.sleep", lambda s: None)
    deep = _mk_daily_bars("2016-01-01", 3800)     # ~2016 -> 2026
    mod = types.ModuleType("ccxt")
    mod.kraken = lambda: _TruncatedVenue(deep, retention=720)
    mod.coinbase = lambda: _PagedVenue(deep, page=300)
    monkeypatch.setitem(sys.modules, "ccxt", mod)

    loader = CcxtPricesLoader(symbols=["BTC/USD"])
    raw = loader.fetch("2016-01-01", "2026-07-07")
    bars = raw["BTC/USD"]
    assert bars[0][0] == deep[0][0]               # reached the requested start
    assert len(bars) == 3800


class _VariantDepthVenue:
    """okx-style: a venue can list more than one quote-currency pair for the same
    base asset with very different depth (okx added direct SYM/USD spot markets
    across most alts on 2025-01-15, ~500 daily bars, while the long-standing
    SYM/USDT pair reaches back to ~2020, ~2300+ bars). Each variant still honors
    `since` correctly (ascending) — the two symbols just have different true depth."""
    def __init__(self, series: dict):
        self.series = series  # {vendor_symbol: bars}

    def fetch_ohlcv(self, sym, timeframe="1d", since=None):
        bars = self.series.get(sym, [])
        return [b for b in bars if since is None or b[0] >= since]


def test_basis_try_ohlcv_prefers_deepest_quote_variant(monkeypatch):
    """Regression (live-probe 2026-07-11): _try_ohlcv accepted the FIRST non-empty
    candidate ("SYM/USD" is tried before "SYM/USDT"), so once bybit — the venue with
    genuinely deep listings — became geo-blocked (Amazon CloudFront country block,
    confirmed live) and every symbol fell through to okx, the loader silently locked
    onto okx's shallow SYM/USD listing for nearly every symbol: basis fell from
    33,567 to 11,347 rows with NO warning, because each symbol still "succeeded",
    just on a few hundred days of history instead of the ~2020+ SYM/USDT depth.
    Live-probed on okx: BTC/USD only reaches back to 2024-12-06 (583 bars) while
    BTC/USDT reaches 2018-01-11 (3104 bars); same 543-vs-2384 pattern reproduced
    for DOGE and LINK. _try_ohlcv must keep the DEEPEST candidate, not the first."""
    monkeypatch.setattr("time.sleep", lambda s: None)
    from production.data.loaders.ccxt_perp_basis import CcxtPerpBasisLoader

    deep = _mk_daily_bars("2020-01-01", 2000)      # BTC/USDT-style: reaches back to 2020
    shallow = _mk_daily_bars("2025-01-15", 500)    # BTC/USD-style: only ~500 recent bars
    venue = _VariantDepthVenue({"BTC/USD": shallow, "BTC/USDT": deep})

    got = CcxtPerpBasisLoader._try_ohlcv(venue, ["BTC/USD", "BTC/USDT"], since=deep[0][0])
    assert got is not None
    assert got[0][0] == deep[0][0]                 # the deep series won, not the shallow first hit
    assert len(got) == len(deep)


def test_basis_try_ohlcv_early_exits_when_first_candidate_is_already_deep(monkeypatch):
    """When the first-tried candidate already reaches (near) the requested start,
    a second full pagination of the next candidate is skipped (matches
    CcxtPricesLoader's depth-slack early-exit) — asserted via call count, not just
    depth, so a regression that always tries every candidate doesn't slip by."""
    monkeypatch.setattr("time.sleep", lambda s: None)
    from production.data.loaders.ccxt_perp_basis import CcxtPerpBasisLoader

    deep = _mk_daily_bars("2020-01-01", 2000)
    calls = []

    class _CountingVenue(_VariantDepthVenue):
        def fetch_ohlcv(self, sym, timeframe="1d", since=None):
            calls.append(sym)
            return super().fetch_ohlcv(sym, timeframe, since)

    venue = _CountingVenue({"BTC/USD": deep, "BTC/USDT": deep})
    got = CcxtPerpBasisLoader._try_ohlcv(venue, ["BTC/USD", "BTC/USDT"], since=deep[0][0])
    assert got[0][0] == deep[0][0]
    assert "BTC/USDT" not in calls                 # never consulted — BTC/USD was deep enough


# ==================================================== (13) tiingo secondary feed
def test_tiingo_rows_to_long_uses_adjusted_fields():
    """Canned payload of the REAL tiingo shape (live probe 2026-07-07): adjClose /
    adjVolume must feed close/volume (total-return semantics, like yfinance)."""
    from production.data.loaders.tiingo_prices import TiingoPricesLoader

    rows = [
        {"date": "2016-01-04T00:00:00.000Z", "close": 105.35, "high": 105.368,
         "low": 102.0, "open": 102.61, "volume": 67649387,
         "adjClose": 23.7083102515, "adjHigh": 23.71236, "adjLow": 22.95441,
         "adjOpen": 23.09169, "adjVolume": 270597548, "divCash": 0.0,
         "splitFactor": 1.0},
        {"date": "2016-01-05T00:00:00.000Z", "close": 102.71, "high": 105.85,
         "low": 102.41, "open": 105.75, "volume": 55790992,
         "adjClose": 23.1141959747, "adjHigh": 23.82083, "adjLow": 23.04668,
         "adjOpen": 23.79832, "adjVolume": 223163968, "divCash": 0.0,
         "splitFactor": 1.0},
    ]
    loader = TiingoPricesLoader(instruments=_instruments(), api_key="test-key")
    ohlcv = loader._to_ohlcv(rows)
    long = loader.transform({"AAA": ohlcv})

    assert set(long["instrument_id"]) == {"EQ:AAA:2000-01-03"}
    assert len(long) == 2
    np.testing.assert_allclose(long["close"].iloc[0], 23.7083102515)
    np.testing.assert_allclose(long["volume"].iloc[0], 270597548)
    np.testing.assert_allclose(long["dollar_volume"], long["close"] * long["volume"])


def test_tiingo_requires_api_key_and_guards_quota(monkeypatch):
    from production.data.loaders.tiingo_prices import TiingoPricesLoader

    loader = TiingoPricesLoader(instruments=_instruments(), symbols=["AAA"], api_key="")
    with pytest.raises(IngestError):
        loader.fetch("2016-01-01", "2016-02-01")

    # A 429 mid-universe stops the pull with a warning instead of burning the quota.
    import sys
    import types

    class _Resp:
        def __init__(self, code, payload=None):
            self.status_code = code
            self._payload = payload or []
        def raise_for_status(self):
            pass
        def json(self):
            return self._payload

    calls = []
    def fake_get(url, params=None, timeout=None):
        calls.append(url)
        if len(calls) == 1:
            return _Resp(200, [{"date": "2016-01-04T00:00:00.000Z",
                                "adjClose": 10.0, "adjVolume": 100, "volume": 100}])
        return _Resp(429)

    mod = types.ModuleType("requests")
    mod.get = fake_get
    monkeypatch.setitem(sys.modules, "requests", mod)
    monkeypatch.setattr("time.sleep", lambda s: None)

    loader = TiingoPricesLoader(instruments=_instruments(),
                                symbols=["AAA", "BBB", "CCC"], api_key="k", pause_s=0)
    raw = loader.fetch("2016-01-01", "2016-02-01")
    assert list(raw) == ["AAA"]                    # kept what landed before the 429
    assert len(calls) == 2                         # stopped immediately at the 429
    assert any("rate-limited" in w for w in loader.warnings)


def test_tiingo_fetches_uncovered_symbols_first(tmp_lake, monkeypatch):
    """Frontier ordering: symbols already carrying tiingo-sourced lake rows are
    deprioritized, so the ~50-request hourly quota EXTENDS coverage each run
    instead of re-fetching the same head of the list forever (found 2026-07-11:
    two successive runs pulled the same ~54 names)."""
    import sys
    import types

    from production.data.loaders.tiingo_prices import TiingoPricesLoader

    # AAA already covered by a prior tiingo pull
    seed = pd.DataFrame({
        "obs_date": [pd.Timestamp("2016-01-04")],
        "instrument_id": ["EQ:AAA:2000-01-03"],
        "close": [10.0], "volume": [1.0], "dollar_volume": [10.0],
        "available_from": [pd.Timestamp("2016-01-04 21:15", tz="UTC")],
        "source": ["tiingo:prices"],
        "ingested_at": [pd.Timestamp("2026-01-01", tz="UTC")],
    })
    tmp_lake.write_curated(seed, "prices", "equity")

    calls = []

    class _Resp:
        status_code = 200
        def raise_for_status(self):
            pass
        def json(self):
            return []

    def fake_get(url, params=None, timeout=None):
        calls.append(url)
        return _Resp()

    mod = types.ModuleType("requests")
    mod.get = fake_get
    monkeypatch.setitem(sys.modules, "requests", mod)
    monkeypatch.setattr("time.sleep", lambda s: None)

    loader = TiingoPricesLoader(tmp_lake, instruments=_instruments(),
                                symbols=["AAA", "BBB"], api_key="k", pause_s=0)
    loader.fetch("2016-01-01", "2016-02-01")
    # BBB (uncovered) must be requested before AAA (covered)
    assert len(calls) == 2
    assert "BBB" in calls[0] and "AAA" in calls[1]


def test_tiingo_404s_are_remembered_and_deprioritized(tmp_lake, monkeypatch):
    """Negative-result memory: an all-404 tranche (delisted names — found live
    2026-07-11 when the frontier reached the dead stretch of the alphabet) must
    record the symbols in the tiingo_unavailable reference BEFORE transform/audit
    can fail, and the next run's ordering must put them LAST (behind covered)."""
    import sys
    import types

    from production.data.loaders.tiingo_prices import TiingoPricesLoader

    class _Resp:
        def __init__(self, code, payload=None):
            self.status_code = code
            self._payload = payload or []
        def raise_for_status(self):
            pass
        def json(self):
            return self._payload

    calls = []

    def fake_get(url, params=None, timeout=None):
        calls.append(url)
        if "DEADX" in url or "DEADY" in url:
            return _Resp(404)
        return _Resp(200, [{"date": "2016-01-04T00:00:00.000Z",
                            "adjClose": 10.0, "adjVolume": 100, "volume": 100}])

    mod = types.ModuleType("requests")
    mod.get = fake_get
    monkeypatch.setitem(sys.modules, "requests", mod)
    monkeypatch.setattr("time.sleep", lambda s: None)

    # run 1: all-404 tranche — memory persists even though nothing was fetched
    loader = TiingoPricesLoader(tmp_lake, instruments=_instruments(),
                                symbols=["DEADX", "DEADY"], api_key="k", pause_s=0)
    raw = loader.fetch("2016-01-01", "2016-02-01")
    assert raw == {}
    ref = tmp_lake.read_reference("tiingo_unavailable")
    assert set(ref["symbol"]) == {"DEADX", "DEADY"}
    assert any("recorded 2 new" in w for w in loader.warnings)

    # run 2: known-404 names go LAST, after the never-tried live name
    calls.clear()
    loader2 = TiingoPricesLoader(tmp_lake, instruments=_instruments(),
                                 symbols=["DEADX", "AAA", "DEADY"], api_key="k",
                                 pause_s=0)
    loader2.fetch("2016-01-01", "2016-02-01")
    assert "AAA" in calls[0]
    assert all(("DEADX" in c) or ("DEADY" in c) for c in calls[1:])


# ===================================================== (14) alpaca secondary feed
def test_alpaca_bars_to_long_and_pagination(monkeypatch):
    """Canned payload of the REAL alpaca shape (live probe 2026-07-07): {bars:
    {SYM: [{t,o,h,l,c,v},...]}, next_page_token} with adjustment=all semantics.
    Pagination must stitch pages; symbols resolve via the 'alpaca' vendor key or
    the master's plain (dotted) symbol."""
    import sys
    import types

    from production.data.loaders.alpaca_prices import AlpacaPricesLoader

    pages = [
        {"bars": {"AAA": [
            {"t": "2021-01-04T05:00:00Z", "o": 10.0, "h": 11.0, "l": 9.5,
             "c": 10.5, "v": 1000, "n": 5, "vw": 10.4}]},
         "next_page_token": "tok1"},
        {"bars": {"AAA": [
            {"t": "2021-01-05T05:00:00Z", "o": 10.5, "h": 11.5, "l": 10.0,
             "c": 11.0, "v": 2000, "n": 6, "vw": 10.9}]},
         "next_page_token": None},
    ]
    calls = []

    class _Resp:
        def __init__(self, payload):
            self._payload = payload
        def raise_for_status(self):
            pass
        def json(self):
            return self._payload

    def fake_get(url, params=None, headers=None, timeout=None):
        calls.append(dict(params))
        return _Resp(pages[len(calls) - 1])

    mod = types.ModuleType("requests")
    mod.get = fake_get
    monkeypatch.setitem(sys.modules, "requests", mod)
    monkeypatch.setattr("time.sleep", lambda s: None)

    loader = AlpacaPricesLoader(instruments=_instruments(), symbols=["AAA"],
                                key_id="k", secret="s", pause_s=0)
    raw = loader.fetch("2021-01-01", "2021-02-01")
    assert len(calls) == 2 and calls[1].get("page_token") == "tok1"

    long = loader.transform(raw)
    assert set(long["instrument_id"]) == {"EQ:AAA:2000-01-03"}
    assert len(long) == 2
    np.testing.assert_allclose(long["dollar_volume"], long["close"] * long["volume"])


def test_alpaca_requires_credentials():
    from production.data.loaders.alpaca_prices import AlpacaPricesLoader

    loader = AlpacaPricesLoader(symbols=["AAA"], key_id="", secret="")
    with pytest.raises(IngestError):
        loader.fetch("2021-01-01", "2021-02-01")
