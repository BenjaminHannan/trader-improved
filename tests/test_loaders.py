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
    assert loader.hygiene_drops == {"backstop_dropped": 1, "blocklist_dropped": 2}
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
