"""Basis loader + BasisCarry signal — NO NETWORK.

The loader's `fetch` is monkeypatched (or ccxt is replaced with a fake module) so the
spot/swap alignment arithmetic, the 24h availability stamp, and the per-symbol
degrade-with-warning path are all exercised without touching an exchange. The signal is
checked against a hand-built constant-basis panel where the trailing-mean carry is
analytically determined.
"""
from __future__ import annotations

import json
import sys
import types

import numpy as np
import pandas as pd
import pytest

from production.data.loaders.ccxt_perp_basis import CcxtPerpBasisLoader
from production.signals.base import all_signals, sleeve_from_id

UTC = "UTC"
IID = "CR:BTC:2017-01-01"


# --------------------------------------------------------------------- helpers
def _crypto_instruments() -> pd.DataFrame:
    """Instrument master with BTC's ccxt spot symbol -> synthetic id."""
    return pd.DataFrame([dict(
        instrument_id=IID, asset_class="crypto", sleeve="crypto", symbol="BTC",
        vendor_symbols=json.dumps({"ccxt": "BTC/USD"}),
        currency="USD", valid_from="1990-01-01", valid_to="2099-01-01",
        proxy_of=None, sector=None, meta="{}",
    )])


def _ohlcv(dates, closes) -> list:
    """ccxt-style [ts_ms, o, h, l, c, v] rows on midnight-UTC daily bars."""
    ts = [int(pd.Timestamp(d).tz_localize(UTC).timestamp() * 1000) for d in dates]
    return [[t, c, c, c, float(c), 1000.0] for t, c in zip(ts, closes)]


class _FakeEx:
    """A ccxt exchange whose fetch_ohlcv serves a fixed {symbol: bars} map."""

    def __init__(self, book: dict):
        self._book = book

    def fetch_ohlcv(self, symbol, timeframe="1d", since=None):
        if symbol not in self._book:
            raise Exception(f"{symbol} not listed")
        return self._book[symbol]


def _install_fake_ccxt(monkeypatch, bybit_book, okx_book=None):
    fake = types.ModuleType("ccxt")
    fake.bybit = lambda: _FakeEx(bybit_book)
    fake.okx = lambda: _FakeEx(okx_book or {})
    monkeypatch.setitem(sys.modules, "ccxt", fake)


# =========================================================== (1) loader transform
def test_transform_basis_arithmetic_exact():
    dates = pd.date_range("2021-01-01", periods=4, freq="D")
    spot = _ohlcv(dates, [100.0, 200.0, 50.0, 100.0])
    swap = _ohlcv(dates, [101.0, 201.0, 49.5, 100.0])  # +1%, +0.5%, -1%, 0%
    loader = CcxtPerpBasisLoader(instruments=_crypto_instruments(), symbols=["BTC"])
    long = loader.transform({"BTC": {"spot": spot, "swap": swap}})

    assert set(long["instrument_id"]) == {IID}
    assert list(long.columns) >= ["obs_date", "instrument_id", "basis"]
    np.testing.assert_allclose(
        long.sort_values("obs_date")["basis"].to_numpy(),
        [101 / 100 - 1, 201 / 200 - 1, 49.5 / 50 - 1, 0.0])
    assert set(long["asset_class"]) == {"crypto"}


def test_transform_aligns_on_bar_date_inner_join():
    # spot has an extra trailing day the swap lacks -> that day is dropped (no basis).
    sdates = pd.date_range("2021-01-01", periods=3, freq="D")
    wdates = pd.date_range("2021-01-01", periods=2, freq="D")
    loader = CcxtPerpBasisLoader(instruments=_crypto_instruments(), symbols=["BTC"])
    long = loader.transform({"BTC": {
        "spot": _ohlcv(sdates, [100.0, 100.0, 100.0]),
        "swap": _ohlcv(wdates, [102.0, 103.0]),
    }})
    assert len(long) == 2
    np.testing.assert_allclose(long.sort_values("obs_date")["basis"].to_numpy(),
                               [0.02, 0.03])


# ================================================ (2) run end-to-end + availability
def test_run_stamps_availability_obs_plus_24h(tmp_lake, monkeypatch):
    dates = pd.date_range("2021-03-01", periods=5, freq="D")
    raw = {"BTC": {"spot": _ohlcv(dates, [100.0] * 5),
                   "swap": _ohlcv(dates, [101.0] * 5)}}
    loader = CcxtPerpBasisLoader(tmp_lake, _crypto_instruments(), symbols=["BTC"])
    monkeypatch.setattr(loader, "fetch", lambda start, end: raw)

    res = loader.run("2021-03-01", "2021-03-31", incremental=False)
    assert res.rows == 5

    cur = tmp_lake.read_curated("basis", "crypto").sort_values("obs_date")
    first = cur.iloc[0]
    # basis = 101/100 - 1 = 0.01, and available_from = bar close + 24h.
    assert first["basis"] == pytest.approx(0.01)
    assert first["available_from"] == pd.Timestamp("2021-03-02 00:00", tz=UTC)
    assert first["obs_date"] == pd.Timestamp("2021-03-01")


# ==================================================== (3) per-symbol degradation
def test_fetch_degrades_per_symbol_with_warning(monkeypatch):
    dates = pd.date_range("2021-01-01", periods=3, freq="D")
    # bybit lists BTC (spot+swap) but only ETH spot (no ETH perp) -> ETH degraded.
    bybit = {
        "BTC/USD": _ohlcv(dates, [100.0, 100.0, 100.0]),
        "BTC/USDT:USDT": _ohlcv(dates, [101.0, 101.0, 101.0]),
        "ETH/USD": _ohlcv(dates, [10.0, 10.0, 10.0]),
    }
    _install_fake_ccxt(monkeypatch, bybit)  # okx empty -> no fallback coverage
    loader = CcxtPerpBasisLoader(instruments=_crypto_instruments(), symbols=["BTC", "ETH"])

    raw = loader.fetch("2021-01-01", "2021-01-10")
    assert set(raw) == {"BTC"}                       # ETH dropped, BTC survives
    assert any("ETH" in w for w in loader.warnings)  # and it is audited, not silent
    assert not any("BTC" in w for w in loader.warnings)


def test_fetch_uses_usdt_spot_fallback(monkeypatch):
    # No SYM/USD spot on the venue, but SYM/USDT exists -> loader still builds basis.
    dates = pd.date_range("2021-01-01", periods=2, freq="D")
    bybit = {
        "BTC/USDT": _ohlcv(dates, [100.0, 100.0]),
        "BTC/USDT:USDT": _ohlcv(dates, [101.0, 101.0]),
    }
    _install_fake_ccxt(monkeypatch, bybit)
    loader = CcxtPerpBasisLoader(instruments=_crypto_instruments(), symbols=["BTC"])
    raw = loader.fetch("2021-01-01", "2021-01-10")
    assert set(raw) == {"BTC"}
    assert loader.warnings == []


# =================================================== (4) signal known answer
def _panel_with_mandatory(df: pd.DataFrame, offset: pd.Timedelta, source: str) -> pd.DataFrame:
    avail = pd.to_datetime(df["obs_date"]).dt.tz_localize(UTC) + offset
    return df.assign(available_from=avail, source=source,
                     ingested_at=avail + pd.Timedelta(minutes=5))


def _const_basis_bundle(b: float, n: int = 60) -> dict[str, pd.DataFrame]:
    dates = pd.date_range("2021-01-01", periods=n, freq="D")
    prices = _panel_with_mandatory(pd.DataFrame({
        "obs_date": dates, "instrument_id": IID,
        "close": 100.0, "volume": 1e6, "dollar_volume": 1e8,
    }), pd.Timedelta(hours=24), "test:prices")
    basis = _panel_with_mandatory(pd.DataFrame({
        "obs_date": dates, "instrument_id": IID, "basis": b,
    }), pd.Timedelta(hours=24), "test:basis")
    return {"prices": prices, "basis": basis}


def test_basis_carry_constant_panel_is_negated_mean():
    b = 0.004
    out = all_signals()["basis_carry"]().compute(_const_basis_bundle(b))
    assert not out.empty
    # constant basis => trailing 7d mean == b => value == -b everywhere emitted.
    np.testing.assert_allclose(out["value"].to_numpy(), -b, rtol=1e-12)
    assert set(out["instrument_id"]) == {IID}


def test_basis_carry_respects_24h_availability():
    """Value on price date D must use only basis rows knowable by end of day D.

    Basis available_from = obs + 24h, so a spike on basis date B can only surface on
    price date B+1 onward — never on B itself.
    """
    bundle = _const_basis_bundle(0.001, n=40)
    basis = bundle["basis"].copy()
    spike_date = pd.Timestamp("2021-01-20")
    basis.loc[basis["obs_date"] == spike_date, "basis"] = 5.0  # huge one-day jump
    bundle["basis"] = basis

    out = all_signals()["basis_carry"]().compute(bundle).set_index("obs_date")["value"]
    # On the spike date itself the value is still the pre-spike mean (spike not knowable).
    assert out.loc[spike_date] == pytest.approx(-0.001, abs=1e-9)
    # The day after, the spike has become knowable and moves the trailing mean.
    assert out.loc[spike_date + pd.Timedelta(days=1)] < -0.001


def test_basis_carry_only_emits_crypto(signal_data):
    out = all_signals()["basis_carry"]().compute(signal_data)
    assert not out.empty
    assert {sleeve_from_id(i) for i in out["instrument_id"].unique()} == {"crypto"}


# ======================================================= (5) ingest CLI smoke
def test_ingest_dataset_basis_smoke(tmp_lake, monkeypatch):
    import scripts.ingest as ingest

    tmp_lake.write_reference(_crypto_instruments(), "instruments")
    dates = pd.date_range("2021-01-01", periods=6, freq="D")
    raw = {"BTC": {"spot": _ohlcv(dates, [100.0] * 6),
                   "swap": _ohlcv(dates, [101.0] * 6)}}
    monkeypatch.setattr(CcxtPerpBasisLoader, "fetch", lambda self, start, end: raw)

    rc = ingest.main(["--dataset", "basis", "--sleeve", "crypto",
                      "--start", "2021-01-01", "--end", "2021-01-31",
                      "--full", "--lake-root", str(tmp_lake.root)])
    assert rc == 0
    cur = tmp_lake.read_curated("basis", "crypto")
    assert len(cur) == 6
    assert cur["basis"].round(6).eq(0.01).all()
