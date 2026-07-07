"""Signal-bundle loader tests for ``production.core.lake.read_signal_bundle``.

The factor-gate (``scripts/build_factors.py``) and backtest (``scripts/run_backtest.py``)
CLIs feed their signals through this loader. These tests pin that:

1. every bundle key the newer signals declare (``basis``/``fundamentals``/``mcap``/``tvl``
   alongside ``prices``/``funding``/``macro``/``cot``) is picked up when present in the
   lake, and absent datasets are simply skipped (non-fatal);
2. the two crypto snapshot loaders write differently-named curated datasets whose columns
   also differ from the signal contract, and the loader maps them:
     - CoinGecko  curated ``crypto_meta`` (col ``market_cap``) -> bundle ``mcap`` (col ``mcap``)
     - DefiLlama  curated ``defi_tvl``    (col ``value``)      -> bundle ``tvl``  (col ``tvl``)
   and a series-keyed (no ``instrument_id``) TVL frame is NOT admitted, because the
   per-instrument signal contract can't consume chain-level series.

NO NETWORK: everything is written to a ``tmp_path`` lake.
"""
from __future__ import annotations

import pandas as pd
import pytest

from production.core.lake import Lake, read_signal_bundle

UTC = "UTC"


def _mandatory(dates: pd.DatetimeIndex) -> dict:
    avail = dates.tz_localize(UTC) + pd.Timedelta(hours=24)
    return {"available_from": avail, "source": "test",
            "ingested_at": avail + pd.Timedelta(minutes=5)}


def _instrument_frame(iid: str, dates: pd.DatetimeIndex, **cols) -> pd.DataFrame:
    df = pd.DataFrame({"obs_date": dates, "instrument_id": iid, **cols})
    return df.assign(**_mandatory(dates))


def _series_frame(series_id: str, dates: pd.DatetimeIndex, **cols) -> pd.DataFrame:
    df = pd.DataFrame({"obs_date": dates, "series_id": series_id, **cols})
    return df.assign(**_mandatory(dates))


@pytest.fixture()
def lake(tmp_path) -> Lake:
    return Lake(tmp_path / "data")


def test_new_datasets_admitted_under_bundle_keys(lake):
    """basis + fundamentals + mcap (from crypto_meta) all show up in the bundle."""
    dates = pd.date_range("2020-01-02", periods=5, freq="D")
    lake.write_curated(_instrument_frame("EQ:AAA:2000-01-03", dates, close=100.0),
                       "prices", "equity")
    lake.write_curated(_instrument_frame("CR:ETH:2017-01-01", dates, basis=0.01),
                       "basis", "crypto")
    lake.write_curated(_instrument_frame("EQ:AAA:2000-01-03", dates, field="eps", value=1.5),
                       "fundamentals", "equity")
    # CoinGecko writes curated 'crypto_meta' with a 'market_cap' column.
    lake.write_curated(_instrument_frame("CR:ETH:2017-01-01", dates, market_cap=1e9),
                       "crypto_meta", "crypto")

    bundle = read_signal_bundle(lake)

    assert set(bundle) == {"prices", "basis", "fundamentals", "mcap"}
    # mcap bundle key was mapped from the crypto_meta dataset and renamed to the
    # signal-contract column.
    assert "mcap" in bundle["mcap"].columns
    assert "market_cap" not in bundle["mcap"].columns
    assert (bundle["mcap"]["mcap"] == 1e9).all()


def test_missing_datasets_are_non_fatal(lake):
    """A lake with only prices still yields a usable, smaller bundle."""
    dates = pd.date_range("2020-01-02", periods=3, freq="D")
    lake.write_curated(_instrument_frame("EQ:AAA:2000-01-03", dates, close=100.0),
                       "prices", "equity")
    bundle = read_signal_bundle(lake)
    assert set(bundle) == {"prices"}


def test_instrument_keyed_tvl_is_mapped_and_renamed(lake):
    """defi_tvl with an instrument_id + 'value' col -> bundle 'tvl' with 'tvl' col."""
    dates = pd.date_range("2020-01-02", periods=4, freq="D")
    lake.write_curated(_instrument_frame("EQ:AAA:2000-01-03", dates, close=100.0),
                       "prices", "equity")
    lake.write_curated(_instrument_frame("CR:ETH:2017-01-01", dates, value=2e8),
                       "defi_tvl", "crypto")
    bundle = read_signal_bundle(lake)
    assert "tvl" in bundle
    assert "tvl" in bundle["tvl"].columns
    assert "value" not in bundle["tvl"].columns
    assert (bundle["tvl"]["tvl"] == 2e8).all()


def test_series_keyed_tvl_is_not_admitted(lake):
    """DefiLlama's real shape is chain-level series (series_id, no instrument_id).

    Such a frame cannot satisfy the per-instrument 'tvl' signal contract, so it is
    skipped rather than admitted with a wrong key.
    """
    dates = pd.date_range("2020-01-02", periods=4, freq="D")
    lake.write_curated(_instrument_frame("EQ:AAA:2000-01-03", dates, close=100.0),
                       "prices", "equity")
    lake.write_curated(_series_frame("TVL_Ethereum", dates, value=2e8),
                       "defi_tvl", "macro")
    bundle = read_signal_bundle(lake)
    assert "tvl" not in bundle
    assert set(bundle) == {"prices"}
