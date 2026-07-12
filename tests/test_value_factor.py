"""Known-answer, availability-lag, and dropout tests for the crypto value factor.

`mcap_tvl` (CryptoMcapTvl) emits ``log(TVL / mcap)`` per crypto instrument, joining the
market-cap and TVL snapshot panels onto each instrument's price dates by availability
date. These tests pin: (1) the exact ratio arithmetic on constant snapshots, (2) that a
snapshot published *after* a decision date is invisible on that date, and (3) that an
instrument with no TVL series (BTC) drops out of the factor entirely.

The registry-wide corruption harness in ``test_no_lookahead.py`` already covers the
signal generically (the VALUE_COLS tuple there was extended with ``mcap``/``tvl``); the
checks here are the analytic complements.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from production.signals.base import all_signals
from production.signals.value import CryptoMcapTvl

UTC = "UTC"


# --------------------------------------------------------------------- builders
def _mandatory(dates: pd.DatetimeIndex, offset: pd.Timedelta, source: str) -> dict:
    avail = dates.tz_localize(UTC) + offset
    return {"available_from": avail, "source": source,
            "ingested_at": avail + pd.Timedelta(minutes=5)}


def _prices(iid: str, dates: pd.DatetimeIndex) -> pd.DataFrame:
    close = 100.0 * (1.0 + 0.001) ** np.arange(len(dates))
    df = pd.DataFrame({"obs_date": dates, "instrument_id": iid,
                       "close": close, "volume": 1e6, "dollar_volume": close * 1e6})
    return df.assign(**_mandatory(dates, pd.Timedelta(hours=21, minutes=30),
                                  "test:prices"))


def _snapshot(iid: str, dates: pd.DatetimeIndex, col: str,
              values) -> pd.DataFrame:
    """A CoinGecko/DefiLlama-shape snapshot frame; available the morning after obs."""
    df = pd.DataFrame({"obs_date": dates, "instrument_id": iid, col: values})
    return df.assign(**_mandatory(dates, pd.Timedelta(hours=30), f"test:{col}"))


# ------------------------------------------------------------------ known answer
def test_ratio_known_answer():
    """Constant mcap/tvl => every emitted value equals log(tvl/mcap) exactly."""
    iid = "CR:ETH:2017-01-01"
    dates = pd.date_range("2018-01-02", periods=80, freq="D")
    mcap, tvl = 1000.0, 200.0
    data = {
        "prices": _prices(iid, dates),
        "mcap": _snapshot(iid, dates, "mcap", mcap),
        "tvl": _snapshot(iid, dates, "tvl", tvl),
    }
    out = CryptoMcapTvl().compute(data)
    assert not out.empty
    assert set(out["instrument_id"]) == {iid}
    expected = np.log(tvl / mcap)
    np.testing.assert_allclose(out["value"].to_numpy(), expected, rtol=1e-12)

    # min_history_days=30: nothing emitted before first_obs + 30 calendar days.
    floor = dates[0] + pd.Timedelta(days=30)
    assert out["obs_date"].min() >= floor


def test_availability_lag_hides_future_snapshot():
    """A snapshot observed on D (published D+1) is invisible in the value at D.

    mcap is a flat 1000 except for a spike to 5000 on one obs_date d0. Its snapshot is
    available only the next morning, so the decision at price date d0 must still use the
    prior (1000) snapshot, while the decision at d0+1 sees the spike.
    """
    iid = "CR:SOL:2017-01-01"
    dates = pd.date_range("2018-01-02", periods=80, freq="D")
    tvl = 200.0
    mcap = np.full(len(dates), 1000.0)
    spike_i = 60
    mcap[spike_i] = 5000.0
    data = {
        "prices": _prices(iid, dates),
        "mcap": _snapshot(iid, dates, "mcap", mcap),
        "tvl": _snapshot(iid, dates, "tvl", tvl),
    }
    s = CryptoMcapTvl().compute(data).set_index("obs_date")["value"]

    d0 = dates[spike_i]
    d1 = dates[spike_i + 1]
    # At d0 the spike snapshot (available d0+1) is not yet knowable -> prior mcap 1000.
    np.testing.assert_allclose(s.loc[d0], np.log(tvl / 1000.0), rtol=1e-12)
    # At d0+1 the spike has landed.
    np.testing.assert_allclose(s.loc[d1], np.log(tvl / 5000.0), rtol=1e-12)


def test_btc_without_tvl_drops_out():
    """BTC carries no TVL series, so it produces no value; ETH (with TVL) survives."""
    btc = "CR:BTC:2017-01-01"
    eth = "CR:ETH:2017-01-01"
    dates = pd.date_range("2018-01-02", periods=80, freq="D")
    prices = pd.concat([_prices(btc, dates), _prices(eth, dates)], ignore_index=True)
    mcap = pd.concat([_snapshot(btc, dates, "mcap", 5e11),
                      _snapshot(eth, dates, "mcap", 2e11)], ignore_index=True)
    tvl = _snapshot(eth, dates, "tvl", 4e10)  # ETH only — no BTC row
    out = CryptoMcapTvl().compute({"prices": prices, "mcap": mcap, "tvl": tvl})

    ids = set(out["instrument_id"])
    assert btc not in ids
    assert eth in ids


def test_registered_and_emits_only_crypto(signal_data):
    """mcap_tvl is in the registry and emits only crypto instruments on the bundle."""
    reg = all_signals()
    assert reg["mcap_tvl"] is CryptoMcapTvl
    out = CryptoMcapTvl().compute(signal_data)
    assert not out.empty
    prefixes = {i.split(":", 1)[0] for i in out["instrument_id"].unique()}
    assert prefixes == {"CR"}
    # BTC has no TVL in the shared fixture -> absent from the output.
    assert not any(":BTC:" in i for i in out["instrument_id"].unique())
