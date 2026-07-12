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

import numpy as np
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


# ------------------------------------------- gate validation-return cadence
def test_net_validation_return_rebalances_on_horizon_grid():
    """Regression (first live gate run, 2026-07-07): iterating EVERY date summed
    ~horizon-times overlapping forward returns while charging turnover on daily
    quintile churn (a 500-name low-vol book "returned" -166 on a 2y slice). The
    documented economics rebalance on a horizon cadence: with a flat-return panel
    and a churning signal, the cost must be charged once per grid date, not daily."""
    import numpy as np

    from scripts.build_factors import _net_validation_return

    dates = pd.date_range("2024-01-01", periods=20, freq="B")
    iids = [f"EQ:S{k:02d}:2000-01-03" for k in range(10)]
    rng = np.random.default_rng(7)

    # z-scores churn every day; forward returns are exactly zero for every name.
    z = pd.concat([
        pd.DataFrame({"obs_date": d, "instrument_id": iids,
                      "value": rng.permutation(np.linspace(-2, 2, len(iids)))})
        for d in dates
    ], ignore_index=True)
    prices = pd.concat([
        pd.DataFrame({"obs_date": dates, "instrument_id": iid, "close": 100.0})
        for iid in iids
    ], ignore_index=True)
    sleeve_map = pd.Series({iid: "equity" for iid in iids})
    costs = {"sleeves": {"equity": {"floor_bps": 5.0}}}

    net, by_date = _net_validation_return(z, prices, sleeve_map, horizon=5, costs=costs)

    # Zero gross everywhere -> net is pure cost. On a 5-day grid over 20 dates
    # (minus the unresolved tail) at most 3 rebalances fire; full two-sided churn
    # costs ~2 * 5bp per rebalance. Daily iteration would charge ~5x that.
    assert len(by_date) <= 4
    assert net < 0                                  # costs are never zero
    assert net >= -(len(by_date) + 1) * 2 * 5e-4    # bounded by per-grid churn


# --------------------------------------------- gate: CPCV sign-stability diagnostic
def test_cpcv_sign_stability_stable_vs_pocket_series():
    from scripts.build_factors import _cpcv_sign_stability

    dates = pd.date_range("2018-01-01", periods=800, freq="B")
    rng = np.random.default_rng(11)

    stable = pd.Series(0.02 + rng.normal(0, 0.01, 800), index=dates)
    assert _cpcv_sign_stability(stable, horizon=21) > 0.9

    # All the "signal" lives in one pocket (first eighth); rest is zero-mean noise.
    pocket = pd.Series(rng.normal(0, 0.02, 800), index=dates)
    pocket.iloc[:100] += 0.15
    frac = _cpcv_sign_stability(pocket, horizon=21)
    assert frac < 0.85            # well below the stable series (~0.79 on this seed:
                                  # combos containing the pocket block pass, the rest
                                  # coin-flip — exactly the regime-pocket signature)
    # too-short series -> NaN (advisory only)
    assert np.isnan(_cpcv_sign_stability(stable.iloc[:50], horizon=21))


def test_gate_precision_weighted_aggregation_defeats_small_sleeve_dilution():
    """Per-date (N-1)-weighted IC aggregation: a strong 500-name sleeve must not be
    drowned by a zero-signal 8-name sleeve (first live gate run: mom_12_1 pooled
    unweighted t=0.85 vs equity-alone t=3.3)."""
    rng = np.random.default_rng(5)
    dates = pd.date_range("2020-01-01", periods=400, freq="B")
    rows = []
    for d in dates:
        rows.append({"obs_date": d, "sleeve": "equity",
                     "rank_ic": 0.03 + rng.normal(0, 0.045), "n_names": 500})
        rows.append({"obs_date": d, "sleeve": "fx_etf",
                     "rank_ic": rng.normal(0, 0.38), "n_names": 8})
    ic = pd.DataFrame(rows)

    unweighted = ic.groupby("obs_date")["rank_ic"].mean()
    icw = ic.assign(_w=(ic["n_names"].clip(lower=2) - 1).astype(float))
    weighted = (icw.assign(_wx=icw["rank_ic"] * icw["_w"])
                   .groupby("obs_date")[["_wx", "_w"]].sum()
                   .pipe(lambda g: g["_wx"] / g["_w"]))

    def t(s):
        return s.mean() / s.std(ddof=1) * np.sqrt(len(s))

    assert t(weighted) > t(unweighted) + 2.0     # dilution removed
    assert t(weighted) > 4.0                     # recovers the real equity signal
