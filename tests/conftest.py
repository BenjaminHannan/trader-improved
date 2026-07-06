"""Shared synthetic fixtures — GBM toy data in exact curated-lake schema.

These fixtures ARE the contract for every test in the suite. Column layouts here
mirror the curated zone exactly (mandatory columns included), so tests exercise
the same schemas production code reads.

Signal input bundle contract (used across signal/alpha/backtest tests):
    data: dict[str, pd.DataFrame] with keys among
      "prices":  [obs_date, instrument_id, close, volume, dollar_volume, + mandatory]
      "funding": [obs_date, instrument_id, funding_rate, + mandatory]
      "macro":   [obs_date, series_id, value, + mandatory]
      "cot":     [obs_date, instrument_id, noncomm_net, open_interest, + mandatory]
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from production.core.lake import Lake

UTC = "UTC"


def _mandatory(obs_dates: pd.Series, avail_offset: pd.Timedelta,
               source: str) -> dict:
    avail = pd.to_datetime(obs_dates).dt.tz_localize(UTC) + avail_offset
    return {
        "available_from": avail,
        "source": source,
        "ingested_at": avail + pd.Timedelta(minutes=5),
    }


# --------------------------------------------------------------------- makers
def make_gbm_prices(instruments: dict[str, str], start="2018-01-02",
                    end="2021-12-31", seed=7) -> pd.DataFrame:
    """Long curated-format daily close/volume panel via GBM.

    instruments: {instrument_id: sleeve}. Crypto ids get a 7-day calendar,
    everything else NYSE-ish business days. available_from = obs 21:30 UTC.
    """
    rng = np.random.default_rng(seed)
    bdays = pd.bdate_range(start, end)
    alldays = pd.date_range(start, end, freq="D")
    frames = []
    for i, (iid, sleeve) in enumerate(sorted(instruments.items())):
        dates = alldays if sleeve == "crypto" else bdays
        n = len(dates)
        mu, sig = 0.0003, 0.012 + 0.002 * (i % 5)
        if sleeve == "crypto":
            mu, sig = 0.0008, 0.035
        rets = rng.normal(mu, sig, n)
        close = 100.0 * (1 + i * 0.13) * np.exp(np.cumsum(rets))
        volume = rng.lognormal(13 + (i % 7) * 0.4, 0.3, n)
        df = pd.DataFrame({
            "obs_date": dates, "instrument_id": iid,
            "close": close, "volume": volume,
            "dollar_volume": close * volume,
        })
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out = out.assign(**_mandatory(out["obs_date"], pd.Timedelta(hours=21, minutes=30),
                                  "synthetic:prices"))
    return out.sort_values(["obs_date", "instrument_id"]).reset_index(drop=True)


def make_funding(crypto_ids: list[str], start="2018-01-02", end="2021-12-31",
                 seed=11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq="D")
    frames = []
    for iid in sorted(crypto_ids):
        df = pd.DataFrame({
            "obs_date": dates, "instrument_id": iid,
            "funding_rate": rng.normal(1e-4, 3e-4, len(dates)),
        })
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out = out.assign(**_mandatory(out["obs_date"], pd.Timedelta(hours=24),
                                  "synthetic:funding"))
    return out.reset_index(drop=True)


def make_macro(series: dict[str, float], start="2018-01-02", end="2021-12-31",
               seed=13) -> pd.DataFrame:
    """series: {series_id: base_level}. Daily random-walk levels, published next day."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, end)
    frames = []
    for sid, base in sorted(series.items()):
        level = base + np.cumsum(rng.normal(0, abs(base) * 0.01 + 0.01, len(dates)))
        frames.append(pd.DataFrame({"obs_date": dates, "series_id": sid, "value": level}))
    out = pd.concat(frames, ignore_index=True)
    out = out.assign(**_mandatory(out["obs_date"], pd.Timedelta(hours=32),
                                  "synthetic:macro"))
    return out.reset_index(drop=True)


def make_mcap_tvl(crypto_ids: list[str], start="2018-01-02", end="2021-12-31",
                  seed=19) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Daily mcap and TVL snapshot panels (CoinGecko/DefiLlama shape).

    Snapshots are stamped at ingest time (next day 06:00 UTC) — no rewritten
    history is trusted, mirroring the real loaders. BTC gets no TVL row
    (not a smart-contract chain), mirroring reality.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq="D")
    mcap_frames, tvl_frames = [], []
    for i, iid in enumerate(sorted(crypto_ids)):
        mcap = np.exp(np.cumsum(rng.normal(0.0005, 0.03, len(dates)))) * 1e9 * (i + 1)
        mcap_frames.append(pd.DataFrame({
            "obs_date": dates, "instrument_id": iid, "mcap": mcap}))
        if ":BTC:" not in iid:
            tvl = mcap * np.clip(rng.normal(0.15, 0.05, len(dates)), 0.01, None)
            tvl_frames.append(pd.DataFrame({
                "obs_date": dates, "instrument_id": iid, "tvl": tvl}))
    offset = pd.Timedelta(hours=30)  # snapshot pulled next morning
    mcap_df = pd.concat(mcap_frames, ignore_index=True)
    mcap_df = mcap_df.assign(**_mandatory(mcap_df["obs_date"], offset, "synthetic:mcap"))
    tvl_df = pd.concat(tvl_frames, ignore_index=True)
    tvl_df = tvl_df.assign(**_mandatory(tvl_df["obs_date"], offset, "synthetic:tvl"))
    return mcap_df.reset_index(drop=True), tvl_df.reset_index(drop=True)


def make_cot(etf_ids: list[str], start="2018-01-02", end="2021-12-31",
             seed=17) -> pd.DataFrame:
    """Tuesday obs_date, Friday 20:30 UTC availability — the canonical lag."""
    rng = np.random.default_rng(seed)
    tuesdays = pd.date_range(start, end, freq="W-TUE")
    frames = []
    for iid in sorted(etf_ids):
        oi = rng.lognormal(11, 0.2, len(tuesdays))
        frames.append(pd.DataFrame({
            "obs_date": tuesdays, "instrument_id": iid,
            "noncomm_net": rng.normal(0, 0.15, len(tuesdays)) * oi,
            "open_interest": oi,
        }))
    out = pd.concat(frames, ignore_index=True)
    out = out.assign(**_mandatory(out["obs_date"],
                                  pd.Timedelta(days=3, hours=20, minutes=30),
                                  "synthetic:cot"))
    return out.reset_index(drop=True)


# ------------------------------------------------------------------- fixtures
@pytest.fixture(scope="session")
def synthetic_instruments() -> dict[str, str]:
    eq = {f"EQ:SYN{i:02d}:2000-01-03": "equity" for i in range(10)}
    cr = {f"CR:{s}:2017-01-01": "crypto" for s in ["BTC", "ETH", "SOL", "LTC", "ADA"]}
    fx = {f"FX:{s}:2007-01-03": "fx_etf" for s in ["FXE", "FXY", "FXB"]}
    co = {f"CO:{s}:2006-01-03": "commodity_etf" for s in ["GLD", "USO", "SLV"]}
    return {**eq, **cr, **fx, **co}


@pytest.fixture(scope="session")
def price_panel(synthetic_instruments) -> pd.DataFrame:
    return make_gbm_prices(synthetic_instruments)


@pytest.fixture(scope="session")
def funding_panel(synthetic_instruments) -> pd.DataFrame:
    ids = [i for i, s in synthetic_instruments.items() if s == "crypto"]
    return make_funding(ids)


@pytest.fixture(scope="session")
def macro_panel() -> pd.DataFrame:
    return make_macro({
        "DGS3MO_US": 2.0, "RATE_EU": 0.5, "RATE_JP": -0.1, "RATE_GB": 1.0,
        "BAMLH0A0HYM2": 4.0, "VIXCLS": 18.0,
    })


@pytest.fixture(scope="session")
def cot_panel(synthetic_instruments) -> pd.DataFrame:
    ids = [i for i, s in synthetic_instruments.items()
           if s in ("fx_etf", "commodity_etf")]
    return make_cot(ids)


@pytest.fixture(scope="session")
def mcap_tvl_panels(synthetic_instruments) -> tuple[pd.DataFrame, pd.DataFrame]:
    ids = [i for i, s in synthetic_instruments.items() if s == "crypto"]
    return make_mcap_tvl(ids)


@pytest.fixture(scope="session")
def signal_data(price_panel, funding_panel, macro_panel, cot_panel,
                mcap_tvl_panels) -> dict[str, pd.DataFrame]:
    """The full signal input bundle."""
    mcap, tvl = mcap_tvl_panels
    return {"prices": price_panel, "funding": funding_panel,
            "macro": macro_panel, "cot": cot_panel,
            "mcap": mcap, "tvl": tvl}


@pytest.fixture()
def tmp_lake(tmp_path) -> Lake:
    return Lake(tmp_path / "data")


@pytest.fixture(scope="session")
def sleeve_of(synthetic_instruments):
    """instrument_id -> sleeve lookup as a pd.Series."""
    return pd.Series(synthetic_instruments)
