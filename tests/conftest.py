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


def make_basis(crypto_ids: list[str], start="2018-01-02", end="2021-12-31",
               seed=23) -> pd.DataFrame:
    """Daily perp-vs-spot basis panel (ccxt perp/spot close ratio - 1).

    Mildly positive mean (contango is the normal state), AR(1)-ish persistence so a
    trailing-mean carry signal has something to chew on. available_from = bar close
    (obs midnight UTC + 24h), same rule as ccxt prices.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq="D")
    frames = []
    for iid in sorted(crypto_ids):
        b = np.empty(len(dates))
        b[0] = 5e-4
        eps = rng.normal(0, 4e-4, len(dates))
        for i in range(1, len(dates)):
            b[i] = 0.9 * b[i - 1] + 1e-4 + eps[i]
        frames.append(pd.DataFrame({
            "obs_date": dates, "instrument_id": iid, "basis": b}))
    out = pd.concat(frames, ignore_index=True)
    out = out.assign(**_mandatory(out["obs_date"], pd.Timedelta(hours=24),
                                  "synthetic:basis"))
    return out.reset_index(drop=True)


def make_fundamentals(equity_ids: list[str], start="2018-01-02", end="2021-12-31",
                      seed=29) -> pd.DataFrame:
    """Quarterly EPS/revenue panel in EDGAR-shaped curated form.

    obs_date = fiscal period end; available_from = the FILING timestamp (~40 days
    later, jittered) — the whole point of the EDGAR pipeline is that fundamentals
    are knowable at filing, never at period end. One row per (period, instrument,
    field) with field in {eps, revenue, shares}.
    """
    rng = np.random.default_rng(seed)
    quarters = pd.date_range(start, end, freq="QE")
    rows = []
    for i, iid in enumerate(sorted(equity_ids)):
        eps = 1.0 + 0.2 * (i % 5) + np.cumsum(rng.normal(0.02, 0.15, len(quarters)))
        rev = (50 + 10 * i) * np.exp(np.cumsum(rng.normal(0.01, 0.05, len(quarters))))
        shares = np.full(len(quarters), 1e8 * (1 + i * 0.3))
        for q, (d, e, r, s) in enumerate(zip(quarters, eps, rev, shares)):
            filing = (pd.Timestamp(d).tz_localize(UTC)
                      + pd.Timedelta(days=int(35 + rng.integers(0, 20)), hours=21))
            for field, val in (("eps", e), ("revenue", r), ("shares", s)):
                rows.append((d, iid, field, val, filing, "synthetic:fundamentals",
                             filing + pd.Timedelta(minutes=5)))
    return pd.DataFrame(rows, columns=[
        "obs_date", "instrument_id", "field", "value",
        "available_from", "source", "ingested_at"])


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
    rt = {f"RT:{s}:2002-07-26": "rates_etf" for s in ["TLT", "IEF", "LQD"]}
    ie = {f"IE:{s}:1996-03-12": "intl_etf" for s in ["EWJ", "EWG", "EWU"]}
    se = {f"SE:{s}:1998-12-16": "sector_etf" for s in ["XLK", "XLF", "XLE"]}
    return {**eq, **cr, **fx, **co, **rt, **ie, **se}


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
        "BAMLH0A0HYM2": 4.0, "VIXCLS": 18.0, "VXVCLS": 20.0,
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
def basis_panel(synthetic_instruments) -> pd.DataFrame:
    ids = [i for i, s in synthetic_instruments.items() if s == "crypto"]
    return make_basis(ids)


@pytest.fixture(scope="session")
def fundamentals_panel(synthetic_instruments) -> pd.DataFrame:
    ids = [i for i, s in synthetic_instruments.items() if s == "equity"]
    return make_fundamentals(ids)


@pytest.fixture(scope="session")
def signal_data(price_panel, funding_panel, macro_panel, cot_panel,
                mcap_tvl_panels, basis_panel, fundamentals_panel) -> dict[str, pd.DataFrame]:
    """The full signal input bundle."""
    mcap, tvl = mcap_tvl_panels
    return {"prices": price_panel, "funding": funding_panel,
            "macro": macro_panel, "cot": cot_panel,
            "mcap": mcap, "tvl": tvl, "basis": basis_panel,
            "fundamentals": fundamentals_panel}


@pytest.fixture()
def tmp_lake(tmp_path) -> Lake:
    return Lake(tmp_path / "data")


@pytest.fixture(scope="session")
def sleeve_of(synthetic_instruments):
    """instrument_id -> sleeve lookup as a pd.Series."""
    return pd.Series(synthetic_instruments)


@pytest.fixture()
def pregate_engine_registry(tmp_path, monkeypatch):
    """Pin the engine's factor selection to an all-candidate, no-gate-records registry.

    Same rationale as test_backtest_engine._pregate_registry: tests that exercise
    mechanics against synthetic bundles were authored pre-gate; the live registry's
    accepted set (real verdicts since 2026-07-07) selects factors the synthetic
    bundles cannot feed, so no sleeve produces weights. Hermetic input, identical
    mechanics. Function-scoped so any test file can opt in.
    """
    import yaml

    import production.backtest.engine as _eng
    from production.alpha.registry import FactorRegistry
    from production.core.config import CONFIG_DIR

    cfg = yaml.safe_load(open(CONFIG_DIR / "factors.yaml"))
    for spec in cfg["factors"].values():
        spec["status"] = "candidate"
        spec["gate_stats"] = {}
    path = tmp_path / "factors.yaml"
    yaml.safe_dump(cfg, open(path, "w"), sort_keys=False)

    class _PinnedRegistry(FactorRegistry):  # a real class: engine isinstance()s it
        def __init__(self, cfg_path=None):
            super().__init__(cfg_path if cfg_path is not None else path)

    monkeypatch.setattr(_eng, "FactorRegistry", _PinnedRegistry)
    return path
