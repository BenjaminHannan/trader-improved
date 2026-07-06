"""Risk-model tests: PSD guarantees, PIT corruption harnesses, known-structure
recovery, diversification, shrinkage arithmetic, and French validation.

All data comes from the synthetic ``price_panel`` fixture (or locally simulated
panels in the same curated-long schema). No network, no lake I/O.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from production.risk.covariance import ewma_cov
from production.risk.exposures import build_exposures
from production.risk.factor_returns import estimate_factor_returns
from production.risk.model import RiskModel, validate_against_french
from production.risk.specific import specific_vol

AS_OF = pd.Timestamp("2021-12-31")
UTC = "UTC"


def _ids(sleeve_of, sleeve):
    return [i for i, s in sleeve_of.items() if s == sleeve]


def _mandatory(obs_dates, hours=21.5, source="synthetic:test"):
    avail = pd.to_datetime(obs_dates).dt.tz_localize(UTC) + pd.Timedelta(hours=hours)
    return {"available_from": avail, "source": source,
            "ingested_at": avail + pd.Timedelta(minutes=5)}


def _panel_from_close(close_wide: pd.DataFrame, volume=1e6) -> pd.DataFrame:
    """Build a curated-long price panel from a wide close matrix."""
    frames = []
    for iid in close_wide.columns:
        s = close_wide[iid].dropna()
        frames.append(pd.DataFrame({
            "obs_date": s.index, "instrument_id": iid, "close": s.to_numpy(),
            "volume": volume, "dollar_volume": s.to_numpy() * volume,
        }))
    out = pd.concat(frames, ignore_index=True)
    out = out.assign(**_mandatory(out["obs_date"]))
    return out.sort_values(["obs_date", "instrument_id"]).reset_index(drop=True)


# --------------------------------------------------------------------- PSD
def test_structural_covariance_psd(price_panel, sleeve_of):
    ids = _ids(sleeve_of, "equity")
    model = RiskModel.build(price_panel, "equity", AS_OF, ids)
    assert model.factor_form() is not None  # enough history -> structural form
    cov = model.covariance().to_numpy()
    eig = np.linalg.eigvalsh(0.5 * (cov + cov.T))
    assert eig.min() >= -1e-8


def test_small_sleeve_covariance_psd_and_none_factor_form(price_panel, sleeve_of):
    ids = _ids(sleeve_of, "fx_etf")
    model = RiskModel.build(price_panel, "fx_etf", AS_OF, ids)
    assert model.factor_form() is None
    assert model._cov is not None
    cov = model.covariance().to_numpy()
    eig = np.linalg.eigvalsh(0.5 * (cov + cov.T))
    assert eig.min() >= -1e-8
    # resid_vol is sqrt of the covariance diagonal for small sleeves.
    np.testing.assert_allclose(model.resid_vol.to_numpy(),
                               np.sqrt(np.diag(model.covariance().to_numpy())),
                               rtol=1e-9)


# ------------------------------------------------------- exposure PIT corruption
def test_exposures_pit_corruption(price_panel, sleeve_of):
    ids = _ids(sleeve_of, "equity")
    base = build_exposures(price_panel, "equity", AS_OF, ids)

    # Corrupt every row with obs_date > as_of (NaN the close, then drop them).
    corrupt = price_panel.copy()
    future = pd.to_datetime(corrupt["obs_date"]) > AS_OF
    corrupt.loc[future, ["close", "volume", "dollar_volume"]] = np.nan
    corrupt = corrupt[~future].reset_index(drop=True)

    after = build_exposures(corrupt, "equity", AS_OF, ids)
    pd.testing.assert_frame_equal(base, after)


# --------------------------------------------------- factor-return refit PIT
def test_factor_returns_refit_pit(price_panel, sleeve_of):
    ids = _ids(sleeve_of, "equity")
    sp = price_panel[price_panel["instrument_id"].isin(ids)].reset_index(drop=True)
    start, end = pd.Timestamp("2019-01-01"), pd.Timestamp("2021-12-31")

    fr_base, _ = estimate_factor_returns(sp, "equity", start, end, None)

    cut = pd.Timestamp("2020-06-15")  # corrupt everything strictly after this
    corrupt = sp.copy()
    fut = pd.to_datetime(corrupt["obs_date"]) > cut
    corrupt.loc[fut, ["close", "volume", "dollar_volume"]] *= 3.0
    fr_corr, _ = estimate_factor_returns(corrupt, "equity", start, end, None)

    # Factor returns for dates on/before the last refit that used only pre-cut
    # data must be unchanged. Compare strictly before the month of the cut.
    safe = fr_base.index < pd.Timestamp("2020-06-01")
    assert safe.sum() > 100
    pd.testing.assert_frame_equal(fr_base.loc[safe], fr_corr.loc[safe])


# ---------------------------------------------------- known-structure recovery
def _simulate_market_model(n_names=12, n_days=520, seed=0):
    """r_i = beta_i * f + eps_i with a known beta spread and factor vol."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-02", periods=n_days)
    betas = np.linspace(0.5, 1.8, n_names)
    f_vol, eps_vol = 0.010, 0.006
    f = rng.normal(0.0002, f_vol, n_days)
    ids = [f"EQ:SIM{i:02d}:2000-01-03" for i in range(n_names)]
    closes = {}
    for i, iid in enumerate(ids):
        eps = rng.normal(0, eps_vol, n_days)
        rets = betas[i] * f + eps
        closes[iid] = 100.0 * np.exp(np.cumsum(rets))
    close_wide = pd.DataFrame(closes, index=dates)
    truth = {"betas": pd.Series(betas, index=ids), "f_vol": f_vol,
             "eps_vol": eps_vol}
    return close_wide, ids, truth


def test_known_structure_recovery():
    close_wide, ids, truth = _simulate_market_model()
    prices = _panel_from_close(close_wide)
    as_of = close_wide.index[-1]

    B = build_exposures(prices, "equity", as_of, ids)
    corr = np.corrcoef(B["market"].reindex(ids).to_numpy(),
                       truth["betas"].to_numpy())[0, 1]
    assert corr > 0.9

    model = RiskModel.build(prices, "equity", as_of, ids)
    w = pd.Series(1.0 / len(ids), index=ids)
    pv = model.portfolio_vol(w)

    # Truth: equal-weight portfolio return has factor part (wbar*beta)*f plus
    # diversified specific risk; annualize the daily vol.
    wbar_beta = truth["betas"].mean()
    true_daily = np.sqrt((wbar_beta * truth["f_vol"]) ** 2 +
                         (truth["eps_vol"] ** 2) / len(ids))
    true_ann = true_daily * np.sqrt(252)
    assert abs(pv - true_ann) / true_ann < 0.25


# ------------------------------------------------------- diversification sanity
def test_diversification(price_panel, sleeve_of):
    ids = _ids(sleeve_of, "equity")
    model = RiskModel.build(price_panel, "equity", AS_OF, ids)
    cov = model.covariance()
    w = pd.Series(1.0 / len(ids), index=ids)
    pv = model.portfolio_vol(w)
    single = np.sqrt(252 * np.diag(cov.to_numpy()))
    weighted_avg = float((w.reindex(cov.index).to_numpy() * single).sum())
    assert pv < weighted_avg


# ----------------------------------------------------- shrinkage arithmetic
def test_specific_vol_shrinkage_exact():
    rng = np.random.default_rng(3)
    dates = pd.bdate_range("2020-01-01", periods=120)
    res = pd.DataFrame(rng.normal(0, 0.01, (120, 4)),
                       index=dates, columns=["a", "b", "c", "d"])
    v0 = specific_vol(res, shrink_weight=0.0)
    v = specific_vol(res, shrink_weight=0.25)
    expected = 0.75 * v0 + 0.25 * v0.median()
    pd.testing.assert_series_equal(v, expected)


def test_ewma_cov_shrinkage_exact():
    rng = np.random.default_rng(5)
    dates = pd.bdate_range("2020-01-01", periods=200)
    r = pd.DataFrame(rng.normal(0, 0.01, (200, 3)),
                     index=dates, columns=["x", "y", "z"])
    c0 = ewma_cov(r, halflife=30, shrink=0.0)
    cs = ewma_cov(r, halflife=30, shrink=0.4)
    # (1-s)*C + s*diag(C): jitter cancels because it is in both.
    diag0 = pd.DataFrame(np.diag(np.diag(c0.to_numpy())),
                         index=c0.index, columns=c0.columns)
    expected = 0.6 * c0 + 0.4 * diag0
    np.testing.assert_allclose(cs.to_numpy(), expected.to_numpy(), atol=1e-15)

    c1 = ewma_cov(r, halflife=30, shrink=1.0)  # pure diagonal (up to jitter)
    off = c1.to_numpy() - np.diag(np.diag(c1.to_numpy()))
    assert np.allclose(off, 0.0, atol=1e-14)


def test_ewma_cov_min_obs_raises():
    dates = pd.bdate_range("2020-01-01", periods=10)
    r = pd.DataFrame(np.random.default_rng(0).normal(0, 0.01, (10, 2)),
                     index=dates, columns=["a", "b"])
    with pytest.raises(Exception):
        ewma_cov(r, halflife=30, min_obs=60)


# --------------------------------------------------------- French validation
def test_validate_against_french_synthetic():
    rng = np.random.default_rng(9)
    dates = pd.bdate_range("2016-01-01", periods=750)
    mkt = rng.normal(0.0004, 0.01, len(dates))
    mom = rng.normal(0.0002, 0.008, len(dates))
    fr = pd.DataFrame({"market": mkt, "momentum": mom}, index=dates)

    # Benchmark = monthly-compounded estimated factor + small noise.
    monthly = (1.0 + fr).groupby(fr.index.to_period("M")).prod() - 1.0
    nz = rng.normal(0, 1e-4, monthly.shape)
    french = pd.DataFrame({
        "Mkt-RF": monthly["market"].to_numpy() + nz[:, 0],
        "Mom": monthly["momentum"].to_numpy() + nz[:, 1],
    }, index=monthly.index.to_timestamp())

    out = validate_against_french(fr, french)
    assert out["market"] > 0.9
    assert out["momentum"] > 0.9
