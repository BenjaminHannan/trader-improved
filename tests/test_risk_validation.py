"""Risk-model scoring harness tests (research/wiki/questions/research-risk-model-validation.md
Q1) -- fully synthetic (no lake, no network), tuned to run in well under 60s total.

Six things pinned here: (1) a well-calibrated forecast lands every family-1/2/3/4
cell inside the acceptance band; (2) a mis-scaled forecast is correctly flagged
under/over; (3) the MVP horse race + LW-2011 bootstrap picks out a genuinely
better risk model with p<0.05; (4) adoption_verdict implements the 4 pass
criteria; (5) the z-window bookkeeping is provably non-overlapping; (6) Sigma_t
never sees data at or after the evaluation date, even across a volatility break.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from production.risk.validation import (RiskValidationError, adoption_verdict,
                                        baseline_health_check, bias_stats,
                                        build_test_portfolios, evaluation_dates,
                                        mvp_horse_race)

_H = 21
_MIN_OBS = 300


# ------------------------------------------------------------------- DGP makers
def _simulate_iid_mvn(n_assets=8, n_days=2600, seed=0, base_vol=0.012):
    """iid multivariate normal daily returns with a KNOWN, non-trivial Sigma."""
    rng = np.random.default_rng(seed)
    vols = rng.uniform(base_vol * 0.7, base_vol * 1.4, n_assets)
    L = rng.normal(0, 0.4, (n_assets, n_assets))
    corr = L @ L.T
    d = np.sqrt(np.diag(corr))
    corr = corr / np.outer(d, d)
    np.fill_diagonal(corr, 1.0)
    sigma = np.outer(vols, vols) * corr
    cols = [f"n{i}" for i in range(n_assets)]
    dates = pd.bdate_range("2010-01-04", periods=n_days)
    R = rng.multivariate_normal(np.zeros(n_assets), sigma, size=n_days)
    returns_panel = pd.DataFrame(R, index=dates, columns=cols)
    sigma_df = pd.DataFrame(sigma, index=cols, columns=cols)
    return sigma_df, returns_panel


def _const_sigma_fn(sigma_df: pd.DataFrame):
    def fn(_window: pd.DataFrame) -> pd.DataFrame:
        return sigma_df
    return fn


def _simulate_one_factor(n_assets=8, n_days=1200, seed=44, f_vol=0.020, eps_vol=0.006):
    """r_i = beta_i * f + eps_i -- a strong common-factor DGP where ignoring
    correlation (diagonal-only Sigma) leaves real, hedge-able factor risk on
    the table for the MVP horse race."""
    rng = np.random.default_rng(seed)
    betas = rng.uniform(0.5, 1.8, n_assets)
    dates = pd.bdate_range("2015-01-02", periods=n_days)
    f = rng.normal(0.0, f_vol, n_days)
    eps = rng.normal(0.0, eps_vol, (n_days, n_assets))
    R = np.outer(f, betas) + eps
    cols = [f"n{i}" for i in range(n_assets)]
    returns_panel = pd.DataFrame(R, index=dates, columns=cols)
    sigma_true = f_vol ** 2 * np.outer(betas, betas) + np.diag(np.full(n_assets, eps_vol ** 2))
    true_df = pd.DataFrame(sigma_true, index=cols, columns=cols)
    diag_df = pd.DataFrame(np.diag(np.diag(sigma_true)), index=cols, columns=cols)
    return returns_panel, true_df, diag_df


# ---------------------------------------------------------- 1. calibrated DGP
def test_calibrated_dgp_every_family_in_band():
    # Seeds fixed after verifying (dgp_seed, rng_seed) empirically clears the
    # [0.87, 1.13] band for all four families -- with T ~ 120 samples per cell,
    # roughly 1-in-20 (dgp, rng) pairs land a well-calibrated cell just outside
    # the nominal 95% band by chance alone; this pair does not.
    sigma_df, returns_panel = _simulate_iid_mvn(n_assets=8, n_days=2600, seed=0)
    sigma_fn = _const_sigma_fn(sigma_df)
    rng = np.random.default_rng(0)

    for family in (1, 2, 3, 4):
        if family == 4:
            weights = lambda Sig, _r=returns_panel, _rng=rng: \
                build_test_portfolios(_r, "test", 4, _rng, sigma=Sig)
        else:
            weights = build_test_portfolios(returns_panel, "test", family, rng, n=100)
        cell = bias_stats(returns_panel, sigma_fn, weights, h=_H, min_obs=_MIN_OBS)
        assert cell["T"] >= 100, f"family {family}: T={cell['T']} < 100"
        assert 0.87 <= cell["B"] <= 1.13, f"family {family}: B={cell['B']:.4f} outside band"
        assert cell["in_band"]

    cells = {("test", f): bias_stats(
        returns_panel, sigma_fn,
        (lambda Sig, _r=returns_panel, _rng=rng: build_test_portfolios(_r, "test", 4, _rng, sigma=Sig))
        if f == 4 else build_test_portfolios(returns_panel, "test", f, rng, n=100),
        h=_H, min_obs=_MIN_OBS) for f in (1, 2)}
    health = baseline_health_check(cells)
    assert health["passed"]
    assert health["fraction_in_band"] >= 0.90


# ------------------------------------------------------------ 2. miscalibrated
def test_miscalibrated_dgp_flags_under_and_over_forecast():
    sigma_df, returns_panel = _simulate_iid_mvn(n_assets=8, n_days=2600, seed=3)
    rng = np.random.default_rng(4)
    weights = build_test_portfolios(returns_panel, "test", 1, rng, n=100)

    under_fn = _const_sigma_fn(sigma_df / 4.0)     # sigma_hat halved -> z doubles
    cell_under = bias_stats(returns_panel, under_fn, weights, h=_H, min_obs=_MIN_OBS)
    assert cell_under["B"] > 1.13
    assert not cell_under["in_band"]
    assert abs(cell_under["B"] - 2.0) < 0.35

    over_fn = _const_sigma_fn(sigma_df * 4.0)      # sigma_hat doubled -> z halves
    cell_over = bias_stats(returns_panel, over_fn, weights, h=_H, min_obs=_MIN_OBS)
    assert cell_over["B"] < 0.87
    assert not cell_over["in_band"]
    assert abs(cell_over["B"] - 0.5) < 0.2


# --------------------------------------------------------------- 3. MVP horse race
def test_mvp_horse_race_true_sigma_beats_diagonal():
    returns_panel, true_sigma, diag_sigma = _simulate_one_factor()
    result = mvp_horse_race(_const_sigma_fn(true_sigma), _const_sigma_fn(diag_sigma),
                            returns_panel, step=5, min_obs=100, n_boot=500, seed=7)
    assert result["candidate_lower"]
    assert result["candidate_ann_vol"] < result["incumbent_ann_vol"]
    assert result["p_value"] < 0.05


# ----------------------------------------------------------------- 4. adoption
def test_adoption_verdict_criteria():
    hr_good = {"candidate_lower": True, "p_value": 0.01}
    hr_bad = {"candidate_lower": False, "p_value": 0.5}

    # criterion 1: a cell that was in-band under the incumbent is NOT under the candidate.
    inc1 = {("eq", 1): {"B": 1.0, "in_band": True}, ("eq", 2): {"B": 1.0, "in_band": True}}
    cand1 = {("eq", 1): {"B": 1.5, "in_band": False}, ("eq", 2): {"B": 1.0, "in_band": True}}
    v1 = adoption_verdict(inc1, cand1, hr_good, pit_harness_green=True)
    assert not v1["passed"]
    assert not v1["criteria"]["no_new_failures"]
    assert ("eq", 1) in v1["new_failures"]

    # criterion 2: fewer than 60% of cells improve |B-1|.
    inc2 = {("eq", i): {"B": 1.05, "in_band": True} for i in range(1, 6)}
    cand2 = {("eq", 1): {"B": 1.20, "in_band": True}, ("eq", 2): {"B": 1.20, "in_band": True},
            ("eq", 3): {"B": 1.00, "in_band": True}, ("eq", 4): {"B": 1.30, "in_band": True},
            ("eq", 5): {"B": 1.40, "in_band": True}}
    v2 = adoption_verdict(inc2, cand2, hr_good, pit_harness_green=True)
    assert not v2["passed"]
    assert not v2["criteria"]["improved_ge_60pct"]

    # criterion 3: inconclusive horse race fails unless candidate is claimed simpler.
    inc3 = {("eq", 1): {"B": 1.0, "in_band": True}}
    cand3 = {("eq", 1): {"B": 1.0, "in_band": True}}
    v3 = adoption_verdict(inc3, cand3, hr_bad, pit_harness_green=True)
    assert not v3["passed"]
    assert not v3["criteria"]["mvp_horse_race"]
    v3_simpler = adoption_verdict(inc3, cand3, hr_bad, pit_harness_green=True,
                                  candidate_is_simpler=True)
    assert v3_simpler["criteria"]["mvp_horse_race"]
    assert v3_simpler["passed"]

    # criterion 4: PIT harness not green fails regardless of everything else.
    v4 = adoption_verdict(inc3, cand3, hr_good, pit_harness_green=False)
    assert not v4["passed"]
    assert not v4["criteria"]["pit_harness_green"]

    # all four pass.
    inc5 = {("eq", i): {"B": 1.10, "in_band": True} for i in range(1, 4)}
    cand5 = {("eq", i): {"B": 1.02, "in_band": True} for i in range(1, 4)}
    v5 = adoption_verdict(inc5, cand5, hr_good, pit_harness_green=True)
    assert v5["passed"]
    assert v5["criteria"] == {"no_new_failures": True, "frac_improved": 1.0,
                              "improved_ge_60pct": True, "mvp_horse_race": True,
                              "pit_harness_green": True}


# ----------------------------------------------------------------- 5. non-overlap
def test_evaluation_dates_and_z_count_are_non_overlapping():
    dates = pd.bdate_range("2015-01-01", periods=2600)
    ev = evaluation_dates(dates, h=_H, min_obs=_MIN_OBS)
    positions = np.array([dates.get_loc(d) for d in ev])

    assert len(ev) >= 100
    assert np.all(np.diff(positions) == _H)               # exactly h apart, never overlapping
    expected = len(np.arange(_MIN_OBS, len(dates) - _H + 1, _H))
    assert len(ev) == expected

    # A single-portfolio-per-window family (3: equal weight) must contribute
    # exactly one z per evaluation date -- T == number of non-overlapping windows.
    n_assets = 5
    cols = [f"n{i}" for i in range(n_assets)]
    R = np.random.default_rng(9).normal(0, 0.01, (len(dates), n_assets))
    returns_panel = pd.DataFrame(R, index=dates, columns=cols)
    sigma_fn = _const_sigma_fn(pd.DataFrame(np.eye(n_assets) * 1e-4, index=cols, columns=cols))
    weights = build_test_portfolios(returns_panel, "test", 3, np.random.default_rng(0))
    cell = bias_stats(returns_panel, sigma_fn, weights, h=_H, min_obs=_MIN_OBS)
    assert cell["T"] == len(ev)


# ------------------------------------------------------------------- 6. PIT guard
def test_pit_guard_sigma_never_sees_data_at_or_after_t():
    n_assets, n_days, break_pos = 6, 1200, 800
    dates = pd.bdate_range("2016-01-01", periods=n_days)
    rng = np.random.default_rng(55)
    vol = np.where(np.arange(n_days) < break_pos, 0.01, 0.02)   # variance doubles at break_pos
    R = rng.normal(0.0, 1.0, (n_days, n_assets)) * vol[:, None]
    cols = [f"n{i}" for i in range(n_assets)]
    returns_panel = pd.DataFrame(R, index=dates, columns=cols)
    break_date = dates[break_pos]

    seen_windows: dict = {}

    def spy_sigma_fn(window: pd.DataFrame) -> pd.DataFrame:
        # Recover the evaluation date t from the full panel index: bias_stats
        # sliced returns_panel.iloc[:pos], so t is the row immediately after
        # this window's last (visible) date.
        last = window.index.max()
        t = returns_panel.index[returns_panel.index.get_loc(last) + 1]
        seen_windows[t] = window.index
        cov = window.tail(60).cov().to_numpy()
        return pd.DataFrame(cov, index=window.columns, columns=window.columns)

    weights = build_test_portfolios(returns_panel, "test", 3, np.random.default_rng(0))
    bias_stats(returns_panel, spy_sigma_fn, weights, h=_H, min_obs=300)

    assert len(seen_windows) >= 10
    for eval_date, window_index in seen_windows.items():
        assert window_index.max() < eval_date            # strict shift(1)-equivalent contract
        if eval_date <= break_date:
            assert window_index.max() < break_date        # pre-break forecasts see NO post-break rows
            assert bool((window_index < break_date).all())


# ------------------------------------------------------------- portfolio families
def test_build_test_portfolios_family_dispatch():
    n_assets = 6
    cols = [f"n{i}" for i in range(n_assets)]
    dates = pd.bdate_range("2020-01-01", periods=50)
    returns_panel = pd.DataFrame(np.random.default_rng(0).normal(0, 0.01, (50, n_assets)),
                                 index=dates, columns=cols)
    rng = np.random.default_rng(1)

    f1 = build_test_portfolios(returns_panel, "test", 1, rng, n=20)
    assert f1.shape == (20, n_assets)
    assert np.all(f1.to_numpy() >= 0.0)
    np.testing.assert_allclose(f1.sum(axis=1).to_numpy(), 1.0, atol=1e-10)

    f2 = build_test_portfolios(returns_panel, "test", 2, rng, n=20)
    np.testing.assert_allclose(f2.sum(axis=1).to_numpy(), 0.0, atol=1e-10)
    np.testing.assert_allclose(f2.abs().sum(axis=1).to_numpy(), 2.0, atol=1e-10)

    f3 = build_test_portfolios(returns_panel, "test", 3, rng)
    assert f3.shape == (1, n_assets)
    np.testing.assert_allclose(f3.iloc[0].to_numpy(), 1.0 / n_assets)

    sigma_df = pd.DataFrame(np.eye(n_assets), index=cols, columns=cols)
    f4 = build_test_portfolios(returns_panel, "test", 4, rng, sigma=sigma_df)
    assert 1 <= f4.shape[0] <= 2
    np.testing.assert_allclose(f4.sum(axis=1).to_numpy(), 1.0, atol=1e-8)

    try:
        build_test_portfolios(returns_panel, "test", 4, rng)
        assert False, "family 4 without sigma must raise"
    except RiskValidationError:
        pass

    f5 = build_test_portfolios(returns_panel, "test", 5, rng)     # not-computed-yet fallback
    assert f5.empty

    ext = pd.DataFrame([[1.0] + [0.0] * (n_assets - 1)], columns=cols)
    f6 = build_test_portfolios(returns_panel, "test", 6, rng, external=ext)
    assert f6.shape == (1, n_assets)

    try:
        build_test_portfolios(returns_panel, "test", 7, rng)
        assert False, "unknown family must raise"
    except RiskValidationError:
        pass
