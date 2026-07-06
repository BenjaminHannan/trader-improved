"""Known-answer tests for the IC machinery. No production.signals import — every panel
here is synthetic and constructed by hand, so the arithmetic is fully determined."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from production.alpha.ic import (decay_halflife, forward_returns, ic_decay,
                                 ic_tstat, rank_ic, rolling_shrunk_ic, shrunk_ic)


# ------------------------------------------------------------------ forward_returns
def test_forward_returns_positional_known_answer():
    # two instruments, tiny explicit close series.
    dates = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"])
    df = pd.DataFrame({
        "obs_date": list(dates) * 2,
        "instrument_id": ["A"] * 4 + ["B"] * 4,
        "close": [10.0, 11.0, 12.0, 13.0, 100.0, 90.0, 99.0, 110.0],
    })
    fwd = forward_returns(df, horizon_days=1)
    # last row of each instrument has no forward obs -> dropped (4 - 1) * 2 = 6 rows
    assert len(fwd) == 6
    a = fwd[fwd.instrument_id == "A"].set_index("obs_date")["fwd_ret"]
    assert a.loc["2020-01-01"] == pytest.approx(11.0 / 10.0 - 1)
    assert a.loc["2020-01-03"] == pytest.approx(13.0 / 12.0 - 1)
    b = fwd[fwd.instrument_id == "B"].set_index("obs_date")["fwd_ret"]
    assert b.loc["2020-01-02"] == pytest.approx(99.0 / 90.0 - 1)


def test_forward_returns_horizon_two():
    dates = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"])
    df = pd.DataFrame({"obs_date": dates, "instrument_id": "A",
                       "close": [10.0, 11.0, 12.0, 13.0]})
    fwd = forward_returns(df, horizon_days=2).set_index("obs_date")["fwd_ret"]
    assert len(fwd) == 2  # last 2 rows dropped
    assert fwd.loc["2020-01-01"] == pytest.approx(12.0 / 10.0 - 1)
    assert fwd.loc["2020-01-02"] == pytest.approx(13.0 / 11.0 - 1)


# ------------------------------------------------------------------------- rank_ic
def _panel(n_names, values_by_name, date="2020-01-01", col="value"):
    return pd.DataFrame({
        "obs_date": pd.Timestamp(date),
        "instrument_id": list(values_by_name.keys()),
        col: list(values_by_name.values()),
    })


def test_rank_ic_perfect_monotone_is_one():
    ids = [f"EQ:S{i}" for i in range(8)]
    sleeve_map = pd.Series({i: "equity" for i in ids})
    fwd_vals = {i: 0.01 * k for k, i in enumerate(ids)}          # increasing fwd returns
    score_vals = {i: float(k) for k, i in enumerate(ids)}        # scores == rank of fwd
    scores = _panel(8, score_vals, col="value")
    fwd = _panel(8, fwd_vals, col="fwd_ret")
    ic = rank_ic(scores, fwd, sleeve_map)
    assert len(ic) == 1
    assert ic["rank_ic"].iloc[0] == pytest.approx(1.0)
    assert ic["n_names"].iloc[0] == 8


def test_rank_ic_anti_monotone_is_minus_one():
    ids = [f"EQ:S{i}" for i in range(8)]
    sleeve_map = pd.Series({i: "equity" for i in ids})
    fwd_vals = {i: 0.01 * k for k, i in enumerate(ids)}
    score_vals = {i: float(-k) for k, i in enumerate(ids)}       # exactly reversed
    scores = _panel(8, score_vals, col="value")
    fwd = _panel(8, fwd_vals, col="fwd_ret")
    ic = rank_ic(scores, fwd, sleeve_map)
    assert ic["rank_ic"].iloc[0] == pytest.approx(-1.0)


def test_rank_ic_random_is_near_zero_over_many_dates():
    rng = np.random.default_rng(0)
    n_dates, n_names = 250, 12
    ids = [f"EQ:S{i}" for i in range(n_names)]
    sleeve_map = pd.Series({i: "equity" for i in ids})
    dates = pd.bdate_range("2020-01-01", periods=n_dates)
    s_rows, f_rows = [], []
    for d in dates:
        for i in ids:
            s_rows.append((d, i, rng.normal()))
            f_rows.append((d, i, rng.normal()))   # independent of scores
    scores = pd.DataFrame(s_rows, columns=["obs_date", "instrument_id", "value"])
    fwd = pd.DataFrame(f_rows, columns=["obs_date", "instrument_id", "fwd_ret"])
    ic = rank_ic(scores, fwd, sleeve_map)
    assert abs(ic["rank_ic"].mean()) < 0.1


def test_rank_ic_requires_five_names():
    ids = [f"EQ:S{i}" for i in range(4)]
    sleeve_map = pd.Series({i: "equity" for i in ids})
    scores = _panel(4, {i: float(k) for k, i in enumerate(ids)}, col="value")
    fwd = _panel(4, {i: 0.01 * k for k, i in enumerate(ids)}, col="fwd_ret")
    ic = rank_ic(scores, fwd, sleeve_map)
    assert ic.empty  # fewer than 5 names -> dropped


# --------------------------------------------------------------------- ic_tstat
def test_ic_tstat_arithmetic():
    ic = pd.Series([0.02, 0.04, 0.03, 0.05, 0.01])
    expected = ic.mean() / ic.std(ddof=1) * np.sqrt(len(ic))
    assert ic_tstat(ic) == pytest.approx(expected)


def test_ic_tstat_zero_dispersion_is_nan():
    assert np.isnan(ic_tstat(pd.Series([0.03, 0.03, 0.03])))


# ---------------------------------------------------------------------- shrunk_ic
def test_shrunk_ic_arithmetic():
    ic = pd.Series([0.1] * 42)          # n = 42
    assert shrunk_ic(ic, n0=126) == pytest.approx(0.1 * 42 / (42 + 126))


# ------------------------------------------------------ rolling_shrunk_ic embargo
def _ic_series():
    idx = pd.to_datetime([
        "2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04", "2020-01-05",
        "2020-01-06", "2020-01-07", "2020-01-08", "2020-01-09", "2020-01-10",
    ])
    return pd.Series(np.linspace(0.01, 0.10, len(idx)), index=idx)


def test_rolling_shrunk_ic_embargoes_unresolved_tail():
    s = _ic_series()
    horizon = 5
    roll = rolling_shrunk_ic(s, window=252, n0=126, horizon_days=horizon)
    t = pd.Timestamp("2020-01-10")
    # eligible IC dates are d <= t - 5 days = 2020-01-05
    eligible = s[s.index <= t - pd.Timedelta(days=horizon)]
    assert eligible.index.max() == pd.Timestamp("2020-01-05")
    expected = shrunk_ic(eligible, n0=126)
    assert roll.loc[t] == pytest.approx(expected)


def test_rolling_shrunk_ic_corrupting_tail_does_not_change_value_at_t():
    s = _ic_series()
    horizon = 5
    t = pd.Timestamp("2020-01-10")
    base = rolling_shrunk_ic(s, horizon_days=horizon).loc[t]
    # corrupt every IC in the unresolved half-open window (t - horizon, t]
    corrupt = s.copy()
    mask = (corrupt.index > t - pd.Timedelta(days=horizon)) & (corrupt.index <= t)
    assert mask.sum() > 0  # the corruption actually touches something
    corrupt[mask] = 999.0
    after = rolling_shrunk_ic(corrupt, horizon_days=horizon).loc[t]
    assert after == pytest.approx(base)


def test_rolling_shrunk_ic_early_dates_have_no_eligible_history():
    s = _ic_series()
    roll = rolling_shrunk_ic(s, horizon_days=5)
    # first date has no d <= t - 5 -> NaN
    assert np.isnan(roll.loc[pd.Timestamp("2020-01-01")])


# ------------------------------------------------------------ ic_decay / halflife
def test_decay_halflife_interpolation_known_answer():
    # base |IC| at h=1 is 0.10; crosses 0.05 linearly between h=5 (0.06) and h=10 (0.04)
    decay = pd.DataFrame({
        "sleeve": ["equity"] * 4,
        "horizon": [1, 2, 5, 10],
        "ic": [0.10, 0.08, 0.06, 0.04],
    })
    hl = decay_halflife(decay)
    # half = 0.05; between (5, 0.06) and (10, 0.04): frac = (0.06-0.05)/(0.06-0.04)=0.5
    assert hl == pytest.approx(5 + 0.5 * (10 - 5))


def test_decay_halflife_never_crosses_is_inf():
    decay = pd.DataFrame({"sleeve": ["equity"] * 3, "horizon": [1, 2, 5],
                          "ic": [0.10, 0.09, 0.08]})
    assert np.isinf(decay_halflife(decay))


def test_decay_halflife_averages_absolute_ic_across_sleeves():
    # two sleeves; per-horizon mean |IC| curve = [0.10, 0.04]; half=0.05 crossed
    # between h=1 (0.10) and h=2 (0.04): frac = (0.10-0.05)/(0.10-0.04)
    decay = pd.DataFrame({
        "sleeve": ["equity", "crypto", "equity", "crypto"],
        "horizon": [1, 1, 2, 2],
        "ic": [0.10, -0.10, 0.04, -0.04],
    })
    hl = decay_halflife(decay)
    assert hl == pytest.approx(1 + (0.10 - 0.05) / (0.10 - 0.04))


def test_ic_decay_on_constructed_decaying_panel():
    # Signal at t predicts the 1-day forward return plus noise. Over horizon h the
    # forward return sums h independent daily returns, so the signal (which only tracks
    # the first) explains a shrinking share -> IC decays with horizon.
    rng = np.random.default_rng(42)
    n_names, n_dates = 30, 220
    ids = [f"EQ:S{i:02d}" for i in range(n_names)]
    sleeve_map = pd.Series({i: "equity" for i in ids})
    dates = pd.bdate_range("2019-01-01", periods=n_dates)

    # iid small daily returns; build close from cumulative product
    daily = rng.normal(0.0, 0.02, size=(n_dates, n_names))
    close = 100.0 * np.cumprod(1.0 + daily, axis=0)
    price_rows, score_rows = [], []
    for ti, d in enumerate(dates):
        for ni, iid in enumerate(ids):
            price_rows.append((d, iid, close[ti, ni]))
            if ti + 1 < n_dates:
                # score = next-day return + modest noise -> strong h=1 IC, decaying after
                score = daily[ti + 1, ni] + rng.normal(0.0, 0.01)
                score_rows.append((d, iid, score))
    prices = pd.DataFrame(price_rows, columns=["obs_date", "instrument_id", "close"])
    scores = pd.DataFrame(score_rows, columns=["obs_date", "instrument_id", "value"])

    decay = ic_decay(scores, prices, sleeve_map, horizons=(1, 2, 5, 10, 21))
    curve = decay.groupby("horizon")["ic"].apply(lambda x: x.abs().mean())
    assert curve.loc[1] > curve.loc[21]        # IC genuinely decays
    hl = decay_halflife(decay)
    assert np.isfinite(hl)
    assert 1.0 < hl < 21.0
