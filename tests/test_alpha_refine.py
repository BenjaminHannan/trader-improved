"""Hand-built arithmetic tests for zscore, refine, combine and the factor registry.
No production.signals import — score panels are constructed directly."""
from __future__ import annotations

import shutil

import numpy as np
import pandas as pd
import pytest
import yaml

from production.alpha.combine import (combine_alphas, combine_alphas_grinold,
                                      factor_momentum_tilt, score_correlation,
                                      single_factor_returns)
from production.alpha.refine import refine_alpha
from production.alpha.registry import FactorRegistry, GateStats
from production.alpha.zscore import mad, winsorize, zscore_scores
from production.core.config import CONFIG_DIR


# ================================================================= zscore
def test_winsorize_clips_outlier_to_exact_median_plus_3mad():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 100.0])
    med = np.median(x)                       # 3.5
    scaled_mad = 1.4826 * mad(x)
    bound = med + 3.0 * scaled_mad
    w = winsorize(x, n_mad=3.0)
    assert w[-1] == pytest.approx(bound)     # the planted outlier hits the upper bound
    assert np.all(w[:-1] == x[:-1])          # in-range values are untouched


def test_winsorize_no_dispersion_returns_input():
    x = np.array([7.0, 7.0, 7.0, 7.0])
    assert np.all(winsorize(x) == x)


def test_zscore_stats_mean_zero_std_one_within_group():
    ids = [f"EQ:S{i}" for i in range(6)]
    sleeve_map = pd.Series({i: "equity" for i in ids})
    panel = pd.DataFrame({
        "obs_date": pd.Timestamp("2020-01-01"),
        "instrument_id": ids,
        "value": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    })
    z = zscore_scores(panel, sleeve_map)
    assert z["value"].mean() == pytest.approx(0.0, abs=1e-12)
    assert z["value"].std(ddof=0) == pytest.approx(1.0)


def test_zscore_winsorizes_before_standardizing():
    ids = [f"EQ:S{i}" for i in range(6)]
    sleeve_map = pd.Series({i: "equity" for i in ids})
    vals = [1.0, 2.0, 3.0, 4.0, 5.0, 100.0]
    panel = pd.DataFrame({"obs_date": pd.Timestamp("2020-01-01"),
                          "instrument_id": ids, "value": vals})
    z = zscore_scores(panel, sleeve_map).set_index("instrument_id")["value"]
    # reproduce: winsorize -> standardize with population std
    xw = winsorize(np.array(vals))
    expected = (xw - xw.mean()) / xw.std()
    for k, iid in enumerate(ids):
        assert z.loc[iid] == pytest.approx(expected[k])


def test_zscore_drops_groups_smaller_than_three():
    ids = ["EQ:A", "EQ:B", "CR:X", "CR:Y", "CR:Z"]
    sleeve_map = pd.Series({"EQ:A": "equity", "EQ:B": "equity",
                            "CR:X": "crypto", "CR:Y": "crypto", "CR:Z": "crypto"})
    panel = pd.DataFrame({"obs_date": pd.Timestamp("2020-01-01"),
                          "instrument_id": ids, "value": [1.0, 2.0, 1.0, 2.0, 3.0]})
    z = zscore_scores(panel, sleeve_map)
    # equity has only 2 names -> dropped; crypto (3) kept
    assert set(z["instrument_id"]) == {"CR:X", "CR:Y", "CR:Z"}


def test_zscore_drops_zero_dispersion_group():
    ids = [f"EQ:S{i}" for i in range(4)]
    sleeve_map = pd.Series({i: "equity" for i in ids})
    panel = pd.DataFrame({"obs_date": pd.Timestamp("2020-01-01"),
                          "instrument_id": ids, "value": [5.0, 5.0, 5.0, 5.0]})
    assert zscore_scores(panel, sleeve_map).empty


# ================================================================= refine
def test_refine_alpha_exact_arithmetic_scalar_ic():
    z = pd.DataFrame({"obs_date": pd.Timestamp("2020-01-01"),
                      "instrument_id": ["A", "B", "C"], "value": [1.0, -2.0, 0.5]})
    vol = pd.DataFrame({"obs_date": pd.Timestamp("2020-01-01"),
                        "instrument_id": ["A", "B", "C"], "value": [0.02, 0.03, 0.04]})
    out = refine_alpha(z, 0.10, vol).set_index("instrument_id")["value"]
    assert out.loc["A"] == pytest.approx(0.02 * 0.10 * 1.0)
    assert out.loc["B"] == pytest.approx(0.03 * 0.10 * -2.0)
    assert out.loc["C"] == pytest.approx(0.04 * 0.10 * 0.5)


def test_refine_alpha_drops_name_missing_sigma():
    z = pd.DataFrame({"obs_date": pd.Timestamp("2020-01-01"),
                      "instrument_id": ["A", "B", "C"], "value": [1.0, 2.0, 3.0]})
    vol = pd.DataFrame({"obs_date": pd.Timestamp("2020-01-01"),
                        "instrument_id": ["A", "C"], "value": [0.02, 0.04]})
    out = refine_alpha(z, 0.10, vol)
    assert set(out["instrument_id"]) == {"A", "C"}   # B dropped (no sigma)


def test_refine_alpha_per_sleeve_ic_mapping():
    z = pd.DataFrame({
        "obs_date": pd.Timestamp("2020-01-01"),
        "instrument_id": ["A", "B"],
        "value": [1.0, 1.0],
        "sleeve": ["equity", "crypto"],
    })
    vol = pd.DataFrame({"obs_date": pd.Timestamp("2020-01-01"),
                        "instrument_id": ["A", "B"], "value": [0.02, 0.05]})
    out = refine_alpha(z, {"equity": 0.10, "crypto": 0.20}, vol).set_index("instrument_id")["value"]
    assert out.loc["A"] == pytest.approx(0.02 * 0.10 * 1.0)
    assert out.loc["B"] == pytest.approx(0.05 * 0.20 * 1.0)


# ================================================================= combine
def test_combine_outer_aligned_sum():
    p1 = pd.DataFrame({"obs_date": pd.Timestamp("2020-01-01"),
                       "instrument_id": ["A", "B"], "value": [1.0, 2.0]})
    p2 = pd.DataFrame({"obs_date": pd.Timestamp("2020-01-01"),
                       "instrument_id": ["B", "C"], "value": [10.0, 20.0]})
    out = combine_alphas({"f1": p1, "f2": p2}).set_index("instrument_id")["value"]
    assert out.loc["A"] == pytest.approx(1.0)    # only f1 -> f2 contributes 0
    assert out.loc["B"] == pytest.approx(12.0)   # both factors
    assert out.loc["C"] == pytest.approx(20.0)   # only f2
    assert set(out.index) == {"A", "B", "C"}     # union, no fabricated cells


def test_combine_empty_input():
    assert combine_alphas({}).empty


# ============================================= combine — Grinold (correlation-aware)
def _z_panel(ids, vals, date="2020-01-01"):
    return pd.DataFrame({"obs_date": pd.Timestamp(date), "instrument_id": list(ids),
                         "value": list(vals)})


def test_grinold_duplicated_signal_no_double_count():
    """Two identical z panels with equal ICs: the Grinold blend collapses them to ~one
    signal's worth of alpha, while the plain sum double-counts to ~2x."""
    ids = ["A", "B", "C", "D", "E"]
    z = _z_panel(ids, [1.0, -2.0, 0.5, -0.5, 1.5])
    vol = _z_panel(ids, [0.02] * 5)
    ic = 0.10

    # Score correlation of two identical panels is exactly 1 -> C = all-ones.
    corr = pd.DataFrame([[1.0, 1.0], [1.0, 1.0]], index=["f1", "f2"], columns=["f1", "f2"])
    g = combine_alphas_grinold({"f1": z, "f2": z.copy()},
                               {"f1": ic, "f2": ic}, vol, corr, ridge=0.10)
    g = g.set_index("instrument_id")["value"]

    a_single = refine_alpha(z, ic, vol)
    single = a_single.set_index("instrument_id")["value"]

    # plain: two refined copies summed -> exactly 2x the single-factor alpha (the bug).
    plain = combine_alphas({"f1": a_single, "f2": a_single.copy()})
    plain = plain.set_index("instrument_id")["value"]

    # Grinold ratio is a known answer: w1=w2=ic/(2-ridge), so g = 2/(2-ridge) * single.
    ratio_g = (g / single).to_numpy()
    ratio_plain = (plain / single).to_numpy()
    assert np.allclose(ratio_g, 2.0 / (2.0 - 0.10))          # ~1.0526 exact
    assert np.allclose(ratio_plain, 2.0)                     # the double-count
    # near the single signal (not near 2x): comfortably inside the "~single" band.
    assert np.all(np.abs(ratio_g - 1.0) < 0.06)


def test_grinold_independent_reduces_to_plain_sum():
    """C = I special case: the Grinold blend equals the plain per-factor alpha sum."""
    ids = ["A", "B", "C", "D"]
    z1 = _z_panel(ids, [1.0, -1.0, 0.5, -0.5])
    z2 = _z_panel(ids, [-0.5, 1.5, -1.0, 0.0])
    vol = _z_panel(ids, [0.02, 0.03, 0.04, 0.05])
    ic = {"f1": 0.10, "f2": 0.05}
    C = pd.DataFrame(np.eye(2), index=["f1", "f2"], columns=["f1", "f2"])

    g = combine_alphas_grinold({"f1": z1, "f2": z2}, ic, vol, C, ridge=0.0)
    g = g.set_index("instrument_id")["value"]

    a1 = refine_alpha(z1, ic["f1"], vol)
    a2 = refine_alpha(z2, ic["f2"], vol)
    plain = combine_alphas({"f1": a1, "f2": a2}).set_index("instrument_id")["value"]

    assert np.allclose(g.reindex(plain.index).to_numpy(), plain.to_numpy())


def test_grinold_weights_known_answer():
    """w = C_r^{-1} ic recovered directly: name scored by a single factor carries w_k."""
    rho, ridge = 0.5, 0.10
    ic = np.array([0.05, 0.03])
    C = pd.DataFrame([[1.0, rho], [rho, 1.0]], index=["f1", "f2"], columns=["f1", "f2"])
    C_r = (1 - ridge) * C.to_numpy() + ridge * np.eye(2)
    w_expected = np.linalg.solve(C_r, ic)

    # f1 scores only A (z=1), f2 scores only B (z=1), unit vol -> alpha_A=w1, alpha_B=w2.
    z1 = _z_panel(["A"], [1.0])
    z2 = _z_panel(["B"], [1.0])
    vol = _z_panel(["A", "B"], [1.0, 1.0])
    out = combine_alphas_grinold({"f1": z1, "f2": z2},
                                 {"f1": ic[0], "f2": ic[1]}, vol, C, ridge=ridge)
    out = out.set_index("instrument_id")["value"]
    assert out.loc["A"] == pytest.approx(w_expected[0])
    assert out.loc["B"] == pytest.approx(w_expected[1])


def test_grinold_negative_ic_gets_negative_weight():
    """A negative-IC factor keeps a negative weight through the C^{-1} inversion."""
    rho, ridge = 0.5, 0.10
    ic = {"f1": 0.05, "f2": -0.03}
    C = pd.DataFrame([[1.0, rho], [rho, 1.0]], index=["f1", "f2"], columns=["f1", "f2"])
    z1 = _z_panel(["A"], [1.0])
    z2 = _z_panel(["B"], [1.0])
    vol = _z_panel(["A", "B"], [1.0, 1.0])
    out = combine_alphas_grinold({"f1": z1, "f2": z2}, ic, vol, C, ridge=ridge)
    out = out.set_index("instrument_id")["value"]
    assert out.loc["A"] > 0.0       # positive-IC factor -> positive weight
    assert out.loc["B"] < 0.0       # negative-IC factor -> negative weight (sign preserved)


def test_score_correlation_pit_and_window():
    """score_correlation is a pure function of scores dated <= as_of, and honors window."""
    ids = [f"S{i}" for i in range(8)]
    dates = pd.bdate_range("2020-01-01", periods=120)
    rng = np.random.default_rng(7)
    # two positively-correlated factors: z2 = z1 + noise on each date.
    rows1, rows2 = [], []
    for d in dates:
        base = rng.normal(size=len(ids))
        rows1.append(pd.DataFrame({"obs_date": d, "instrument_id": ids, "value": base}))
        rows2.append(pd.DataFrame({"obs_date": d, "instrument_id": ids,
                                   "value": base + 0.3 * rng.normal(size=len(ids))}))
    z1 = pd.concat(rows1, ignore_index=True)
    z2 = pd.concat(rows2, ignore_index=True)

    as_of = dates[80]
    C = score_correlation({"f1": z1, "f2": z2}, as_of, window=252, min_obs=30)
    assert C.loc["f1", "f2"] == pytest.approx(C.loc["f2", "f1"])   # symmetric
    assert C.loc["f1", "f1"] == pytest.approx(1.0)                 # unit diagonal
    assert C.loc["f1", "f2"] > 0.5                                 # genuine correlation seen
    base_val = C.loc["f1", "f2"]

    # PIT: corrupt every score strictly after as_of; the matrix must not move.
    def corrupt(z):
        z = z.copy()
        fut = z["obs_date"] > as_of
        z.loc[fut, "value"] = z.loc[fut, "value"].to_numpy() * 1e6 + 42.0
        return z
    C2 = score_correlation({"f1": corrupt(z1), "f2": corrupt(z2)},
                           as_of, window=252, min_obs=30)
    assert C2.loc["f1", "f2"] == pytest.approx(base_val)

    # min_obs: with a window shorter than the required min_obs, no pair clears the bar -> 0.
    C_short = score_correlation({"f1": z1, "f2": z2}, as_of, window=10, min_obs=30)
    assert C_short.loc["f1", "f2"] == 0.0


# ==================================================== factor momentum: tilt arithmetic
def test_factor_momentum_tilt_arithmetic_and_clip():
    up = pd.Series([0.01, 0.02, -0.005])       # sum > 0
    down = pd.Series([-0.01, -0.02, 0.005])    # sum < 0
    # sign(+)*gamma -> 1+gamma ; sign(-)*gamma -> 1-gamma
    assert factor_momentum_tilt(up, 0.25) == pytest.approx(1.25)
    assert factor_momentum_tilt(down, 0.25) == pytest.approx(0.75)
    # clip bounds [0, 2]: a large gamma cannot flip sign or exceed 2.
    assert factor_momentum_tilt(up, 3.0) == pytest.approx(2.0)
    assert factor_momentum_tilt(down, 3.0) == pytest.approx(0.0)
    # zero trailing sum -> sign 0 -> exactly 1.0 (no tilt); empty -> 1.0 (no evidence)
    assert factor_momentum_tilt(pd.Series([0.01, -0.01]), 0.25) == pytest.approx(1.0)
    assert factor_momentum_tilt(pd.Series([], dtype=float), 0.25) == pytest.approx(1.0)


def test_factor_momentum_gamma_zero_is_identity():
    """gamma=0 -> tilt is exactly 1.0 for any trailing series, so ic*tilt == ic and the
    combination is bit-identical to the no-feature path."""
    for s in (pd.Series([0.5, 0.5]), pd.Series([-0.5, -0.5]), pd.Series([], dtype=float)):
        assert factor_momentum_tilt(s, 0.0) == 1.0

    # applied to a Grinold combine: tilting every ic by 1.0 reproduces the untilted output.
    ids = ["A", "B", "C", "D"]
    z1 = _z_panel(ids, [1.0, -1.0, 0.5, -0.5])
    z2 = _z_panel(ids, [-0.5, 1.5, -1.0, 0.0])
    vol = _z_panel(ids, [0.02, 0.03, 0.04, 0.05])
    ic = {"f1": 0.10, "f2": 0.05}
    C = pd.DataFrame(np.eye(2), index=["f1", "f2"], columns=["f1", "f2"])
    base = combine_alphas_grinold({"f1": z1, "f2": z2}, ic, vol, C, ridge=0.0)
    tilted_ic = {f: ic[f] * factor_momentum_tilt(pd.Series([1.0]), 0.0) for f in ic}
    same = combine_alphas_grinold({"f1": z1, "f2": z2}, tilted_ic, vol, C, ridge=0.0)
    assert np.allclose(base.set_index("instrument_id")["value"].to_numpy(),
                       same.set_index("instrument_id")["value"].to_numpy())


def test_factor_momentum_losing_factor_downweighted_vs_ic_only():
    """A persistently-losing factor's IC is tilted down (0.75), so its alpha contribution
    is strictly smaller than the IC-only (untilted) weight — the Ehsani-Linnainmaa effect."""
    losing = pd.Series([-0.02, -0.01, -0.03, -0.015])   # persistently negative
    winning = pd.Series([0.02, 0.01, 0.03, 0.015])
    tilt_lose = factor_momentum_tilt(losing, 0.25)
    tilt_win = factor_momentum_tilt(winning, 0.25)
    assert tilt_lose == pytest.approx(0.75) and tilt_lose < 1.0
    assert tilt_win == pytest.approx(1.25)

    # C = I so w = ic; a name scored only by the losing factor carries sigma*ic*tilt*z.
    z = _z_panel(["A"], [1.0])
    vol = _z_panel(["A"], [1.0])
    ic_only = combine_alphas_grinold({"f1": z}, {"f1": 0.10}, vol,
                                     pd.DataFrame([[1.0]], index=["f1"], columns=["f1"]),
                                     ridge=0.0).set_index("instrument_id")["value"]
    tilted = combine_alphas_grinold({"f1": z}, {"f1": 0.10 * tilt_lose}, vol,
                                    pd.DataFrame([[1.0]], index=["f1"], columns=["f1"]),
                                    ridge=0.0).set_index("instrument_id")["value"]
    assert tilted.loc["A"] < ic_only.loc["A"]
    assert tilted.loc["A"] == pytest.approx(ic_only.loc["A"] * 0.75)


# ==================================================== single_factor_returns: known answer
def _price_panel(price_by_id_by_date: dict, dates) -> pd.DataFrame:
    rows = []
    for iid, series in price_by_id_by_date.items():
        for d, px in zip(dates, series):
            rows.append({"obs_date": d, "instrument_id": iid, "close": float(px)})
    return pd.DataFrame(rows)


def test_single_factor_returns_known_answer_tiny_panel():
    """5 names, one week (Mon-Fri): long the top-z name (A), short the bottom (E), one-day
    lag. A jumps +10% on Tue, E drops -5% on Fri; every other move is zero. The single
    weekly long-short return is (1+0.10)*(1+0.05)-1 = 0.155 exactly."""
    dates = pd.bdate_range("2020-01-06", periods=5)   # Mon .. Fri (one W-FRI bucket)
    ids = ["EQ:A", "EQ:B", "EQ:C", "EQ:D", "EQ:E"]
    prices = _price_panel({
        "EQ:A": [100, 110, 110, 110, 110],   # +10% Tue, then flat
        "EQ:B": [100, 100, 100, 100, 100],
        "EQ:C": [100, 100, 100, 100, 100],
        "EQ:D": [100, 100, 100, 100, 100],
        "EQ:E": [100, 100, 100, 100, 95],    # -5% Fri
    }, dates)
    # constant z ranking A>B>C>D>E on every date.
    zrows = []
    for d in dates:
        zrows.append(pd.DataFrame({"obs_date": d, "instrument_id": ids,
                                   "value": [5.0, 4.0, 3.0, 2.0, 1.0]}))
    z_panel = pd.concat(zrows, ignore_index=True)

    ser = single_factor_returns(z_panel, prices, ids, quantile=0.2)
    assert len(ser) == 1
    assert ser.iloc[0] == pytest.approx(1.10 * 1.05 - 1.0)   # 0.155


def test_single_factor_returns_embargo_future_prices_do_not_move_past_weeks():
    """PIT/embargo: corrupting prices strictly after a pivot changes only weekly returns
    for weeks ending after it; the trailing weeks a tilt at the pivot would read are
    bit-identical. This is the property the engine's `index < t` embargo relies on."""
    dates = pd.bdate_range("2020-01-06", periods=15)  # 3 weeks (Fridays: Jan10, Jan17, Jan24)
    ids = ["EQ:A", "EQ:B", "EQ:C", "EQ:D", "EQ:E"]
    rng = np.random.default_rng(4)
    price_map = {iid: (100.0 * np.cumprod(1.0 + rng.normal(0.001, 0.02, len(dates))))
                 for iid in ids}
    prices = _price_panel(price_map, dates)
    zrows = [pd.DataFrame({"obs_date": d, "instrument_id": ids,
                           "value": [5.0, 4.0, 3.0, 2.0, 1.0]}) for d in dates]
    z_panel = pd.concat(zrows, ignore_index=True)

    ser_clean = single_factor_returns(z_panel, prices, ids)

    pivot = pd.Timestamp("2020-01-17")               # end of week 2
    corrupt = prices.copy()
    fut = corrupt["obs_date"] > pivot
    corrupt.loc[fut, "close"] = corrupt.loc[fut, "close"].to_numpy() * 7.0 + 13.0
    ser_corrupt = single_factor_returns(z_panel, corrupt, ids)

    pre_c = ser_clean[ser_clean.index <= pivot]
    pre_k = ser_corrupt[ser_corrupt.index <= pivot]
    assert pre_c.index.equals(pre_k.index)
    assert np.allclose(pre_c.to_numpy(), pre_k.to_numpy(), atol=1e-12, rtol=0.0)
    # non-vacuous: the corruption did reach a later week.
    assert not np.allclose(ser_clean.reindex(ser_corrupt.index).fillna(0).to_numpy(),
                           ser_corrupt.fillna(0).to_numpy())


# ================================================================= registry gate
def _stats(train_tstat=3.0, train_ic=0.03, oos_ic=0.02,
           halflife=10.0, net=0.01, n_dates=300):
    return GateStats(train_ic=train_ic, train_tstat=train_tstat, oos_ic=oos_ic,
                     decay_halflife_days=halflife, net_validation_return=net,
                     n_dates=n_dates)


@pytest.fixture()
def registry():
    return FactorRegistry()


def test_gate_all_pass(registry):
    v = registry.gate("mom_12_1", _stats())
    assert v.passed
    assert v.reasons == ["all gates passed"]


@pytest.mark.parametrize("tstat,expected", [(1.99, False), (2.01, True)])
def test_gate_tstat_boundary(registry, tstat, expected):
    v = registry.gate("mom_12_1", _stats(train_tstat=tstat))
    assert v.passed is expected


def test_gate_oos_sign_flip_fails(registry):
    v = registry.gate("mom_12_1", _stats(train_ic=0.03, oos_ic=-0.02))
    assert not v.passed
    assert any("sign disagrees" in r for r in v.reasons)


def test_gate_oos_magnitude_boundary(registry):
    assert not registry.gate("mom_12_1", _stats(oos_ic=0.004)).passed  # < 0.005
    assert registry.gate("mom_12_1", _stats(oos_ic=0.006)).passed      # >= 0.005


@pytest.mark.parametrize("halflife,expected", [(4.0, False), (5.0, True)])
def test_gate_halflife_boundary(registry, halflife, expected):
    # min_decay_halflife_days = 5 (weekly interval) in factors.yaml
    v = registry.gate("mom_12_1", _stats(halflife=halflife))
    assert v.passed is expected


def test_gate_negative_net_fails_when_required(registry):
    assert registry.cfg["gate"]["require_net_positive_validation"] is True
    v = registry.gate("mom_12_1", _stats(net=-0.001))
    assert not v.passed
    assert any("net validation" in r for r in v.reasons)


# ================================================= registry record/save roundtrip
def test_record_and_save_roundtrip(tmp_path):
    src = CONFIG_DIR / "factors.yaml"
    dst = tmp_path / "factors.yaml"
    shutil.copy(src, dst)

    reg = FactorRegistry(dst)
    n0 = reg.n_trials
    passing = reg.gate("mom_12_1", _stats())
    failing = reg.gate("low_vol", _stats(train_tstat=0.5, oos_ic=-0.02))
    reg.record("mom_12_1", passing)
    reg.record("low_vol", failing)
    reg.save()

    reloaded = yaml.safe_load(open(dst))
    assert reloaded["factors"]["mom_12_1"]["status"] == "accepted"
    assert reloaded["factors"]["low_vol"]["status"] == "rejected"
    assert reloaded["n_trials"] == n0 + 2               # bumped once per record
    # gate stats persisted
    assert reloaded["factors"]["mom_12_1"]["gate_stats"]["train_tstat"] == pytest.approx(3.0)
    # unrelated factors untouched — compare against the SOURCE registry rather than
    # hardcoding statuses (the live registry's statuses legitimately change once the
    # real gate has run; the invariant is that record() only touches what it records).
    src_reg = yaml.safe_load(open(src))
    for other in ("carry_funding", "str_reversal_1m"):
        assert reloaded["factors"][other]["status"] ==             src_reg["factors"][other]["status"]
    # structure preserved (gate block + keys intact)
    assert reloaded["gate"]["train_ic_tstat_min"] == 2.0
    assert set(reloaded["factors"]["mom_12_1"]["sleeves"]) == \
        {"equity", "crypto", "fx_etf", "commodity_etf",
         "rates_etf", "intl_etf", "sector_etf"}


# ======================================================= registry demote
def test_demote_flips_status_and_stores_evidence(tmp_path):
    src = CONFIG_DIR / "factors.yaml"
    dst = tmp_path / "factors.yaml"
    shutil.copy(src, dst)

    reg = FactorRegistry(dst)
    # basis_carry is accepted in the live fixture; demote it with representative
    # factor-health evidence.
    evidence = {"mean_sign_adjusted_ic": -0.001, "month_clustered_t": -0.3,
               "n_days": 500, "window_start": "2024-07-10", "window_end": "2026-07-10"}
    pre_gate_stats = dict(reg.cfg["factors"]["basis_carry"]["gate_stats"])

    reg.demote("basis_carry", evidence)

    assert reg.cfg["factors"]["basis_carry"]["status"] == "demoted"
    assert reg.cfg["factors"]["basis_carry"]["health"] == evidence
    # gate_stats (the original promotion evidence) is untouched by demotion.
    assert reg.cfg["factors"]["basis_carry"]["gate_stats"] == pre_gate_stats


def test_demote_excludes_factor_from_accepted_view(tmp_path):
    src = CONFIG_DIR / "factors.yaml"
    dst = tmp_path / "factors.yaml"
    shutil.copy(src, dst)

    reg = FactorRegistry(dst)
    assert "basis_carry" in reg.factors(status="accepted")

    reg.demote("basis_carry", {"month_clustered_t": -1.0})

    assert "basis_carry" not in reg.factors(status="accepted")
    assert reg.factors(status="demoted") == {"basis_carry": reg.cfg["factors"]["basis_carry"]}


def test_demote_does_not_touch_trial_ledger(tmp_path):
    src = CONFIG_DIR / "factors.yaml"
    dst = tmp_path / "factors.yaml"
    shutil.copy(src, dst)

    reg = FactorRegistry(dst)
    n0 = reg.n_trials

    reg.demote("basis_carry", {"month_clustered_t": -1.0})

    assert reg.n_trials == n0                    # demotion burns no trial


def test_demote_unknown_factor_raises(registry):
    with pytest.raises(KeyError):
        registry.demote("not_a_real_factor", {})


def test_demote_save_roundtrip(tmp_path):
    src = CONFIG_DIR / "factors.yaml"
    dst = tmp_path / "factors.yaml"
    shutil.copy(src, dst)

    reg = FactorRegistry(dst)
    n0 = reg.n_trials
    evidence = {"month_clustered_t": -0.42, "n_days": 480}
    reg.demote("basis_carry", evidence)
    reg.save()

    reloaded = yaml.safe_load(open(dst))
    assert reloaded["factors"]["basis_carry"]["status"] == "demoted"
    assert reloaded["factors"]["basis_carry"]["health"] == evidence
    assert reloaded["n_trials"] == n0             # save roundtrip preserves the ledger too
    # unrelated factors untouched
    src_reg = yaml.safe_load(open(src))
    for other in ("carry_rate_diff", "carry_funding"):
        assert reloaded["factors"][other]["status"] == src_reg["factors"][other]["status"]


def test_signal_class_dotted_path_matches_yaml(registry):
    # do NOT import the signal (written in parallel); just assert the registry exposes the
    # resolver and reads the dotted path from yaml without touching production.signals.
    assert hasattr(registry, "signal_class")
    assert registry.cfg["factors"]["mom_12_1"]["signal"] == \
        "production.signals.momentum.Momentum12m1m"
