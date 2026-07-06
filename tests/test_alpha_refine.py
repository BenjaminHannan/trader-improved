"""Hand-built arithmetic tests for zscore, refine, combine and the factor registry.
No production.signals import — score panels are constructed directly."""
from __future__ import annotations

import shutil

import numpy as np
import pandas as pd
import pytest
import yaml

from production.alpha.combine import combine_alphas
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
    # unrelated factors untouched
    assert reloaded["factors"]["carry_funding"]["status"] == "candidate"
    assert reloaded["factors"]["str_reversal_1m"]["status"] == "candidate"
    # structure preserved (gate block + keys intact)
    assert reloaded["gate"]["train_ic_tstat_min"] == 2.0
    assert set(reloaded["factors"]["mom_12_1"]["sleeves"]) == \
        {"equity", "crypto", "fx_etf", "commodity_etf"}


def test_signal_class_dotted_path_matches_yaml(registry):
    # do NOT import the signal (written in parallel); just assert the registry exposes the
    # resolver and reads the dotted path from yaml without touching production.signals.
    assert hasattr(registry, "signal_class")
    assert registry.cfg["factors"]["mom_12_1"]["signal"] == \
        "production.signals.momentum.Momentum12m1m"
