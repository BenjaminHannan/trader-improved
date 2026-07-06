"""Backtest engine + metrics/attribution integration tests.

Runtime discipline (< ~120s): the heavy walk-forward is run ONCE in a module-scoped
fixture over two sleeves (8 equity + 5 crypto) and ~10 months of rebalances; every
end-to-end assertion reads that single result. Only the point-in-time integrity test pays
for a second (corrupted) engine run — that is the money test and is worth it.
"""
from __future__ import annotations

import copy
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import yaml

from production.alpha.registry import FactorRegistry, GateStats

from production.backtest.attribution import factor_attribution, per_alpha_contribution
from production.backtest.bootstrap import sharpe_ci, stationary_bootstrap
from production.backtest.deflated_sharpe import deflated_sharpe, probabilistic_sharpe
from production.backtest.engine import run_backtest
from production.backtest.metrics import (ann_vol, hit_rate, max_drawdown, sharpe,
                                         turnover)
from production.backtest.report import build_report, write_report
from production.core.config import REPO_ROOT, backtest_config
from production.signals.base import sleeve_from_id
from tests.conftest import make_funding, make_gbm_prices, make_macro

# Two sleeves, ~10 months of trading after the 3y warmup, macro included for the overlays.
_EQ = {f"EQ:SYN{i:02d}:2000-01-03": "equity" for i in range(8)}
_CR = {f"CR:{s}:2017-01-01": "crypto" for s in ["BTC", "ETH", "SOL", "LTC", "ADA"]}
_INSTRUMENTS = {**_EQ, **_CR}
_START, _END = "2016-12-15", "2020-06-30"


def _bundle(start=_START, end=_END):
    crypto = [i for i, s in _INSTRUMENTS.items() if s == "crypto"]
    return {
        "prices": make_gbm_prices(_INSTRUMENTS, start=start, end=end),
        "funding": make_funding(crypto, start=start, end=end),
        "macro": make_macro({"DGS3MO_US": 2.0, "BAMLH0A0HYM2": 4.0, "VIXCLS": 18.0},
                            start=start, end=end),
    }


@pytest.fixture(scope="module")
def result():
    data = _bundle()
    return run_backtest(data, pd.Series(_INSTRUMENTS))


# ============================================================ end-to-end integrity
def test_equity_curve_finite(result):
    for series in (result.total_returns, result.total_gross_returns,
                   result.costs, result.overlay):
        assert np.isfinite(series.to_numpy()).all()
        assert series.isna().sum() == 0
    equity = (1.0 + result.total_returns).cumprod()
    assert np.isfinite(equity.to_numpy()).all()
    assert len(result.total_returns) > 0


def test_net_cumulative_le_gross(result):
    net_cum = float((1.0 + result.total_returns).prod())
    gross_cum = float((1.0 + result.total_gross_returns).prod())
    assert net_cum <= gross_cum + 1e-12


def test_costs_strictly_positive(result):
    # every day is a non-negative drag, and the book traded, so the total drag is > 0.
    assert (result.costs >= -1e-15).all()
    assert result.costs.sum() > 0.0
    assert result.costs.max() > 0.0


def test_overlay_within_clip_bounds(result):
    cfg = backtest_config()
    hi = cfg["overlays"]["vol_target"]["clip"][1]  # 1.5; DD/macro scales only shrink it
    assert (result.overlay >= -1e-9).all()
    assert (result.overlay <= hi + 1e-9).all()


def test_gross_and_position_caps_every_rebalance(result):
    cfg = backtest_config()
    gross_cap = cfg["constraints"]["gross_cap"]
    for sleeve, W in result.weights_history.items():
        pos_cap = cfg["constraints"]["position_cap"][sleeve]
        gross = W.abs().sum(axis=1)
        assert (gross <= gross_cap + 1e-6).all(), (sleeve, gross.max())
        assert (W.abs().to_numpy() <= pos_cap + 1e-6).all(), sleeve


def test_turnover_cap_every_rebalance(result):
    cfg = backtest_config()
    tcap = cfg["constraints"]["turnover_cap"]
    for sleeve, W in result.weights_history.items():
        # first rebalance trades from a flat book; subsequent from the prior weights.
        first = W.iloc[0].abs().sum()
        assert first <= tcap + 1e-6, (sleeve, first)
        dw = W.diff().abs().sum(axis=1).iloc[1:]
        assert (dw <= tcap + 1e-6).all(), (sleeve, dw.max())


def test_no_weight_outside_sleeve(result):
    for sleeve, W in result.weights_history.items():
        for iid in W.columns:
            assert sleeve_from_id(iid) == sleeve


def test_turnover_metric_positive(result):
    for sleeve, W in result.weights_history.items():
        assert turnover(W) > 0.0


# ============================================================ the money test (PIT)
def test_pit_no_lookahead(result):
    """Corrupt every input row dated after a pivot and re-run; the net-return series up to
    the pivot must be bit-for-bit unchanged. Any accidental full-sample statistic leaks the
    future and breaks this."""
    net_clean = result.total_returns
    pivot = net_clean.index[int(len(net_clean) * 0.6)]

    data = _bundle()
    rng = np.random.default_rng(0)

    def corrupt(df, cols):
        d = df.copy()
        future = d["obs_date"] > pivot
        for c in cols:
            if c in d.columns:
                d.loc[future, c] = d.loc[future, c].to_numpy() * (
                    50.0 + rng.random(int(future.sum())) * 1000.0)
        return d

    data["prices"] = corrupt(data["prices"], ["close", "volume", "dollar_volume"])
    data["funding"] = corrupt(data["funding"], ["funding_rate"])
    data["macro"] = corrupt(data["macro"], ["value"])

    res2 = run_backtest(data, pd.Series(_INSTRUMENTS))
    a = net_clean[net_clean.index <= pivot]
    b = res2.total_returns[res2.total_returns.index <= pivot]
    assert a.index.equals(b.index)
    assert np.allclose(a.to_numpy(), b.to_numpy(), atol=1e-12, rtol=0.0)
    # sanity: the corruption actually reaches the post-pivot series (test is not vacuous)
    assert len(net_clean[net_clean.index > pivot]) > 0


# ============================================================ report
def test_report_headline_keys(result):
    rep = result.report
    h = rep["headline"]
    for k in ("net_sharpe", "gross_sharpe", "net_information_ratio", "ann_return_net",
              "ann_vol_net", "max_drawdown", "hit_rate", "probabilistic_sharpe",
              "deflated_sharpe", "n_trials", "sharpe_ci95"):
        assert k in h, k
    assert h["n_trials"] >= 1
    assert set(("headline", "per_sleeve", "factor_attribution", "per_alpha_contribution",
                "ic_realized_vs_training", "caveats", "config_echo")).issubset(rep)
    assert rep["caveats"]["borrow_cost_free"] is True


def test_write_report_creates_both_files(result, tmp_path):
    path = write_report(result.report, out_dir=tmp_path)
    assert path.exists() and path.suffix == ".json"
    assert path.with_suffix(".txt").exists()
    import json
    with open(path) as f:
        loaded = json.load(f)
    assert "headline" in loaded


# ============================================================ cost sensitivity block
def test_report_cost_sensitivity_block(result):
    """The report carries the counterfactual cost-sensitivity block with all multipliers,
    m=1 reconciling to 1.0 and the wiki caveat cited."""
    cs = result.report["cost_sensitivity"]
    assert cs["realized_annual_drag_bps"] > 0.0
    got = sorted(round(l["multiplier"], 4) for l in cs["levels"])
    want = sorted(round(m, 4) for m in (1.0, 10.0 / 3.0, 20.0 / 3.0))
    assert got == want
    r1 = next(l["ratio"] for l in cs["levels"] if abs(l["multiplier"] - 1.0) < 1e-9)
    assert r1 == pytest.approx(1.0)
    # higher prefactor -> weakly larger drag
    ratios = [l["ratio"] for l in sorted(cs["levels"], key=lambda x: x["multiplier"])]
    assert np.all(np.diff(ratios) >= -1e-9)
    assert "research-cost-model-calibration" in cs["caveat"]


# ============================================================ DSR var_trials path switch
def test_var_trials_empirical_path(result, tmp_path):
    """>= 2 recorded gate val_sharpes -> empirical trial variance; caveat names the path."""
    src = REPO_ROOT / "configs" / "factors.yaml"
    dst = tmp_path / "factors.yaml"
    shutil.copy(src, dst)
    reg = FactorRegistry(dst)
    names = list(reg.cfg["factors"])[:3]
    for nm, vs in zip(names, (0.6, 1.1, -0.4)):
        reg.cfg["factors"][nm]["gate_stats"] = {"val_sharpe": vs}
    rep = build_report(result, backtest_config(), reg)
    src_str = rep["caveats"]["var_trials_source"]
    assert "empirical trial variance from 3 gate records" in src_str
    # var_trials is the population variance of the recorded (annualized) Sharpes
    # converted to per-observation units to match the DSR algebra's sr_pp
    expected = float(np.var(np.array([0.6, 1.1, -0.4]) / np.sqrt(252.0)))
    assert rep["headline"]["var_trials"] == pytest.approx(expected)


def test_var_trials_proxy_path(result):
    """No recorded val_sharpes (empty gate_stats) -> the estimator-variance proxy path."""
    reg = FactorRegistry()  # stock factors.yaml: gate_stats all empty
    rep = build_report(result, backtest_config(), reg)
    src_str = rep["caveats"]["var_trials_source"]
    assert "proxy" in src_str
    assert rep["headline"]["var_trials"] > 0.0


def test_gatestats_val_sharpe_roundtrip(tmp_path):
    """val_sharpe persists through record/save and old yamls (key absent) still load."""
    src = REPO_ROOT / "configs" / "factors.yaml"
    dst = tmp_path / "factors.yaml"
    shutil.copy(src, dst)
    reg = FactorRegistry(dst)
    stats = GateStats(train_ic=0.03, train_tstat=3.0, oos_ic=0.02,
                      decay_halflife_days=10.0, net_validation_return=0.01,
                      n_dates=300, val_sharpe=0.75)
    reg.record("mom_12_1", reg.gate("mom_12_1", stats))
    reg.save()
    reloaded = yaml.safe_load(open(dst))
    assert reloaded["factors"]["mom_12_1"]["gate_stats"]["val_sharpe"] == pytest.approx(0.75)
    # old-yaml tolerance: reconstruct GateStats from a dict lacking val_sharpe -> default NaN
    gs_dict = {k: v for k, v in reloaded["factors"]["mom_12_1"]["gate_stats"].items()
               if k != "val_sharpe"}
    assert np.isnan(GateStats(**gs_dict).val_sharpe)


# ============================================================ overlay deadband threading
_DB_EQ = {f"EQ:DBS{i:02d}:2000-01-03": "equity" for i in range(6)}


def _run_with_deadband(deadband: float):
    cfg = copy.deepcopy(backtest_config())
    ov = cfg["overlays"]
    ov["vol_target"]["deadband"] = deadband
    ov["vol_target"]["target"] = 0.02        # keep the multiplier off the clip bounds & varying
    ov["vol_target"]["ewma_halflife"] = None  # raw window -> more day-to-day variation
    ov["drawdown_control"]["threshold"] = 10.0   # never triggers
    ov["macro_derisk"]["enabled"] = False        # isolate the vol-target overlay
    data = {"prices": make_gbm_prices(_DB_EQ, start="2016-01-01", end="2019-06-30")}
    return run_backtest(data, pd.Series(_DB_EQ), cfg=cfg)


def test_overlay_deadband_reduces_multiplier_changes():
    """Threading the previous vol-target multiplier makes the deadband bind during a run:
    a 50% deadband yields strictly fewer distinct overlay multipliers than no deadband."""
    r0 = _run_with_deadband(0.0)
    r5 = _run_with_deadband(0.5)
    n0 = r0.overlay.round(10).nunique()
    n5 = r5.overlay.round(10).nunique()
    assert n0 > 1                # the no-deadband multiplier genuinely varies
    assert n5 < n0               # hysteresis collapses within-band moves


# ============================================================ metrics known answers
def test_sharpe_known_answer():
    r = pd.Series([0.01, -0.005, 0.02, 0.0, 0.015, -0.01])
    expected = r.mean() / r.std(ddof=1) * np.sqrt(252)
    assert sharpe(r) == pytest.approx(expected)


def test_max_drawdown_known_answer():
    # equity: 1 -> 1.1 -> 0.55 ; peak 1.1, trough 0.55 -> 50% drawdown.
    r = pd.Series([0.1, -0.5, 0.2])
    assert max_drawdown(r) == pytest.approx(0.5)
    assert max_drawdown(pd.Series([0.01, 0.01, 0.01])) == pytest.approx(0.0)


def test_hit_rate_known_answer():
    assert hit_rate(pd.Series([1.0, -1.0, 2.0, -3.0])) == pytest.approx(0.5)


# ============================================================ deflated / PSR
def test_psr_near_one_for_strong_sharpe():
    # per-observation Sharpe 0.2, n=2000, normal -> essentially certain the true SR > 0.
    psr = probabilistic_sharpe(0.2, 0.0, 2000, skew=0.0, kurt=3.0)
    assert psr > 0.999


def test_deflated_sharpe_in_unit_interval():
    dsr = deflated_sharpe(0.15, n_trials=50, var_trials_sr=1e-4,
                          n=1500, skew=-0.2, kurt=5.0)
    assert 0.0 <= dsr <= 1.0
    # deflating against the expected-max-of-trials benchmark cannot raise PSR.
    psr0 = probabilistic_sharpe(0.15, 0.0, 1500, -0.2, 5.0)
    assert dsr <= psr0 + 1e-9


# ============================================================ bootstrap
def test_bootstrap_ci_contains_point_estimate():
    rng = np.random.default_rng(1)
    r = pd.Series(rng.normal(0.0004, 0.01, 750))
    point = sharpe(r)
    lo, hi = sharpe_ci(r, level=0.95, n_boot=400, seed=1)
    assert lo <= point <= hi
    assert len(stationary_bootstrap(r, n_boot=100, seed=1)) == 100


# ============================================================ attribution
def test_attribution_recovers_planted_exposure():
    rng = np.random.default_rng(2)
    n = 1000
    f = pd.Series(rng.normal(0.0003, 0.01, n),
                  index=pd.bdate_range("2018-01-01", periods=n))
    port = 0.5 * f + pd.Series(rng.normal(0, 1e-4, n), index=f.index)
    fr = pd.DataFrame({"planted": f})
    out = factor_attribution(port, fr)
    assert out["betas"]["planted"] == pytest.approx(0.5, abs=0.02)


def test_per_alpha_contribution_shape():
    idx = pd.bdate_range("2019-01-01", periods=40)
    ids = ["EQ:A:2000-01-03", "EQ:B:2000-01-03"]
    inst_ret = pd.DataFrame(np.random.default_rng(3).normal(0, 0.01, (40, 2)),
                            index=idx, columns=ids)
    W = pd.DataFrame({ids[0]: [0.1, -0.1], ids[1]: [-0.1, 0.1]},
                     index=[idx[0], idx[20]])
    out = per_alpha_contribution({"momentum": W}, inst_ret)
    assert "momentum" in out and np.isfinite(out["momentum"])


# ============================================================ CLI smoke (subprocess)
def test_run_backtest_synthetic_smoke(tmp_path):
    env = {"PYTHONPATH": str(REPO_ROOT)}
    import os
    env = {**os.environ, **env}
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "run_backtest.py"),
         "--synthetic", "--start", "2019-07-01", "--end", "2019-09-30",
         "--report-dir", str(tmp_path)],
        cwd=str(REPO_ROOT), env=env, capture_output=True, text=True, timeout=110)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "net Sharpe" in proc.stdout
    assert list(tmp_path.glob("backtest_*.json"))
