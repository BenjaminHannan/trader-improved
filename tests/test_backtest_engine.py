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
import warnings

import numpy as np
import pandas as pd
import pytest
import yaml

from production.alpha.registry import FactorRegistry, GateStats

from production.backtest.attribution import factor_attribution, per_alpha_contribution
from production.backtest.bootstrap import (politis_white_block_length, sharpe_ci,
                                           stationary_bootstrap)
from production.backtest.deflated_sharpe import deflated_sharpe, probabilistic_sharpe
from production.backtest.engine import run_backtest


@pytest.fixture(scope="module", autouse=True)
def _pregate_registry(tmp_path_factory):
    """Pin every engine test to an all-candidate registry snapshot.

    These tests exercise ENGINE MECHANICS (tranching, events, determinism) and were
    authored against the pre-gate registry; once the real gate ran (2026-07-07) the
    live registry's accepted set changed which factors the engine trades, making the
    synthetic bundles produce no weights. Hermetic input, identical mechanics.
    """
    import production.backtest.engine as _eng
    from production.core.config import CONFIG_DIR

    cfg = yaml.safe_load(open(CONFIG_DIR / "factors.yaml"))
    for spec in cfg["factors"].values():
        spec["status"] = "candidate"
    path = tmp_path_factory.mktemp("registry") / "factors.yaml"
    yaml.safe_dump(cfg, open(path, "w"), sort_keys=False)
    class _PinnedRegistry(FactorRegistry):   # a real class: engine isinstance()s it
        def __init__(self, cfg_path=None):
            super().__init__(cfg_path if cfg_path is not None else path)

    mp = pytest.MonkeyPatch()   # module-scoped: must outlive the module fixtures
    mp.setattr(_eng, "FactorRegistry", _PinnedRegistry)
    yield
    mp.undo()
from production.core.calendar import offset_grid, rebalance_grid, trading_days
from production.backtest.metrics import (ann_vol, hit_rate, max_drawdown, sharpe,
                                         turnover)
from production.backtest.report import build_report, write_report
from production.core.config import REPO_ROOT, backtest_config
from production.signals.base import sleeve_from_id
from tests.conftest import make_basis, make_funding, make_gbm_prices, make_macro

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


# Deterministic fake sectors for the 8 synthetic equity ids: round-robin over 3 labels.
_EQ_SECTORS = pd.Series(
    {iid: f"sector_{'ABC'[k % 3]}" for k, iid in enumerate(sorted(_EQ))})


@pytest.fixture(scope="module")
def sector_result():
    """One extra equity-only run with sectors threaded (structure assertions only).

    Kept as small as possible: a single sleeve, no funding/macro, and the end pulled in to
    ~6 weeks past the 3y warmup so only a handful of rebalances run (the full ~3y span still
    feeds the risk model, so the sector factors are well-estimated). The module ``result``
    fixture runs WITHOUT sectors and remains the regression guard that the default path is
    unchanged.
    """
    data = {"prices": make_gbm_prices(_EQ, start=_START, end="2020-01-31")}
    return run_backtest(data, pd.Series(_EQ), sectors=_EQ_SECTORS)


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


# ============================================================ sector threading
def test_sector_dummies_present_in_equity_risk_model(sector_result):
    """With sectors threaded, the equity risk model's exposure matrix B carries
    sector_* dummy columns (they were dead code before this wiring)."""
    rm = sector_result.risk_models.get("equity")
    assert rm is not None and rm.B is not None, "equity sleeve did not build a structural model"
    sector_cols = [c for c in rm.B.columns if str(c).startswith("sector_")]
    assert sector_cols, list(rm.B.columns)


def test_sector_band_respected_every_equity_rebalance(sector_result):
    """Per-sector net exposure of each equity rebalance respects the ±sector_band."""
    band = backtest_config()["constraints"]["sector_band"]
    W = sector_result.weights_history["equity"]
    for _, row in W.iterrows():
        for label in _EQ_SECTORS.unique():
            members = [i for i in W.columns if _EQ_SECTORS.get(i) == label]
            net = float(row.reindex(members).fillna(0.0).sum())
            assert abs(net) <= band + 1e-6, (label, net)


def test_sector_run_is_finite(sector_result):
    """The whole sector-threaded run stays finite / green."""
    r = sector_result.total_returns
    assert len(r) > 0
    assert np.isfinite(r.to_numpy()).all() and r.isna().sum() == 0


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
    # The live registry carries real gate records since 2026-07-07; scrub them so the
    # test controls exactly which val_sharpes exist.
    for spec in reg.cfg["factors"].values():
        spec["gate_stats"] = {}
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


def test_var_trials_proxy_path(result, tmp_path):
    """No recorded val_sharpes (empty gate_stats) -> the estimator-variance proxy path."""
    src = REPO_ROOT / "configs" / "factors.yaml"
    dst = tmp_path / "factors.yaml"
    shutil.copy(src, dst)
    reg = FactorRegistry(dst)
    # Live registry has real gate records since 2026-07-07 — scrub to the pre-gate shape
    # this test was authored against (no recorded val_sharpes anywhere).
    for spec in reg.cfg["factors"].values():
        spec["gate_stats"] = {}
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


def test_sharpe_nan_on_solver_noise_floor():
    """2026-07-11 incident: a sleeve the optimizer correctly decided NOT to trade (alpha
    too weak to clear the cost floor) can still carry ~1e-8-1e-9 scale non-zero weights
    from CVXPY solver tolerance on the "hold zero" corner solution. Those phantom weights,
    marked against real market returns, produce a daily-return series whose mean AND std
    both round to ~0 -- but whose ratio does not, because Sharpe is scale-invariant. Before
    the fix this reported a "plausible" nonzero Sharpe next to ann_vol/ann_return/
    max_drawdown that correctly read ~0 (production/backtest/metrics.py ann_vol/ann_return),
    an internally contradictory report (nonzero Sharpe requires nonzero vol). A std this
    small cannot arise from real trading: CLAUDE.md's cost floors are >= 5bp (equities) /
    >= 30bp (crypto), so any position the optimizer genuinely holds moves the book by many
    orders of magnitude more than this on the day it is marked.
    """
    rng = np.random.default_rng(3)
    noise = pd.Series(rng.normal(0.0, 1.0, 300)) * 1e-9  # solver-tolerance-scale residue
    assert np.isnan(sharpe(noise))

    # An exactly-constant-zero series (the pre-solver-noise, "really did not trade" case)
    # must keep behaving exactly as before: also NaN, never a divide-by-zero explosion.
    assert np.isnan(sharpe(pd.Series(0.0, index=range(300))))

    # Guardrail: the fix must not swallow real, legitimately low-vol strategies. CLAUDE.md's
    # cost floors put a genuine trade's daily return contribution many orders of magnitude
    # above the noise floor, so a series at that realistic scale must still score normally.
    real = pd.Series(rng.normal(0.0002, 0.003, 300))  # ~30bp/day vol, a real quiet sleeve
    assert np.isfinite(sharpe(real))


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


# ============================================================ Politis-White block length
def _ar1(rho: float, n: int, seed: int) -> pd.Series:
    rng = np.random.default_rng(seed)
    eps = rng.normal(0.0, 1.0, n)
    x = np.empty(n)
    x[0] = eps[0]
    for t in range(1, n):
        x[t] = rho * x[t - 1] + eps[t]
    return pd.Series(x)


def test_ppw_iid_gives_short_block():
    """iid N(0,1), T=1000: no serial dependence -> a short expected block (< 10)."""
    rng = np.random.default_rng(7)
    r = pd.Series(rng.normal(0.0, 1.0, 1000))
    b = politis_white_block_length(r)
    assert 1.0 <= b < 10.0


def test_ppw_ar1_much_longer_than_iid():
    """AR(1) rho=0.9 at the same T is substantially more dependent -> >= 3x the iid block."""
    rng = np.random.default_rng(7)
    iid = pd.Series(rng.normal(0.0, 1.0, 1000))
    b_iid = politis_white_block_length(iid)
    b_ar1 = politis_white_block_length(_ar1(0.9, 1000, seed=11))
    assert b_ar1 >= 3.0 * b_iid


def test_ppw_block_within_bounds_short_and_constant():
    """b in [1, ceil(T/3)] on a short series; a constant series degenerates to 1.0."""
    short = _ar1(0.5, 30, seed=3)
    b = politis_white_block_length(short)
    assert 1.0 <= b <= np.ceil(len(short) / 3.0)
    assert politis_white_block_length(pd.Series([0.01] * 200)) == 1.0
    assert politis_white_block_length(pd.Series([0.0, 0.0])) == 1.0


def test_sharpe_ci_auto_brackets_true_sharpe():
    """'auto' block selection still yields a CI that brackets a seeded generator's Sharpe."""
    rng = np.random.default_rng(21)
    r = pd.Series(rng.normal(0.0005, 0.01, 1000))
    point = sharpe(r)
    lo, hi = sharpe_ci(r, level=0.95, n_boot=400, avg_block="auto", seed=1)
    assert lo <= point <= hi


def test_sharpe_ci_numeric_block_bit_identical():
    """A numeric avg_block reproduces the pre-'auto' fixed-block CI exactly (same seed)."""
    rng = np.random.default_rng(5)
    r = pd.Series(rng.normal(0.0004, 0.01, 750))
    a = stationary_bootstrap(r, n_boot=200, avg_block=21, seed=1)
    b = stationary_bootstrap(r, n_boot=200, avg_block=21, seed=1)
    assert np.array_equal(a, b)
    ci_a = sharpe_ci(r, n_boot=200, avg_block=21, seed=1)
    ci_b = sharpe_ci(r, n_boot=200, avg_block=21, seed=1)
    assert ci_a == ci_b
    # and 'auto' generally selects a different block than the old fixed 21 here.
    assert politis_white_block_length(r) != 21.0


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


# ============================================================ tranching + per-sleeve cadence
def _cfg(n_tranches=None, rebalance="weekly", drop_ntranches=False):
    """A backtest cfg with the tranching / cadence knobs set (short deep-copied override)."""
    cfg = copy.deepcopy(backtest_config())
    cfg["walk_forward"]["rebalance"] = rebalance
    if drop_ntranches:
        cfg["walk_forward"].pop("n_tranches", None)
    elif n_tranches is not None:
        cfg["walk_forward"]["n_tranches"] = n_tranches
    return cfg


def _daily_book(W: pd.DataFrame, common_idx: pd.DatetimeIndex) -> pd.DataFrame:
    """Averaged (event-resolution) book reindexed onto a common daily grid, held constant
    between events — the object whose day-over-day L1 change is the traded turnover."""
    return W.reindex(common_idx, method="ffill").fillna(0.0)


def test_offset_grid_next_trading_day_known_answer():
    """offset_grid(·, 1) maps every grid date to the next NYSE trading day; 0 is identity."""
    grid = rebalance_grid("2019-01-01", "2019-06-30", "weekly")
    days = trading_days("nyse", "2019-01-01", "2019-08-31")
    o1 = offset_grid(grid, 1)
    assert len(o1) == len(grid)
    for gi, oi in zip(grid, o1):
        pos = int(days.searchsorted(gi))
        assert oi == days[pos + 1]
    # offset 2 = two trading days forward; offset 0 = bit-identical passthrough.
    o2 = offset_grid(grid, 2)
    for gi, oi in zip(grid, o2):
        assert oi == days[int(days.searchsorted(gi)) + 2]
    assert offset_grid(grid, 0).equals(grid)


def test_ntranches_one_matches_pre_change_semantics():
    """n_tranches=1 reproduces the pre-tranching single-grid behavior bit-for-bit, and the
    engine is deterministic: n_tranches=1 (twice) and n_tranches-absent all agree exactly."""
    data = {"prices": make_gbm_prices(_EQ, start="2016-12-15", end="2020-03-31")}
    inst = pd.Series(_EQ)
    r1a = run_backtest(data, inst, cfg=_cfg(n_tranches=1))
    r1b = run_backtest(data, inst, cfg=_cfg(n_tranches=1))
    r_absent = run_backtest(data, inst, cfg=_cfg(drop_ntranches=True))
    # determinism: same cfg twice -> identical net returns
    assert r1a.total_returns.index.equals(r1b.total_returns.index)
    assert np.array_equal(r1a.total_returns.to_numpy(), r1b.total_returns.to_numpy())
    # pre-change equivalence: n_tranches absent behaves exactly like n_tranches=1
    assert r1a.total_returns.index.equals(r_absent.total_returns.index)
    assert np.allclose(r1a.total_returns.to_numpy(), r_absent.total_returns.to_numpy(),
                       atol=0.0, rtol=0.0)


@pytest.fixture(scope="module")
def tranche_eq():
    """Equity-only K=1 vs K=3 over the module window (weekly, so K is the only difference)."""
    data = {"prices": make_gbm_prices(_EQ, start=_START, end=_END)}
    inst = pd.Series(_EQ)
    return (run_backtest(data, inst, cfg=_cfg(n_tranches=1)),
            run_backtest(data, inst, cfg=_cfg(n_tranches=3)))


def test_tranching_lowers_traded_turnover_at_equal_gross(tranche_eq):
    """K=3 averages three offset sub-books: the traded book's day-over-day turnover is
    strictly lower than K=1 (the moving-average smoothing), at the same average gross."""
    r1, r3 = tranche_eq
    W1, W3 = r1.avg_book_weights["equity"], r3.avg_book_weights["equity"]
    idx = pd.date_range(min(W1.index.min(), W3.index.min()),
                        max(W1.index.max(), W3.index.max()), freq="D")
    D1, D3 = _daily_book(W1, idx), _daily_book(W3, idx)
    burn = len(idx) // 3          # drop the position-building transient (K>1 ramps in over K days)
    tv1 = float(D1.iloc[burn:].diff().abs().sum().sum())
    tv3 = float(D3.iloc[burn:].diff().abs().sum().sum())
    assert tv1 > 0.0              # the comparison is non-vacuous (the book actually trades)
    assert tv3 < tv1             # tranching strictly reduces traded turnover
    g1 = float(D1.iloc[burn:].abs().sum(axis=1).mean())
    g3 = float(D3.iloc[burn:].abs().sum(axis=1).mean())
    assert abs(g3 / g1 - 1.0) < 0.10   # same average exposure (within 10%)


def test_twice_weekly_crypto_trades_more_dates_than_weekly_equity():
    """Per-sleeve cadence: crypto on the twice_weekly grid rebalances on ~2x as many dates
    as the weekly equity sleeve in the same run."""
    inst = {**_EQ, **_CR}
    crypto = [i for i, s in inst.items() if s == "crypto"]
    data = {"prices": make_gbm_prices(inst, start="2016-12-15", end="2020-03-31"),
            "funding": make_funding(crypto, start="2016-12-15", end="2020-03-31")}
    cfg = _cfg(n_tranches=1, rebalance={"default": "weekly", "crypto": "twice_weekly"})
    r = run_backtest(data, pd.Series(inst), cfg=cfg)
    eq_dates = len(r.weights_history["equity"].index)
    cr_dates = len(r.weights_history["crypto"].index)
    assert eq_dates > 0 and cr_dates > 0
    assert cr_dates > 1.5 * eq_dates


def test_default_config_ships_tranching_and_crypto_cadence_on():
    """The shipped default drives the PIT money test (test_pit_no_lookahead) through the new
    machinery: n_tranches>1 and a per-sleeve crypto override are ON in the default cfg the
    module `result` fixture uses."""
    wf = backtest_config()["walk_forward"]
    assert int(wf["n_tranches"]) > 1
    assert isinstance(wf["rebalance"], dict)
    assert wf["rebalance"].get("crypto") == "twice_weekly"


# ============================================================ events sleeve integration
from production.events.backtest import event_sleeve_returns
from production.events.sizing import size_event_book, taker_fee_fraction
from production.events.markets import dedupe_related, liquid_universe
from production.events.backtest import _combined_signals
from production.portfolio.allocation import (_ewma_cov, _risk_contrib_shares, erc_weights,
                                              sleeve_allocation)
from production.core.calendar import rebalance_grid


def _event_row_cols():
    return ["obs_date", "instrument_id", "yes_price", "volume", "open_interest",
            "close_time", "status", "event_key"]


def _finish_event_frame(rows) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=_event_row_cols())
    df["obs_date"] = pd.to_datetime(df["obs_date"])
    df["close_time"] = pd.to_datetime(df["close_time"], utc=True)
    df["available_from"] = df["obs_date"].dt.tz_localize("UTC") + pd.Timedelta(hours=1)
    df["question"] = df["instrument_id"]
    df["venue"] = "kalshi"
    df["source"] = "test"
    df["ingested_at"] = df["available_from"]
    return df.sort_values(["obs_date", "instrument_id"]).reset_index(drop=True)


def _favorite_known_answer_frame(start="2020-01-01"):
    """Two markets: a favorite drifting 0.90 -> 0.98 (settles to 1.0), and a flat 0.50 market
    that never signals but extends the grid so the favorite resolves in-sample. Only one
    rebalance (day 0) is used, so the favorite's whole life is a single held position."""
    d0 = pd.Timestamp(start)
    # favorite: 8 daily obs 0.90..0.98, closes shortly after its last obs.
    fav_prices = [0.90, 0.91, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98]
    fav_dates = pd.date_range(d0, periods=len(fav_prices), freq="D")
    fav_close = fav_dates[-1] + pd.Timedelta(days=2)
    rows = [(d, "EV:kalshi:FAV", p, 5000.0, 1000.0, fav_close, "active", "EVT-FAV")
            for d, p in zip(fav_dates, fav_prices)]
    # flat market lives longer (extends the grid past the favorite's last obs -> it settles).
    flat_dates = pd.date_range(d0, periods=len(fav_prices) + 5, freq="D")
    flat_close = flat_dates[-1] + pd.Timedelta(days=40)
    rows += [(d, "EV:kalshi:FLAT", 0.50, 5000.0, 1000.0, flat_close, "active", "EVT-FLAT")
             for d in flat_dates]
    # rebalance one day after the first obs so that first obs (avail = obs+1h) is knowable and
    # the entered price is exactly 0.90 (day-0 print), the only row available at t0.
    return _finish_event_frame(rows), fav_dates[0] + pd.Timedelta(days=1)


def test_event_sleeve_known_answer_long_favorite_resolves():
    """A long-YES favorite entered at 0.90 and settling at 1.0 contributes
    weight*(1/0.9 - 1) of P&L minus the 200bp entry fee; the flat market adds nothing."""
    frame, t0 = _favorite_known_answer_frame()
    out = event_sleeve_returns(frame, [t0])
    # reconstruct the sized weight the sleeve would enter at t0 (PIT book).
    avail = frame[pd.to_datetime(frame["available_from"], utc=True)
                  <= pd.Timestamp(t0).tz_localize("UTC")]
    universe = liquid_universe(avail)
    book = size_event_book(_combined_signals(avail), avail, capital_frac=0.10,
                           groups=dedupe_related(universe))
    assert list(book["instrument_id"]) == ["EV:kalshi:FAV"]
    assert book.iloc[0]["side"] == "YES"
    w = float(book.iloc[0]["weight"])
    # fee: the accurate per-price taker fraction at the 0.90 entry (the flat
    # FEE_BPS haircut was retired in the 2026-07-11 events fee re-spec)
    expected = w * (1.0 / 0.90 - 1.0) - w * taker_fee_fraction(0.90)
    assert out.sum() == pytest.approx(expected, rel=1e-9, abs=1e-12)


# --- richer 8-market frame for PIT + engine integration ---
def _event_markets_frame(start="2019-11-01", end="2020-06-30"):
    """8 markets in 2 dedupe groups (EVT-A/EVT-B), staggered ~60-day lives, prices drifting
    toward 0 or 1 (so late-life longshot/convergence signals fire and resolutions realize)."""
    day0 = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    rng = np.random.default_rng(0)
    rows = []
    for k in range(8):
        life_start = day0 + pd.Timedelta(days=k * 22)
        mdates = pd.date_range(life_start, periods=60, freq="D")
        mdates = mdates[mdates <= end_ts]
        if len(mdates) < 8:
            continue
        target = 0.975 if k % 2 == 0 else 0.025
        n = len(mdates)
        drift = np.linspace(0.5, target, n)
        prices = np.clip(drift + rng.normal(0, 0.01, n), 0.02, 0.98)
        close = mdates[-1] + pd.Timedelta(days=1)
        iid = f"EV:kalshi:M{k}"
        ekey = "EVT-A" if k < 4 else "EVT-B"
        rows += [(d, iid, float(p), 5000.0, 1000.0, close, "active", ekey)
                 for d, p in zip(mdates, prices)]
    return _finish_event_frame(rows)


def test_event_sleeve_pit_future_corruption_leaves_history():
    """Corrupting every event price dated after a pivot leaves the sleeve's daily returns up
    to the pivot bit-for-bit unchanged (PIT money test for the event sleeve)."""
    frame = _event_markets_frame()
    grid = rebalance_grid(frame["obs_date"].min(), frame["obs_date"].max(), "weekly")
    base = event_sleeve_returns(frame, grid, per_group_cap=0.02)
    pivot = base.index[int(len(base) * 0.6)]

    corrupt = frame.copy()
    future = corrupt["obs_date"] > pivot
    corrupt.loc[future, "yes_price"] = 0.99   # slam the future tail
    after = event_sleeve_returns(corrupt, grid, per_group_cap=0.02)

    a = base[base.index <= pivot]
    b = after[after.index <= pivot]
    assert a.index.equals(b.index)
    assert np.allclose(a.to_numpy(), b.to_numpy(), atol=1e-12, rtol=0.0)
    assert np.abs(base[base.index > pivot].to_numpy()).sum() > 0   # non-vacuous


def _events_bundle():
    b = _bundle()
    b["event_markets"] = _event_markets_frame()
    return b


@pytest.fixture(scope="module")
def events_result():
    return run_backtest(_events_bundle(), pd.Series(_INSTRUMENTS))


def test_events_sleeve_present_and_allocated(events_result):
    """With an event_markets panel the engine adds an 'events' return-stream sleeve that draws
    positive allocation, and the 0.10 events risk cap BINDS in the allocation path: at the
    monthly points where events is warm its capped risk-contribution share is pulled well below
    the uncapped (equal-ERC) share and is the smallest of any sleeve — the per-sleeve key bites.

    (With only equity/crypto/events present the three caps sum to 0.70 < 1, so risk shares —
    which must sum to 1 — cannot all sit under their caps simultaneously; the strict <=0.10 is
    a many-sleeve property. The apply_risk_caps unit test pins the exact-cap case directly.)"""
    sr = events_result.sleeve_returns
    assert "events" in sr.columns
    assert np.isfinite(sr["events"].to_numpy()).all()

    cfg = backtest_config()
    cfg_nocap = copy.deepcopy(cfg)
    cfg_nocap["allocation"].pop("risk_caps", None)
    hl = float(cfg["allocation"]["sleeve_cov_halflife_days"])
    warmup = int(cfg["allocation"]["warmup_days"])
    months = pd.DatetimeIndex(sorted({pd.Timestamp(t.year, t.month, 1) for t in sr.index}))
    saw_positive = False
    checked_cap = False
    for t in months:
        w = sleeve_allocation(sr, t, cfg)
        if w.get("events", 0.0) > 0.0:
            saw_positive = True
        r = sr[sr.index < t]
        counts = r.notna().sum()
        warm = [s for s in sr.columns if counts.get(s, 0) >= warmup]
        if "events" not in warm or len(warm) < 2:
            continue
        cov = _ewma_cov(r[warm], hl).to_numpy()
        sh = _risk_contrib_shares(w.reindex(warm).fillna(0.0).to_numpy(), cov)
        wn = sleeve_allocation(sr, t, cfg_nocap)
        shn = _risk_contrib_shares(wn.reindex(warm).fillna(0.0).to_numpy(), cov)
        ei = warm.index("events")
        if shn[ei] > 0.10:                       # uncapped events would exceed its cap
            checked_cap = True
            assert sh[ei] < shn[ei] - 1e-3, (t, sh[ei], shn[ei])   # the cap strictly reduces it
            assert sh[ei] == pytest.approx(min(sh), abs=1e-9), (t, sh)  # events is the smallest
    assert saw_positive
    assert checked_cap


def test_events_absent_key_bit_identical(events_result):
    """Absent the event_markets key the run is bit-identical to a twin no-events run, and the
    events run genuinely differs (the extra sleeve moves the book)."""
    inst = pd.Series(_INSTRUMENTS)
    r_base_a = run_backtest(_bundle(), inst)
    r_base_b = run_backtest(_bundle(), inst)
    assert r_base_a.total_returns.index.equals(r_base_b.total_returns.index)
    assert np.array_equal(r_base_a.total_returns.to_numpy(),
                          r_base_b.total_returns.to_numpy())
    assert "events" not in r_base_a.sleeve_returns.columns
    # the events run shares the index but the total book is not identical (events sleeve bites).
    ev = events_result.total_returns.reindex(r_base_a.total_returns.index)
    assert not np.allclose(ev.to_numpy(), r_base_a.total_returns.to_numpy(), atol=1e-12)


# ================================================== cost overrides wiring (backlog #14)
def test_run_backtest_reads_lake_cost_overrides(monkeypatch, tmp_path_factory):
    """The engine threads the lake's TCA-calibrated cost_overrides table into the CostModel
    construction — previously a dead ``lake`` parameter (backlog #14: TCA calibration wrote
    the table, run_backtest never read it back). A CostModel spy pins the EXACT
    overrides_table argument the engine passes (the narrowest seam for the previously-dead
    wiring), not just an indirect P&L difference. A loud warning must accompany an active
    override — a backtest run must never silently change cost regime (CLAUDE.md)."""
    from production.core.lake import Lake
    from production.execution.tca import write_overrides
    import production.backtest.engine as _eng

    lake = Lake(tmp_path_factory.mktemp("cost_overrides_active"))
    calibrated_id = sorted(_EQ)[0]
    override = {calibrated_id: {"half_spread_bps": 50.0, "n_fills": 30,
                                "median_abs_shortfall_bps": 40.0}}
    write_overrides(override, lake)

    captured = {}
    real_cls = _eng.CostModel

    class _SpyCostModel(real_cls):
        def __init__(self, cfg=None, overrides_table=None):
            captured["overrides_table"] = overrides_table
            super().__init__(cfg, overrides_table=overrides_table)

    monkeypatch.setattr(_eng, "CostModel", _SpyCostModel)

    data = {"prices": make_gbm_prices(_EQ, start=_START, end="2020-01-31")}
    with pytest.warns(UserWarning, match="cost overrides ACTIVE"):
        result = run_backtest(data, pd.Series(_EQ), lake=lake)

    assert captured["overrides_table"] == override
    assert any("cost overrides ACTIVE" in w for w in result._warnings)
    assert any("1 instrument" in w for w in result._warnings)


def test_run_backtest_lake_without_cost_overrides_table_unchanged(monkeypatch,
                                                                   tmp_path_factory):
    """A lake that was never TCA-calibrated (no cost_overrides table written) must leave the
    engine's CostModel construction — and therefore the run — bit-identical to lake=None."""
    from production.core.lake import Lake
    import production.backtest.engine as _eng

    empty_lake = Lake(tmp_path_factory.mktemp("cost_overrides_absent"))
    captured: list = []
    real_cls = _eng.CostModel

    class _SpyCostModel(real_cls):
        def __init__(self, cfg=None, overrides_table=None):
            captured.append(overrides_table)
            super().__init__(cfg, overrides_table=overrides_table)

    monkeypatch.setattr(_eng, "CostModel", _SpyCostModel)

    data = {"prices": make_gbm_prices(_EQ, start=_START, end="2020-01-31")}
    result_lake = run_backtest(data, pd.Series(_EQ), lake=empty_lake)
    result_none = run_backtest(data, pd.Series(_EQ), lake=None)

    assert captured == [None, None]                  # never received a table either time
    assert not any("cost overrides ACTIVE" in w for w in result_lake._warnings)
    assert result_lake.total_returns.equals(result_none.total_returns)


# ============================================================ report consistency (2026-07-11 incident)
@pytest.fixture(scope="module")
def _degenerate_sleeve_result(tmp_path_factory):
    """End-to-end reproduction of the 2026-07-11 incident report's exact shape: one sleeve
    (crypto, via ``basis_carry``) whose only accepted factor's alpha never clears its cost
    floor -- an economically-correct "never trade" decision -- next to a second sleeve
    (fx_etf, via ``carry_rate_diff``) that trades normally. This is the real signal classes
    (not a hand-rolled fixture), a real (temp, 2-factor) registry, and the real engine walk;
    it is what actually produced the incident's per_sleeve.crypto (nonzero Sharpe, ~0
    ann_vol/ann_return/turnover) next to a healthy per_sleeve.fx_etf. Kept as small as the
    mechanism allows (2 sleeves, 5 names each -- the rank_ic min_names floor --, 3 months of
    trading) to bound runtime; still takes ~20s, the cost of exercising the real optimizer/
    solver-tolerance path rather than a synthetic return series.
    """
    start, end = "2019-01-01", "2019-03-31"
    data_start = (pd.Timestamp(start) - pd.DateOffset(years=3, months=3)).strftime("%Y-%m-%d")
    data_end = pd.Timestamp(end).strftime("%Y-%m-%d")

    cr = {f"CR:{s}:2017-01-01": "crypto" for s in ["BTC", "ETH", "SOL", "LTC", "XRP"]}
    fx = {f"FX:{s}:2007-01-03": "fx_etf" for s in ["FXE", "FXY", "FXB", "FXA", "FXC"]}
    instruments = {**cr, **fx}
    crypto_ids = list(cr)

    data = {
        "prices": make_gbm_prices(instruments, start=data_start, end=data_end),
        "funding": make_funding(crypto_ids, start=data_start, end=data_end),
        # RATE_AU/CA/CH cover FXA/FXC/FXF-style currencies so all 5 fx names clear
        # rank_ic's min_names=5 cross-sectional floor (production/alpha/ic.py).
        "macro": make_macro({"DGS3MO_US": 2.0, "RATE_EU": 0.5, "RATE_JP": -0.1,
                             "RATE_GB": 1.0, "RATE_AU": 1.5, "RATE_CA": 1.2, "RATE_CH": -0.5},
                            start=data_start, end=data_end),
        "basis": make_basis(crypto_ids, start=data_start, end=data_end),
    }

    # A minimal, self-contained 2-factor registry (NOT configs/factors.yaml -- decoupled
    # from the live gate's current accepted set, which can change) mirroring the real
    # incident's factor/sleeve pairing: one factor per sleeve, both accepted.
    factors_cfg = {
        "factors": {
            "carry_rate_diff": {"signal": "production.signals.carry.RateDifferentialCarry",
                                "sleeves": ["fx_etf"], "horizon_days": 21,
                                "min_history_days": 21, "status": "accepted"},
            "basis_carry": {"signal": "production.signals.carry.BasisCarry",
                           "sleeves": ["crypto"], "horizon_days": 5,
                           "min_history_days": 7, "status": "accepted"},
        },
        "n_trials": 2,
    }
    reg_path = tmp_path_factory.mktemp("degenerate_registry") / "factors.yaml"
    yaml.safe_dump(factors_cfg, open(reg_path, "w"), sort_keys=False)

    return run_backtest(data, pd.Series(instruments), factors_cfg=str(reg_path))


def test_degenerate_sleeve_turnover_is_near_zero(_degenerate_sleeve_result):
    """Sanity check on the mechanism itself: crypto's basis_carry alpha never clears the
    cost floor, so the optimizer's (correct) decision is to hold ~no position -- turnover
    stays at solver-tolerance scale, nowhere near a real trade."""
    ps = _degenerate_sleeve_result.report["per_sleeve"]
    assert "crypto" in ps and "fx_etf" in ps
    assert ps["crypto"]["turnover"] < 1e-4
    assert ps["fx_etf"]["turnover"] > 1e-3        # the healthy control sleeve trades for real


def test_degenerate_sleeve_sharpe_matches_ann_vol_not_solver_noise(_degenerate_sleeve_result):
    """The incident's core complaint, reproduced and fixed: a sleeve/headline series that
    never really traded must not report a "plausible" Sharpe next to ~0 ann_vol/ann_return
    -- that pairing is impossible for a real return series (nonzero Sharpe requires nonzero
    vol) and is exactly what production/backtest/metrics.py:sharpe's degenerate-std guard
    now prevents."""
    report = _degenerate_sleeve_result.report
    h, ps = report["headline"], report["per_sleeve"]

    # The dead sleeve: Sharpe must be NaN, consistent with its ~0 turnover/ann_vol/ann_return.
    assert np.isnan(ps["crypto"]["sharpe"])
    assert abs(ps["crypto"]["ann_vol"]) < 1e-4
    assert abs(ps["crypto"]["ann_return"]) < 1e-4

    # The healthy control sleeve: a real Sharpe on real vol.
    assert np.isfinite(ps["fx_etf"]["sharpe"])
    assert ps["fx_etf"]["ann_vol"] > 1e-3

    # Headline: whatever the blended total's realized vol is, Sharpe is finite if and only
    # if ann_vol/ann_return are meaningfully (not just technically) nonzero too -- no more
    # of the incident's "net_sharpe=0.18, ann_vol_net=1.6e-09" contradiction.
    if np.isfinite(h["net_sharpe"]):
        assert h["ann_vol_net"] > 1e-4
        assert h["ann_return_net"] != 0.0
    else:
        assert h["ann_vol_net"] < 1e-4


# ============================= allocation: dead-sleeve ERC exclusion (2026-07-11 incident)
def test_degenerate_sleeve_allocation_excludes_dead_sleeve_from_erc(_degenerate_sleeve_result):
    """The 2026-07-11 incident's actual capital-misallocation bug, at the allocation layer:
    the unfixed erc_weights fixed point treated crypto's ~0 variance (a correct "never trade"
    decision, cost floor never cleared) as ~0 risk and levered it to 99.994% of the book next
    to fx_etf's 0.006%. With the fix, at every rebalance date where both sleeves have cleared
    warmup_days (so the ERC/warm code path -- not the cold inverse-vol fallback -- is what's
    under test), crypto must get ~0 and the healthy fx_etf control ~1, and the exclusion must
    warn loudly rather than silently clip."""
    sr = _degenerate_sleeve_result.sleeve_returns
    cfg = backtest_config()
    warmup = int(cfg["allocation"]["warmup_days"])
    months = pd.DatetimeIndex(sorted({pd.Timestamp(t.year, t.month, 1) for t in sr.index}))
    checked = False
    for t in months:
        r = sr[sr.index < t]
        counts = r.notna().sum()
        warm = [s for s in sr.columns if counts.get(s, 0) >= warmup]
        if "crypto" not in warm or "fx_etf" not in warm or len(warm) < 2:
            continue
        checked = True
        with pytest.warns(UserWarning, match="degenerate"):
            w = sleeve_allocation(sr, t, cfg)
        assert w["crypto"] == pytest.approx(0.0, abs=1e-9)
        assert w["fx_etf"] == pytest.approx(1.0, abs=1e-9)
    assert checked, "fixture never reaches a rebalance date where both sleeves are warm"


def test_erc_weights_all_degenerate_falls_back_to_equal_weight_with_warning():
    """If every sleeve in the ERC block is variance-degenerate (the pathological case where
    every sleeve's optimizer refuses to trade) there is no live risk left to equalize --
    erc_weights must fall back to equal weight across the block rather than dividing
    near-zero by near-zero, and must warn loudly rather than silently clipping."""
    idx = ["crypto", "events"]
    cov = pd.DataFrame(np.diag([1e-14, 1e-15]), index=idx, columns=idx)
    with pytest.warns(UserWarning, match="degenerate floor"):
        w = erc_weights(cov)
    assert w.to_numpy() == pytest.approx([0.5, 0.5])
    assert w.sum() == pytest.approx(1.0)


def test_erc_weights_normal_two_sleeve_case_unchanged():
    """Regression guard: with two genuinely healthy (non-degenerate) sleeve variances, the
    dead-sleeve exclusion must never fire, and erc_weights must reproduce the exact analytic
    diagonal-cov result (inverse-vol weights) exactly as it did before the fix."""
    idx = ["equity", "crypto"]
    # Realistic daily vols: 1.5% equity, 4% crypto -- both many orders of magnitude above the
    # 1e-8 degenerate-variance floor, so this must behave identically to the pre-fix code.
    vols = np.array([0.015, 0.04])
    cov = pd.DataFrame(np.diag(vols ** 2), index=idx, columns=idx)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning here would mean the exclusion mis-fired
        w = erc_weights(cov)
    assert w["equity"] / w["crypto"] == pytest.approx(vols[1] / vols[0], rel=1e-4)
    assert w.sum() == pytest.approx(1.0)


# ============================================================ CLI smoke (subprocess)
def test_run_backtest_synthetic_smoke(tmp_path):
    env = {"PYTHONPATH": str(REPO_ROOT)}
    import os
    env = {**os.environ, **env}
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "run_backtest.py"),
         "--synthetic", "--start", "2019-07-01", "--end", "2019-09-30",
         "--report-dir", str(tmp_path)],
        # Hang guard, not a perf SLA: the engine takes ~100s on the reference box, and
        # the 2026-07-07 live-ingest session showed a 110s budget flakes under any
        # concurrent load (ingest jobs, suite parallelism). Functional asserts below
        # are unchanged; an actual hang still trips this.
        cwd=str(REPO_ROOT), env=env, capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "net Sharpe" in proc.stdout
    assert list(tmp_path.glob("backtest_*.json"))
