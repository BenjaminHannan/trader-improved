"""Optimizer / constraints / overlays / allocation tests.

Uses a small deterministic duck-typed stub risk model (8 names, 2 factors) rather than
importing production.risk — the optimizer only needs .ids/.B/.F/.D + covariance(),
portfolio_vol(), factor_form().
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from production.core.config import backtest_config
from production.portfolio.optimizer import optimize_sleeve
from production.portfolio.constraints import build_constraints, check_constraints
from production.portfolio.overlays import (
    vol_target_multiplier, drawdown_multiplier, macro_derisk_multiplier,
)
from production.portfolio.allocation import (
    erc_weights, apply_risk_caps, sleeve_allocation,
)

IDS = [f"EQ:SYN{i:02d}:2000-01-03" for i in range(8)]


class StubRisk:
    """Minimal RiskModel-like object: structural factor form B F B' + diag(D)."""

    def __init__(self, ids, B, F, D, use_factor=True):
        self.ids = pd.Index(ids)
        self.B = pd.DataFrame(np.asarray(B, dtype=float), index=self.ids)
        self.F = pd.DataFrame(np.asarray(F, dtype=float))
        self.D = pd.Series(np.asarray(D, dtype=float), index=self.ids)
        self.use_factor = use_factor

    def covariance(self):
        Sig = self.B.to_numpy() @ self.F.to_numpy() @ self.B.to_numpy().T + np.diag(self.D.to_numpy())
        return pd.DataFrame(Sig, index=self.ids, columns=self.ids)

    def portfolio_vol(self, w):
        wv = pd.Series(w).reindex(self.ids).fillna(0.0).to_numpy(dtype=float)
        return float(np.sqrt(252.0 * wv @ self.covariance().to_numpy() @ wv))

    def factor_form(self):
        return (self.B, self.F, self.D) if self.use_factor else None


def make_stub(use_factor=True, seed=0):
    rng = np.random.default_rng(seed)
    B = rng.normal(0.0, 0.5, (8, 2))
    F = np.diag([4e-4, 2e-4])
    D = rng.uniform(1e-4, 4e-4, 8)
    return StubRisk(IDS, B, F, D, use_factor=use_factor)


def base_cfg():
    return copy.deepcopy(backtest_config())


def loose_cfg():
    """Effectively unconstrained: huge caps, no turnover cap."""
    cfg = base_cfg()
    cc = cfg["constraints"]
    cc["position_cap"] = {k: 100.0 for k in cc["position_cap"]}
    cc["gross_cap"] = 1e6
    cc["net_band"] = 1e6
    cc["turnover_cap"] = None
    return cfg


# ------------------------------------------------------------------ optimizer
def test_zero_alpha_zero_positions():
    """With zero alpha, positive costs, and flat w_prev, the optimum is ~flat."""
    rm = make_stub()
    cfg = loose_cfg()
    alpha = pd.Series(0.0, index=IDS)
    cost = pd.Series(10.0, index=IDS)
    res = optimize_sleeve(alpha, rm, w_prev=pd.Series(0.0, index=IDS),
                          cost_bps=cost, sleeve="equity", cfg=cfg, lam=10.0)
    assert np.all(np.abs(res.w.to_numpy()) < 1e-4)


def test_constraint_satisfaction_aggressive_alpha():
    rm = make_stub(seed=1)
    cfg = base_cfg()
    rng = np.random.default_rng(2)
    alpha = pd.Series(rng.normal(0, 1.0, 8), index=IDS)
    cost = pd.Series(5.0, index=IDS)
    res = optimize_sleeve(alpha, rm, w_prev=pd.Series(0.0, index=IDS),
                          cost_bps=cost, sleeve="equity", cfg=cfg, lam=1.0)
    checks = check_constraints(res.w, w_prev=pd.Series(0.0, index=IDS),
                               sleeve="equity", cfg=cfg)
    assert all(checks.values()), checks


def test_turnover_cap_binds():
    """Flip vs w_prev with huge alpha; total traded ~ turnover_cap."""
    rm = make_stub(seed=3)
    cfg = base_cfg()
    cfg["constraints"]["position_cap"] = {k: 0.10 for k in cfg["constraints"]["position_cap"]}
    cfg["constraints"]["net_band"] = 1e6
    cfg["constraints"]["gross_cap"] = 1e6
    tcap = cfg["constraints"]["turnover_cap"]
    alpha = pd.Series(50.0 * np.array([1, -1, 1, -1, 1, -1, 1, -1]), index=IDS)
    cost = pd.Series(1.0, index=IDS)
    res = optimize_sleeve(alpha, rm, w_prev=pd.Series(0.0, index=IDS),
                          cost_bps=cost, sleeve="equity", cfg=cfg, lam=1e-3)
    traded = float(np.sum(np.abs(res.w.to_numpy())))
    assert traded == pytest.approx(tcap, abs=1e-3)


def test_higher_lambda_lower_risk():
    rm = make_stub(seed=4)
    cfg = loose_cfg()
    alpha = pd.Series(np.random.default_rng(5).normal(0, 1, 8), index=IDS)
    cost = pd.Series(5.0, index=IDS)
    risks = []
    for lam in (1.0, 10.0, 100.0, 1000.0):
        res = optimize_sleeve(alpha, rm, w_prev=pd.Series(0.0, index=IDS),
                              cost_bps=cost, sleeve="equity", cfg=cfg, lam=lam)
        risks.append(res.risk)
    assert all(risks[i] > risks[i + 1] for i in range(len(risks) - 1)), risks


def test_lambda_calibration_hits_vol_target():
    rm = make_stub(seed=6)
    cfg = loose_cfg()
    cfg["optimizer"]["lambda_bisect_iters"] = 8
    alpha = pd.Series(np.random.default_rng(7).normal(0, 1, 8), index=IDS)
    cost = pd.Series(0.0, index=IDS)  # unconstrained-ish: no cost kink
    target = 0.10
    res = optimize_sleeve(alpha, rm, w_prev=pd.Series(0.0, index=IDS),
                          cost_bps=cost, sleeve="equity", cfg=cfg, vol_target=target)
    assert abs(res.risk / target - 1.0) <= 0.15, res.risk


def test_factor_form_equals_full_sigma():
    """Factor-form objective and dense-Sigma objective give the same weights."""
    rm_f = make_stub(use_factor=True, seed=8)
    rm_d = make_stub(use_factor=False, seed=8)
    cfg = loose_cfg()
    alpha = pd.Series(np.random.default_rng(9).normal(0, 1, 8), index=IDS)
    cost = pd.Series(3.0, index=IDS)
    wp = pd.Series(0.0, index=IDS)
    r_f = optimize_sleeve(alpha, rm_f, wp, cost, "equity", cfg, lam=50.0)
    r_d = optimize_sleeve(alpha, rm_d, wp, cost, "equity", cfg, lam=50.0)
    assert np.allclose(r_f.w.to_numpy(), r_d.w.to_numpy(), atol=1e-4)


def test_beta_neutrality():
    rm = make_stub(seed=10)
    cfg = base_cfg()
    band = cfg["constraints"]["beta_neutral_band"]
    betas = pd.Series(rm.B.iloc[:, 0].to_numpy(), index=IDS)  # market betas
    alpha = pd.Series(np.random.default_rng(11).normal(0, 1, 8), index=IDS)
    cost = pd.Series(5.0, index=IDS)
    res = optimize_sleeve(alpha, rm, pd.Series(0.0, index=IDS), cost,
                          "equity", cfg, lam=1.0, betas=betas)
    assert abs(float(betas.to_numpy() @ res.w.to_numpy())) <= band + 1e-6


# ------------------------------------------------------------------ overlays
def test_vol_target_known_answer():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0, 0.01, 60),
                  index=pd.bdate_range("2020-01-01", periods=60))
    t = r.index[-1] + pd.Timedelta(days=1)  # use all 60 as history
    realized = r.iloc[-21:].std(ddof=1) * np.sqrt(252)
    expected = 0.10 / realized  # within clip
    m = vol_target_multiplier(r, t, target=0.10, lookback=21)
    assert m == pytest.approx(expected)


def test_vol_target_clipping_and_history():
    idx = pd.bdate_range("2020-01-01", periods=60)
    # Tiny vol -> target/realized huge -> clipped to upper bound 1.5.
    low = pd.Series(1e-6, index=idx)
    t = idx[-1] + pd.Timedelta(days=1)
    assert vol_target_multiplier(low, t, target=0.10, lookback=21, clip=(0.0, 1.5)) == 1.5
    # Huge vol -> below lower clip.
    high = pd.Series(np.tile([0.5, -0.5], 30), index=idx)
    assert vol_target_multiplier(high, t, target=0.10, lookback=21, clip=(0.5, 1.5)) == 0.5
    # Insufficient history -> neutral.
    short = pd.Series(np.random.default_rng(0).normal(0, 0.01, 10), index=idx[:10])
    assert vol_target_multiplier(short, idx[10], target=0.10, lookback=21) == 1.0


def test_drawdown_multiplier():
    idx = pd.bdate_range("2020-01-01", periods=50)
    t = idx[-1] + pd.Timedelta(days=1)
    flat = pd.Series(np.ones(50), index=idx)
    assert drawdown_multiplier(flat, t, threshold=0.08, scale=0.5) == 1.0
    # Rise to 1.2 then fall to 1.0 -> ~17% DD from peak.
    path = np.concatenate([np.linspace(1.0, 1.2, 25), np.linspace(1.2, 1.0, 25)])
    deep = pd.Series(path, index=idx)
    assert drawdown_multiplier(deep, t, threshold=0.08, scale=0.5) == 0.5


def _macro_panel(vals, avail_offset_days=1):
    n = len(vals)
    obs = pd.bdate_range("2019-01-01", periods=n)
    avail = obs.tz_localize("UTC") + pd.Timedelta(days=avail_offset_days)
    return pd.DataFrame({
        "obs_date": obs, "series_id": "X", "value": vals,
        "available_from": avail, "source": "syn",
        "ingested_at": avail + pd.Timedelta(minutes=5),
    })


def test_macro_spike_triggers_and_causal_shift():
    rng = np.random.default_rng(0)
    zwin = 30
    baseline = 4.0 + rng.normal(0, 0.05, zwin + 5)
    std_base = baseline[-zwin:].std(ddof=1)
    spike = baseline[-zwin:].mean() + 5.0 * std_base   # ~5 sigma jump
    vals = np.append(baseline, spike)
    panel = _macro_panel(vals)
    t = panel["obs_date"].iloc[-1] + pd.Timedelta(days=5)

    m = macro_derisk_multiplier(panel, t, series=("X",), z_window=zwin,
                                z_trigger=1.5, scale=0.6)
    assert m == pytest.approx(0.6)

    # Causal/self-inflation: with shift(1) the spike is excluded from its own baseline,
    # so z is large; if it were (wrongly) included the inflated std would drop z far lower.
    v = pd.Series(vals)
    z_shifted = (v.iloc[-1] - v.rolling(zwin).mean().shift(1).iloc[-1]) / \
                v.rolling(zwin).std().shift(1).iloc[-1]
    incl = v.iloc[-(zwin):]  # window including the spike
    z_naive = (v.iloc[-1] - incl.mean()) / incl.std(ddof=1)
    # Including the spike inflates both mean and std, strictly deflating the z-score;
    # the shifted (causal) z must exceed both the trigger and the self-inflated version.
    assert z_shifted > 1.5
    assert z_shifted > z_naive


def test_macro_causality_future_corruption():
    rng = np.random.default_rng(1)
    zwin = 30
    vals = 4.0 + rng.normal(0, 0.05, zwin + 6)
    panel = _macro_panel(vals)
    t = panel["obs_date"].iloc[-3] + pd.Timedelta(days=1)  # some rows are "future"
    m0 = macro_derisk_multiplier(panel, t, series=("X",), z_window=zwin, z_trigger=1.5)

    corrupt = panel.copy()
    future = corrupt["available_from"] > (pd.Timestamp(t).tz_localize("UTC"))
    corrupt.loc[future, "value"] = 1e9
    m1 = macro_derisk_multiplier(corrupt, t, series=("X",), z_window=zwin, z_trigger=1.5)
    assert m0 == m1


# ------------------------------------------------------------------ allocation
def test_erc_inverse_vol_two_assets():
    cov = pd.DataFrame(np.diag([4.0, 1.0]), index=["a", "b"], columns=["a", "b"])  # vols 2:1
    w = erc_weights(cov)
    # equal risk contribution on diagonal cov -> inverse vol -> weights 1:2
    assert w["a"] / w["b"] == pytest.approx(0.5, rel=1e-4)
    assert w.sum() == pytest.approx(1.0)


def test_erc_equal_cov_equal_weights():
    n = 5
    C = np.full((n, n), 0.2) + np.diag(np.full(n, 0.8))  # equal variances/correlations
    cov = pd.DataFrame(C, index=list("abcde"), columns=list("abcde"))
    w = erc_weights(cov)
    assert np.allclose(w.to_numpy(), 1.0 / n, atol=1e-6)
    # risk contributions equalized
    m = cov.to_numpy() @ w.to_numpy()
    rc = w.to_numpy() * m
    rc = rc / rc.sum()
    assert rc.max() - rc.min() < 1e-6


def test_apply_risk_caps_crypto():
    idx = ["equity", "crypto", "commodity_etf"]
    cov = pd.DataFrame(np.diag([0.01, 0.09, 0.01]), index=idx, columns=idx)  # crypto high vol
    w0 = pd.Series([1 / 3, 1 / 3, 1 / 3], index=idx)
    w = apply_risk_caps(w0, cov, crypto_cap=0.20, max_sleeve=0.40)
    m = cov.to_numpy() @ w.to_numpy()
    rc = w.to_numpy() * m
    shares = rc / rc.sum()
    assert shares[idx.index("crypto")] <= 0.20 + 1e-6
    assert np.all(shares <= 0.40 + 1e-6)
    assert w.sum() == pytest.approx(1.0)


def test_sleeve_allocation_warmup_haircut():
    idx = pd.bdate_range("2019-01-01", periods=200)
    rng = np.random.default_rng(0)
    eq = rng.normal(0, 0.01, 200)
    cr = np.full(200, np.nan)
    cr[-50:] = rng.normal(0, 0.01, 50)  # crypto only 50 obs -> cold (< 126 warmup)
    sr = pd.DataFrame({"equity": eq, "crypto": cr}, index=idx)
    t = idx[-1] + pd.Timedelta(days=1)
    cfg = base_cfg()
    w = sleeve_allocation(sr, t, cfg)
    assert w.sum() == pytest.approx(1.0)
    # crypto warm-up haircut -> ~0.2 vs warm equity ~0.8 (equal vols).
    assert w["crypto"] == pytest.approx(0.2, abs=0.03)
    assert w["crypto"] < w["equity"]
