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
    overlay_multiplier, overlay_components,
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


# ----------------------------------------- robust alpha-uncertainty ellipsoid (opt-in)
def _diag_stub(ids, var=0.04):
    """Diagonal-covariance stub: zero factor exposure, equal specific variance -> Sigma=var*I.
    Isolates the robust term's effect from cross-name risk coupling."""
    n = len(ids)
    return StubRisk(ids, np.zeros((n, 2)), np.diag([1e-8, 1e-8]), np.full(n, var))


def _robust_cfg(kappa):
    cfg = loose_cfg()
    cfg["optimizer"]["robust_kappa"] = kappa
    return cfg


def _eff_n(w):
    a = np.abs(np.asarray(w, dtype=float))
    denom = float((a ** 2).sum())
    return (float(a.sum()) ** 2 / denom) if denom > 0 else 0.0


def test_robust_kappa_zero_identical_to_no_param():
    """robust_kappa=0 with an alpha_se passed reproduces the no-parameter solution bit-for-bit
    (no SOC atom is built when the radius is zero)."""
    rm = make_stub(seed=12)
    alpha = pd.Series(np.random.default_rng(13).normal(0, 1, 8), index=IDS)
    cost = pd.Series(0.0, index=IDS)
    se = pd.Series(np.abs(np.random.default_rng(14).normal(0, 1, 8)), index=IDS)
    base = optimize_sleeve(alpha, rm, pd.Series(0.0, index=IDS), cost, "equity",
                           loose_cfg(), lam=10.0)
    withse = optimize_sleeve(alpha, rm, pd.Series(0.0, index=IDS), cost, "equity",
                             _robust_cfg(0.0), lam=10.0, alpha_se=se)
    assert np.allclose(base.w.to_numpy(), withse.w.to_numpy(), atol=1e-9)


def test_robust_monotonicity_gross_and_effn():
    """As kappa grows 0 -> 0.5 -> 2.0 with dispersed alpha_se (largest bets most uncertain),
    gross exposure is non-increasing and effective N = (sum|w|)^2 / sum(w^2) non-decreasing:
    robustness spreads bets it cannot trust."""
    ids = list(IDS)
    rm = _diag_stub(ids)
    alpha = pd.Series([3.0, -2.5, 2.0, -1.5, 1.0, -0.8, 0.5, -0.3], index=ids)
    se = alpha.abs()                       # estimation error concentrated on the big bets
    cost = pd.Series(0.0, index=ids)
    gross, effn = [], []
    for k in (0.0, 0.5, 2.0):
        res = optimize_sleeve(alpha, rm, pd.Series(0.0, index=ids), cost, "equity",
                              _robust_cfg(k), lam=1.0, alpha_se=se)
        assert res.status in ("optimal", "optimal_inaccurate"), (k, res.status)
        w = res.w.to_numpy()
        gross.append(float(np.abs(w).sum()))
        effn.append(_eff_n(w))
    assert gross[0] >= gross[1] >= gross[2] - 1e-9, gross
    assert effn[0] <= effn[1] <= effn[2] + 1e-9, effn


def test_robust_asymmetric_uncertainty_downweights_uncertain():
    """Two names, equal alpha and equal risk, but one carries 3x the alpha uncertainty: the
    robust optimum weights the certain name strictly more (the whole point of the ellipsoid)."""
    ids = ["EQ:SYN00:2000-01-03", "EQ:SYN01:2000-01-03"]
    rm = _diag_stub(ids)
    alpha = pd.Series([1.0, 1.0], index=ids)
    se = pd.Series([1.0, 3.0], index=ids)  # second name 3x as uncertain
    cost = pd.Series(0.0, index=ids)
    res = optimize_sleeve(alpha, rm, pd.Series(0.0, index=ids), cost, "equity",
                          _robust_cfg(0.3), lam=1.0, alpha_se=se)
    assert res.status in ("optimal", "optimal_inaccurate"), res.status
    w = res.w
    assert w.iloc[0] > w.iloc[1] + 1e-6, w.to_dict()
    # Without robustness the two would be identical.
    plain = optimize_sleeve(alpha, rm, pd.Series(0.0, index=ids), cost, "equity",
                            _robust_cfg(0.0), lam=1.0, alpha_se=se)
    assert plain.w.iloc[0] == pytest.approx(plain.w.iloc[1])


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


# ------------------------------------------- drawdown ramp / hysteresis / PIT
def _equity_with_dd(dd, n=10):
    """Equity series with peak 1.0 and terminal drawdown `dd`, plus a decision date t
    that uses all `n` rows as (strictly-prior) history."""
    idx = pd.bdate_range("2020-01-01", periods=n)
    vals = np.ones(n)
    vals[-1] = 1.0 - dd
    return pd.Series(vals, index=idx), idx[-1] + pd.Timedelta(days=1)


def test_drawdown_ramp_known_answers():
    """Ramp at DD = 0.5x/1x/1.5x/2x/3x threshold -> 1.0, 1.0, 0.75, scale, scale."""
    thr, scale = 0.08, 0.5
    cases = {0.5: 1.0, 1.0: 1.0, 1.5: 0.75, 2.0: scale, 3.0: scale}
    for k, expected in cases.items():
        e, t = _equity_with_dd(k * thr)
        m = drawdown_multiplier(e, t, threshold=thr, scale=scale, mode="ramp")
        assert m == pytest.approx(expected), (k, m)


def test_drawdown_step_bit_identical_to_old():
    """mode='step' (the default) matches the original binary rule on a shared case set."""
    thr, scale = 0.08, 0.5
    for k in (0.0, 0.5, 1.0, 1.01, 1.5, 2.0, 3.0):
        e, t = _equity_with_dd(k * thr)
        dd = k * thr
        old = scale if dd > thr else 1.0
        assert drawdown_multiplier(e, t, threshold=thr, scale=scale) == old
        assert drawdown_multiplier(e, t, threshold=thr, scale=scale, mode="step") == old


def test_drawdown_hysteresis_holds_and_releases():
    """Once de-risked (prev<1), hold through partial recovery; release below recovery bound."""
    thr, scale, rec = 0.08, 0.5, 0.75
    # Partial recovery: DD between recovery bound (0.06) and threshold (0.08) -> stays at prev.
    e_hold, t_hold = _equity_with_dd(0.07)
    held = drawdown_multiplier(e_hold, t_hold, threshold=thr, scale=scale, mode="ramp",
                               recovery_frac=rec, prev_multiplier=0.75)
    assert held == 0.75
    # No re-risk at the boundary: DD deeper than threshold but ramp above prev -> hold prev.
    e_deep, t_deep = _equity_with_dd(0.12)  # ramp value 0.75 > prev 0.6
    held2 = drawdown_multiplier(e_deep, t_deep, threshold=thr, scale=scale, mode="ramp",
                                recovery_frac=rec, prev_multiplier=0.6)
    assert held2 == 0.6
    # Further de-risking still allowed: ramp below prev -> take the deeper scaling.
    deeper = drawdown_multiplier(e_deep, t_deep, threshold=thr, scale=scale, mode="ramp",
                                 recovery_frac=rec, prev_multiplier=0.9)
    assert deeper == pytest.approx(0.75)
    # Release: DD below recovery bound 0.75*0.08 = 0.06 -> back to 1.0.
    e_rel, t_rel = _equity_with_dd(0.05)
    released = drawdown_multiplier(e_rel, t_rel, threshold=thr, scale=scale, mode="ramp",
                                   recovery_frac=rec, prev_multiplier=0.75)
    assert released == 1.0


def test_drawdown_pit_corruption_both_modes():
    """Corrupting equity at/after t leaves the multiplier at t unchanged (step and ramp)."""
    idx = pd.bdate_range("2020-01-01", periods=40)
    # Deep drawdown established in the prior history, then junk from t onward.
    path = np.concatenate([np.linspace(1.0, 1.2, 20), np.linspace(1.2, 1.0, 20)])
    e = pd.Series(path, index=idx)
    t = idx[25]
    for mode in ("step", "ramp"):
        m0 = drawdown_multiplier(e, t, threshold=0.08, scale=0.5, mode=mode)
        corrupt = e.copy()
        corrupt[corrupt.index >= t] = 1e9
        corrupt2 = e.copy()
        corrupt2[corrupt2.index >= t] = -1e9
        assert drawdown_multiplier(corrupt, t, threshold=0.08, scale=0.5, mode=mode) == m0
        assert drawdown_multiplier(corrupt2, t, threshold=0.08, scale=0.5, mode=mode) == m0


def test_drawdown_whipsaw_fewer_changes_ramp_vs_step():
    """Equity oscillating +-1% around the 8% DD boundary produces strictly fewer multiplier
    CHANGES under ramp+hysteresis than under step (the whipsaw-cost property)."""
    thr, scale, rec = 0.08, 0.5, 0.75
    n = 40
    idx = pd.bdate_range("2020-01-01", periods=n)
    vals = np.ones(n)
    # Peak at index 0; thereafter oscillate equity around 0.92 (DD ~ 0.08) by +-~0.5%.
    for i in range(1, n):
        vals[i] = 0.92 + 0.005 * (-1) ** i    # DD alternates 0.075 / 0.085 across the boundary
    e = pd.Series(vals, index=idx)
    ts = [idx[i] + pd.Timedelta(hours=12) for i in range(1, n)]

    def changes(mode, hysteresis):
        ms, prev = [], None
        for t in ts:
            m = drawdown_multiplier(e, t, threshold=thr, scale=scale, mode=mode,
                                    recovery_frac=rec,
                                    prev_multiplier=prev if hysteresis else None)
            ms.append(m)
            prev = m
        return sum(1 for a, b in zip(ms[:-1], ms[1:]) if a != b)

    step_changes = changes("step", hysteresis=False)
    ramp_changes = changes("ramp", hysteresis=True)
    assert ramp_changes < step_changes, (ramp_changes, step_changes)


def test_overlay_components_product_equals_scalar():
    """overlay_components(...)['product'] equals the overlay_multiplier scalar and the
    product of the individual components."""
    idx = pd.bdate_range("2020-01-01", periods=60)
    r = pd.Series(np.random.default_rng(4).normal(0, 0.01, 60), index=idx)
    eq_path = np.concatenate([np.linspace(1.0, 1.2, 30), np.linspace(1.2, 1.0, 30)])
    equity = pd.Series(eq_path, index=idx)
    t = idx[-1] + pd.Timedelta(days=1)
    cfg = base_cfg()
    panel = _macro_panel(4.0 + np.zeros(300))  # no matching series_id -> neutral macro
    comp = overlay_components(r, equity, panel, t, cfg)
    scalar = overlay_multiplier(r, equity, panel, t, cfg)
    assert comp["product"] == pytest.approx(scalar)
    assert comp["product"] == pytest.approx(comp["vol_target"] * comp["drawdown"] * comp["macro"])


def _macro_panel(vals, avail_offset_days=1):
    n = len(vals)
    obs = pd.bdate_range("2019-01-01", periods=n)
    avail = obs.tz_localize("UTC") + pd.Timedelta(days=avail_offset_days)
    return pd.DataFrame({
        "obs_date": obs, "series_id": "X", "value": vals,
        "available_from": avail, "source": "syn",
        "ingested_at": avail + pd.Timedelta(minutes=5),
    })


def _macro_panel_two(num_vals, den_vals, num_id="A", den_id="B", avail_offset_days=1):
    """Two-leg macro panel on a shared obs_date grid — the input for ratio-pair tests."""
    n = len(num_vals)
    obs = pd.bdate_range("2019-01-01", periods=n)
    avail = obs.tz_localize("UTC") + pd.Timedelta(days=avail_offset_days)

    def leg(vals, sid):
        return pd.DataFrame({
            "obs_date": obs, "series_id": sid, "value": vals,
            "available_from": avail, "source": "syn",
            "ingested_at": avail + pd.Timedelta(minutes=5),
        })

    return pd.concat([leg(num_vals, num_id), leg(den_vals, den_id)], ignore_index=True)


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


# ------------------------------------------------- macro ratio-pair (VIX term structure)
def test_macro_ratio_known_answer():
    """Hand-built two-leg frame: the derived ratio z (causal, shift(1)) and the multiplier
    match. A final backwardation jump in the numerator lifts the ratio z above trigger."""
    rng = np.random.default_rng(5)
    zwin = 30
    num = 18.0 + rng.normal(0, 0.3, zwin + 6)
    den = 20.0 + rng.normal(0, 0.3, zwin + 6)
    num[-1] = 34.0                                  # ratio spikes on the last obs
    panel = _macro_panel_two(num, den)
    t = panel["obs_date"].max() + pd.Timedelta(days=5)

    ratio = pd.Series(num / den)                    # legs share the obs grid -> elementwise
    z = (ratio.iloc[-1] - ratio.rolling(zwin).mean().shift(1).iloc[-1]) / \
        ratio.rolling(zwin).std().shift(1).iloc[-1]
    assert z > 1.5                                  # sanity: this is a stress spike

    # Trigger just under z -> scales; just over z -> neutral. Pins the derived z exactly.
    assert macro_derisk_multiplier(panel, t, series=(), z_window=zwin,
                                   z_trigger=float(z) - 0.01, scale=0.6,
                                   ratio_pairs=[["A", "B"]]) == pytest.approx(0.6)
    assert macro_derisk_multiplier(panel, t, series=(), z_window=zwin,
                                   z_trigger=float(z) + 0.01, scale=0.6,
                                   ratio_pairs=[["A", "B"]]) == 1.0


def test_macro_ratio_discriminates_backwardation_from_parallel_shift():
    """The discriminating test: a backwardation episode (num rises above den, ratio z high)
    scales, while a parallel proportional shift of BOTH legs (ratio constant) does NOT —
    even though each level's z would have fired on that very same panel."""
    rng = np.random.default_rng(6)
    zwin = 30
    base_num = 18.0 + rng.normal(0, 0.2, zwin + 6)
    base_den = 20.0 + rng.normal(0, 0.2, zwin + 6)

    # (a) backwardation: numerator spikes above denominator -> ratio z large -> scales.
    num_b, den_b = base_num.copy(), base_den.copy()
    num_b[-1] = 36.0                                # ratio 36/20 = 1.8 vs ~0.9 baseline
    panel_b = _macro_panel_two(num_b, den_b)
    t_b = panel_b["obs_date"].max() + pd.Timedelta(days=5)
    assert macro_derisk_multiplier(panel_b, t_b, series=(), z_window=zwin,
                                   z_trigger=1.5, scale=0.6,
                                   ratio_pairs=[["A", "B"]]) == pytest.approx(0.6)

    # (b) parallel shift: BOTH legs doubled on the last obs -> ratio 36/40 = 0.9 unchanged.
    num_p, den_p = base_num.copy(), base_den.copy()
    num_p[-1], den_p[-1] = 36.0, 40.0
    panel_p = _macro_panel_two(num_p, den_p)
    t_p = panel_p["obs_date"].max() + pd.Timedelta(days=5)

    # the ratio (term-structure) overlay does NOT scale ...
    assert macro_derisk_multiplier(panel_p, t_p, series=(), z_window=zwin,
                                   z_trigger=1.5, scale=0.6,
                                   ratio_pairs=[["A", "B"]]) == 1.0
    # ... even though each LEVEL's z would have fired on the same panel (the pathology).
    assert macro_derisk_multiplier(panel_p, t_p, series=("A", "B"), z_window=zwin,
                                   z_trigger=1.5, scale=0.6) == pytest.approx(0.6)


def test_macro_ratio_pit_future_corruption():
    """Corrupting future macro rows (available_from > t) cannot move the ratio-path
    multiplier at t."""
    rng = np.random.default_rng(7)
    zwin = 30
    num = 18.0 + rng.normal(0, 0.2, zwin + 6)
    den = 20.0 + rng.normal(0, 0.2, zwin + 6)
    panel = _macro_panel_two(num, den)
    t = panel["obs_date"].iloc[-3] + pd.Timedelta(days=1)   # last rows are "future"
    m0 = macro_derisk_multiplier(panel, t, series=(), z_window=zwin,
                                 z_trigger=1.5, ratio_pairs=[["A", "B"]])

    corrupt = panel.copy()
    future = corrupt["available_from"] > pd.Timestamp(t).tz_localize("UTC")
    corrupt.loc[future, "value"] = 1e9
    m1 = macro_derisk_multiplier(corrupt, t, series=(), z_window=zwin,
                                 z_trigger=1.5, ratio_pairs=[["A", "B"]])
    assert m0 == m1


def test_macro_ratio_pairs_absent_back_compat():
    """ratio_pairs absent / None / [] reproduces the pre-existing level-only behavior
    bit-for-bit on the same inputs."""
    rng = np.random.default_rng(8)
    zwin = 30
    baseline = 4.0 + rng.normal(0, 0.05, zwin + 5)
    spike = baseline[-zwin:].mean() + 5.0 * baseline[-zwin:].std(ddof=1)
    vals = np.append(baseline, spike)
    panel = _macro_panel(vals)                       # single level series "X"
    t = panel["obs_date"].iloc[-1] + pd.Timedelta(days=5)

    old = macro_derisk_multiplier(panel, t, series=("X",), z_window=zwin,
                                  z_trigger=1.5, scale=0.6)
    assert old == pytest.approx(0.6)
    assert macro_derisk_multiplier(panel, t, series=("X",), z_window=zwin, z_trigger=1.5,
                                   scale=0.6, ratio_pairs=None) == old
    assert macro_derisk_multiplier(panel, t, series=("X",), z_window=zwin, z_trigger=1.5,
                                   scale=0.6, ratio_pairs=[]) == old


# ------------------------------------------- macro multi-vintage (PIT dedup at the call site)
def _macro_revision(obs_date, value, series_id="X", avail_offset_days=30):
    """A single revised vintage of `obs_date`, published `avail_offset_days` after obs."""
    avail = pd.Timestamp(obs_date).tz_localize("UTC") + pd.Timedelta(days=avail_offset_days)
    return pd.DataFrame({
        "obs_date": [pd.Timestamp(obs_date)], "series_id": [series_id], "value": [value],
        "available_from": [avail], "source": ["syn"],
        "ingested_at": [avail + pd.Timedelta(minutes=5)],
    })


def _multivintage_case():
    """Original single-vintage frame + one revised obs.

    The LAST obs is a stress spike that fires the overlay (m=0.6 on the originals). The
    revised obs is the second-to-last one; its revised value is wild enough that, once
    knowable, it explodes the trailing baseline std and the spike no longer clears the
    trigger (m flips to 1.0). Original published obs+1d; revision published obs+30d.
    Returns (orig_frame, revised_single_vintage_frame, multivintage_frame, revised_obs,
    zwin, trigger_kwargs).
    """
    rng = np.random.default_rng(0)
    zwin = 30
    vals = 4.0 + rng.normal(0, 0.05, zwin + 5)
    # Last obs: a ~5-sigma spike vs its trailing window -> fires on the originals.
    vals[-1] = vals[-(zwin + 1):-1].mean() + 5.0 * vals[-(zwin + 1):-1].std(ddof=1)
    orig = _macro_panel(vals)                                  # single vintage, series "X"

    revised_obs = orig["obs_date"].iloc[-2]
    revised_value = 1.0e6                                      # wild revision of obs[-2]

    # Single-vintage frame carrying the revised value in place of the original obs[-2].
    revised_single = orig.copy()
    revised_single.loc[revised_single["obs_date"] == revised_obs, "value"] = revised_value

    # Multi-vintage: originals + the later revision row for obs[-2].
    rev_row = _macro_revision(revised_obs, revised_value, avail_offset_days=30)
    multi = pd.concat([orig, rev_row], ignore_index=True)

    kw = dict(series=("X",), z_window=zwin, z_trigger=1.5, scale=0.6)
    return orig, revised_single, multi, revised_obs, kw


def test_macro_multivintage_between_publications_bit_identical():
    """At t after every original release but before the revision's availability, the
    multi-vintage frame reduces (via asof_panel dedup) to exactly the original vintage:
    the multiplier is bit-identical to the single-vintage original frame — and non-trivial
    (the last-obs spike fires, so m=0.6, not a vacuous 1.0)."""
    orig, _revised_single, multi, revised_obs, kw = _multivintage_case()
    max_obs = orig["obs_date"].max()
    t_between = max_obs + pd.Timedelta(days=5)                 # < revised_obs + 30d
    assert t_between < pd.Timestamp(revised_obs) + pd.Timedelta(days=30)

    m_orig = macro_derisk_multiplier(orig, t_between, **kw)
    m_multi = macro_derisk_multiplier(multi, t_between, **kw)
    assert m_orig == pytest.approx(0.6)                        # non-trivial: spike fired
    assert m_multi == m_orig                                   # bit-identical


def test_macro_multivintage_after_revision_uses_revised_value():
    """Once the revision is knowable (t >= revised_obs + 30d) the latest visible vintage
    is the revised value: the multiplier matches the single-vintage frame built with that
    revised value, and differs from the between-publications answer."""
    orig, revised_single, multi, revised_obs, kw = _multivintage_case()
    max_obs = orig["obs_date"].max()
    t_after = max_obs + pd.Timedelta(days=40)                  # >= revised_obs + 30d
    assert t_after >= pd.Timestamp(revised_obs) + pd.Timedelta(days=30)

    m_after = macro_derisk_multiplier(multi, t_after, **kw)
    m_revised_single = macro_derisk_multiplier(revised_single, t_after, **kw)
    assert m_after == m_revised_single                         # revised vintage in effect
    assert m_after == pytest.approx(1.0)                       # wild revision kills the z
    # And it genuinely changed the decision vs the between-publications window.
    t_between = max_obs + pd.Timedelta(days=5)
    assert macro_derisk_multiplier(multi, t_between, **kw) == pytest.approx(0.6)


def test_macro_multivintage_future_revision_corruption():
    """Vintage corruption: a revision published strictly after t cannot move the
    multiplier at t, whatever value it carries (it is not yet knowable)."""
    orig, _revised_single, multi, revised_obs, kw = _multivintage_case()
    max_obs = orig["obs_date"].max()
    t_between = max_obs + pd.Timedelta(days=5)                 # revision not yet visible
    m0 = macro_derisk_multiplier(multi, t_between, **kw)

    corrupt = multi.copy()
    is_rev = (corrupt["obs_date"] == revised_obs) & (
        pd.to_datetime(corrupt["available_from"], utc=True)
        > pd.Timestamp(t_between).tz_localize("UTC"))
    corrupt.loc[is_rev, "value"] = -9.9e12                     # arbitrary future junk
    m1 = macro_derisk_multiplier(corrupt, t_between, **kw)
    assert m0 == m1
    # Sanity: the corrupted row IS a future (post-t) vintage.
    assert bool(is_rev.any())


# --------------------------------------------------- vol-target EWMA + deadband
def test_vol_target_ewma_known_answer():
    """EWMA-std estimator reproduces target / (ewm-std * sqrt252), within clip."""
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0, 0.01, 120),
                  index=pd.bdate_range("2020-01-01", periods=120))
    t = r.index[-1] + pd.Timedelta(days=1)
    h = 33.0
    ew = r.ewm(halflife=h, min_periods=21).std(bias=False).iloc[-1]
    expected = 0.10 / (ew * np.sqrt(252.0))  # within clip for ~1% daily vol
    m = vol_target_multiplier(r, t, target=0.10, lookback=21, ewma_halflife=h)
    assert m == pytest.approx(expected)


def test_vol_target_constant_vol_both_estimators():
    """Constant-vol series: multiplier -> target/realized under BOTH estimators."""
    idx = pd.bdate_range("2020-01-01", periods=120)
    a = 0.01
    r = pd.Series(np.tile([a, -a], 60), index=idx)  # stationary magnitude
    t = idx[-1] + pd.Timedelta(days=1)
    for hl in (None, 33.0):
        m = vol_target_multiplier(r, t, target=0.10, lookback=21, ewma_halflife=hl)
        if hl is None:
            realized = r.iloc[-21:].std(ddof=1) * np.sqrt(252.0)
        else:
            realized = r.ewm(halflife=hl, min_periods=21).std(bias=False).iloc[-1] * np.sqrt(252.0)
        expected = float(np.clip(0.10 / realized, 0.0, 1.5))
        assert m == pytest.approx(expected)


def test_vol_target_deadband_holds_and_releases():
    """Within band -> returns prev exactly; outside band -> returns the new value."""
    idx = pd.bdate_range("2020-01-01", periods=60)
    r = pd.Series(np.random.default_rng(1).normal(0, 0.01, 60), index=idx)
    t = idx[-1] + pd.Timedelta(days=1)
    new = vol_target_multiplier(r, t, target=0.10, lookback=21)

    prev_close = new * 1.05          # |new/prev - 1| = 0.0476 <= 0.10 -> hold
    held = vol_target_multiplier(r, t, target=0.10, lookback=21,
                                 prev_multiplier=prev_close, deadband=0.10)
    assert held == prev_close        # exact re-use, not the freshly computed value

    prev_far = new * 2.0             # |new/prev - 1| = 0.5 > 0.10 -> release
    released = vol_target_multiplier(r, t, target=0.10, lookback=21,
                                     prev_multiplier=prev_far, deadband=0.10)
    assert released == pytest.approx(new)


def test_vol_target_deadband_zero_bit_identical():
    """deadband=0 (any prev) reproduces the un-hysteretic value bit-for-bit, both estimators."""
    idx = pd.bdate_range("2020-01-01", periods=90)
    r = pd.Series(np.random.default_rng(2).normal(0, 0.01, 90), index=idx)
    t = idx[-1] + pd.Timedelta(days=1)
    for hl in (None, 33.0):
        base = vol_target_multiplier(r, t, target=0.10, lookback=21, ewma_halflife=hl)
        with_prev = vol_target_multiplier(r, t, target=0.10, lookback=21, ewma_halflife=hl,
                                          prev_multiplier=0.123456, deadband=0.0)
        assert with_prev == base     # exact equality


def test_vol_target_pit_corruption_both_estimators():
    """Corrupting returns at/after t leaves the multiplier at t unchanged (both estimators)."""
    idx = pd.bdate_range("2020-01-01", periods=80)
    r = pd.Series(np.random.default_rng(3).normal(0, 0.01, 80), index=idx)
    t = idx[50]  # rows with index >= t are the future and must not enter the answer
    for hl in (None, 33.0):
        m0 = vol_target_multiplier(r, t, target=0.10, lookback=21, ewma_halflife=hl)
        corrupt = r.copy()
        corrupt[corrupt.index >= t] = 1e6
        m1 = vol_target_multiplier(corrupt, t, target=0.10, lookback=21, ewma_halflife=hl)
        assert m0 == m1


def test_vol_target_turnover_reduction():
    """On seeded noisy-vol synthetic returns, sum|m_t - m_{t-1}| is strictly lower with
    (EWMA smoothing + deadband) than with the raw window and no deadband."""
    rng = np.random.default_rng(7)
    n = 400
    idx = pd.bdate_range("2019-01-01", periods=n)
    base_vol = 0.008 + 0.006 * np.abs(np.sin(np.linspace(0, 8 * np.pi, n)))
    vol_noise = np.exp(rng.normal(0, 0.4, n))     # multiplicative whipsaw in realized vol
    r = pd.Series(rng.normal(0, 1, n) * base_vol * vol_noise, index=idx)
    ts = idx[60:]                                  # decision dates with enough trailing history

    def path(ewma_halflife, deadband):
        ms, prev = [], None
        for t in ts:
            m = vol_target_multiplier(r, t, target=0.10, lookback=21,
                                      ewma_halflife=ewma_halflife,
                                      prev_multiplier=prev, deadband=deadband)
            ms.append(m)
            prev = m
        return np.asarray(ms)

    tv_raw = float(np.abs(np.diff(path(None, 0.0))).sum())
    tv_smooth = float(np.abs(np.diff(path(33.0, 0.10))).sum())
    assert tv_smooth < tv_raw


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
