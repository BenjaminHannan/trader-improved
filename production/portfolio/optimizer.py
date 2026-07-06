"""Single-sleeve mean-variance optimizer (Grinold-Kahn), factor-form aware.

Objective (per contract):
    max   alpha' w  -  lambda * risk(w)  -  kappa * sum_i c_i |w_i - w_prev_i|

`risk(w)` is expressed in the low-rank factor form when the risk model exposes one —
    ||F^(1/2) B' w||^2 + ||D^(1/2) w||^2  ==  w' (B F B' + diag(D)) w
— which keeps the QP at dimension K+N instead of forming the dense N x N covariance;
for small sleeves (factor_form() is None) we fall back to w' Sigma w with a PSD wrap.

`c_i` is cost in *return units* (cost_bps / 1e4) and kappa = optimizer.tcost_weight.
lambda is calibrated by a bracketed search (log-space, interpolated) so the solved
portfolio's annualized vol lands near the sleeve vol target; a caller-supplied `lam`
short-circuits the calibration. Solver order: CLARABEL then OSQP; total failure returns
`w_prev` with a non-optimal status rather than raising.
"""
from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp
import numpy as np
import pandas as pd

from .constraints import build_constraints


@dataclass
class OptResult:
    w: pd.Series
    lam: float
    status: str
    expected_alpha: float
    risk: float


def _factor_sqrt(F: np.ndarray) -> np.ndarray:
    """Symmetric factor of F with non-negative eigenvalues: returns S with S' S = F_clipped."""
    evals, evecs = np.linalg.eigh(np.asarray(F, dtype=float))
    evals = np.clip(evals, 0.0, None)
    return np.sqrt(evals)[:, None] * evecs.T


def optimize_sleeve(alpha, risk_model, w_prev, cost_bps, sleeve, cfg,
                    vol_target=None, lam=None, betas=None, sectors=None,
                    alpha_se=None) -> OptResult:
    ids = pd.Index(risk_model.ids)
    n = len(ids)

    a = pd.Series(alpha).reindex(ids).fillna(0.0).to_numpy(dtype=float)
    wp = (pd.Series(w_prev).reindex(ids).fillna(0.0).to_numpy(dtype=float)
          if w_prev is not None else np.zeros(n))
    c = pd.Series(cost_bps).reindex(ids).fillna(0.0).to_numpy(dtype=float) / 1e4
    kappa = float(cfg["optimizer"]["tcost_weight"])
    # Alpha-uncertainty robustness (Goldfarb-Iyengar ellipsoid). Opt-in: default
    # robust_kappa=0 leaves the objective bit-identical to the plain MV problem (no extra
    # SOC atom is built), see research/wiki/questions/research-robust-alpha.md.
    robust_kappa = float(cfg["optimizer"].get("robust_kappa", 0.0))

    betas_a = betas.reindex(ids).to_numpy(dtype=float) if isinstance(betas, pd.Series) else betas
    sectors_a = sectors.reindex(ids).to_numpy() if isinstance(sectors, pd.Series) else sectors

    w = cp.Variable(n)
    lam_param = cp.Parameter(nonneg=True)

    ff = risk_model.factor_form()
    if ff is not None:
        B, F, D = ff
        Bm = pd.DataFrame(B).reindex(ids).fillna(0.0).to_numpy(dtype=float)  # N x K
        Fsq = _factor_sqrt(np.asarray(F, dtype=float))                       # K x K
        Dv = pd.Series(D).reindex(ids).fillna(0.0).to_numpy(dtype=float)     # N
        risk_expr = (cp.sum_squares(Fsq @ Bm.T @ w)
                     + cp.sum_squares(cp.multiply(np.sqrt(np.clip(Dv, 0.0, None)), w)))
    else:
        Sigma = pd.DataFrame(risk_model.covariance()).reindex(index=ids, columns=ids).to_numpy(dtype=float)
        Sigma = 0.5 * (Sigma + Sigma.T)
        risk_expr = cp.quad_form(w, cp.psd_wrap(Sigma))

    cost_expr = kappa * cp.sum(cp.multiply(c, cp.abs(w - wp)))
    obj_expr = a @ w - lam_param * risk_expr - cost_expr
    if robust_kappa > 0.0 and alpha_se is not None:
        # Diagonal ellipsoid: penalize sqrt(sum_i (se_i * w_i)^2) = ||diag(se) w||_2.
        se = pd.Series(alpha_se).reindex(ids).fillna(0.0).to_numpy(dtype=float)
        obj_expr = obj_expr - robust_kappa * cp.norm(cp.multiply(se, w), 2)
    objective = cp.Maximize(obj_expr)
    cons = build_constraints(w, wp, sleeve, cfg, betas=betas_a, sectors=sectors_a)
    prob = cp.Problem(objective, cons)

    def solve(lam_value) -> np.ndarray | None:
        lam_param.value = float(lam_value)
        for solver in (cp.CLARABEL, cp.OSQP):
            try:
                prob.solve(solver=solver)
            except Exception:
                continue
            if prob.status in ("optimal", "optimal_inaccurate") and w.value is not None:
                return np.asarray(w.value, dtype=float).ravel()
        return None

    def result(wsol, lam_value, status) -> OptResult:
        w_ser = pd.Series(wsol, index=ids)
        return OptResult(w_ser, float(lam_value), status,
                         float(a @ wsol), float(risk_model.portfolio_vol(w_ser)))

    def fail() -> OptResult:
        w_ser = pd.Series(wp, index=ids)
        return OptResult(w_ser, float("nan"), "infeasible",
                         float(a @ wp), float(risk_model.portfolio_vol(w_ser)))

    # --- explicit lambda: single solve ------------------------------------
    if lam is not None:
        wsol = solve(lam)
        return result(wsol, lam, "optimal") if wsol is not None else fail()

    target = vol_target if vol_target is not None else cfg.get("sleeve_vol_target")

    # --- no target to calibrate against: use the config init lambda -------
    if target is None:
        lam0 = float(cfg["optimizer"].get("risk_aversion_init", 10.0))
        wsol = solve(lam0)
        return result(wsol, lam0, "optimal") if wsol is not None else fail()

    # --- calibrate lambda so annualized vol ~ target ----------------------
    # Portfolio vol is (near-)monotone decreasing in lambda; in log-log it is close to
    # linear (unconstrained: vol ∝ 1/lambda), so we bracket [1e-2, 1e4] and step by
    # log-linear interpolation, keeping the best (closest-to-target) solve seen.
    iters = int(cfg["optimizer"].get("lambda_bisect_iters", 5))
    log_lo, log_hi = np.log10(1e-2), np.log10(1e4)
    best: tuple[float, np.ndarray, float] | None = None

    def evaluate(x):
        s = solve(10.0 ** x)
        if s is None:
            return None, None
        return s, float(risk_model.portfolio_vol(pd.Series(s, index=ids)))

    s_lo, v_lo = evaluate(log_lo)
    s_hi, v_hi = evaluate(log_hi)
    for s, x, v in ((s_lo, log_lo, v_lo), (s_hi, log_hi, v_hi)):
        if s is not None and target > 0:
            d = abs(v / target - 1.0)
            if best is None or d < best[0]:
                best = (d, s, 10.0 ** x)

    for _ in range(max(1, iters)):
        if (v_lo is not None and v_hi is not None and v_lo > 0 and v_hi > 0
                and abs(np.log10(v_lo) - np.log10(v_hi)) > 1e-9):
            frac = (np.log10(target) - np.log10(v_lo)) / (np.log10(v_hi) - np.log10(v_lo))
            frac = min(max(frac, 0.02), 0.98)
            x = log_lo + frac * (log_hi - log_lo)
        else:
            x = 0.5 * (log_lo + log_hi)
        s, v = evaluate(x)
        if s is None:
            log_lo = x  # push toward more penalty (smaller, tamer solutions)
            continue
        d = abs(v / target - 1.0) if target > 0 else abs(v)
        if best is None or d < best[0]:
            best = (d, s, 10.0 ** x)
        if v > target:
            log_lo, v_lo = x, v
        else:
            log_hi, v_hi = x, v

    if best is None:
        return fail()
    _, wsol, used_lam = best
    return result(wsol, used_lam, "optimal")
