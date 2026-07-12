"""Declarative portfolio constraints — cvxpy builders + a post-solve verifier.

Constraints are read from `cfg["constraints"]` (backtest.yaml) so the optimizer and the
verifier share one source of truth. Every constraint is expressed symbolically for the
solver by `build_constraints`, and re-checked numerically after the solve by
`check_constraints` (with a small tolerance to absorb solver slack).
"""
from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp
import numpy as np
import pandas as pd


@dataclass
class ConstraintSet:
    """Resolved numeric bounds for one sleeve."""
    position_cap: float | None
    gross_cap: float
    net_band: float
    beta_band: float | None
    sector_band: float | None
    turnover_cap: float | None


def _cc(cfg: dict) -> dict:
    """Accept either the full backtest cfg or the constraints sub-dict."""
    return cfg["constraints"] if "constraints" in cfg else cfg


def constraint_set(sleeve, cfg: dict) -> ConstraintSet:
    cc = _cc(cfg)
    pc = cc["position_cap"]
    cap = pc[sleeve] if isinstance(sleeve, str) else None
    return ConstraintSet(
        position_cap=cap,
        gross_cap=cc["gross_cap"],
        net_band=cc["net_band"],
        beta_band=cc.get("beta_neutral_band"),
        sector_band=cc.get("sector_band"),
        turnover_cap=cc.get("turnover_cap"),
    )


def _beta_matrix(betas) -> np.ndarray:
    """betas -> array. Series/1-D -> (N,) single factor; DataFrame -> (K, N)."""
    if isinstance(betas, pd.DataFrame):
        return betas.to_numpy(dtype=float).T
    if isinstance(betas, pd.Series):
        return betas.to_numpy(dtype=float)
    return np.asarray(betas, dtype=float)


def _position_caps(sleeve, cc: dict, n: int):
    pc = cc["position_cap"]
    if isinstance(sleeve, (pd.Series, list, tuple, np.ndarray)):
        return np.array([pc[s] for s in list(sleeve)], dtype=float)
    return float(pc[sleeve])


def build_constraints(w, w_prev, sleeve, cfg, betas=None, sectors=None) -> list:
    """Return the cvxpy constraint list for variable `w` (length N, in ids order).

    - |w_i| <= position_cap[sleeve]
    - sum|w| <= gross_cap
    - |sum w| <= net_band
    - Sum|w - w_prev| <= turnover_cap                       (when turnover_cap set)
    - |beta' w| <= beta_neutral_band                        (when betas given)
    - per sector: |sum_{i in sector} w_i| <= sector_band    (when sectors given)
    """
    cc = _cc(cfg)
    n = w.shape[0]
    cons: list = []

    caps = _position_caps(sleeve, cc, n)
    cons += [w <= caps, w >= -caps]

    cons.append(cp.norm1(w) <= cc["gross_cap"])
    cons.append(cp.abs(cp.sum(w)) <= cc["net_band"])

    if cc.get("turnover_cap") is not None and w_prev is not None:
        wp = np.asarray(w_prev, dtype=float)
        cons.append(cp.norm1(w - wp) <= cc["turnover_cap"])

    if betas is not None and cc.get("beta_neutral_band") is not None:
        B = _beta_matrix(betas)
        cons.append(cp.abs(B @ w) <= cc["beta_neutral_band"])

    if sectors is not None and cc.get("sector_band") is not None:
        band = cc["sector_band"]
        sec = pd.Series(sectors).to_numpy()
        for s in pd.unique(sec):
            mask = (sec == s).astype(float)
            cons.append(cp.abs(mask @ w) <= band)

    return cons


def check_constraints(w, w_prev=None, sleeve=None, cfg=None,
                      betas=None, sectors=None, tol: float = 1e-6) -> dict[str, bool]:
    """Numerically verify a solved weight vector. Returns {constraint_name: satisfied}."""
    cc = _cc(cfg)

    # Align any Series args onto the weight index so positions line up.
    if isinstance(w, pd.Series):
        idx = w.index
        if isinstance(betas, pd.Series):
            betas = betas.reindex(idx)
        if isinstance(sectors, pd.Series):
            sectors = sectors.reindex(idx)
        if isinstance(w_prev, pd.Series):
            w_prev = w_prev.reindex(idx)
        if isinstance(sleeve, pd.Series):
            sleeve = sleeve.reindex(idx)

    wv = np.asarray(w, dtype=float)
    res: dict[str, bool] = {}

    caps = _position_caps(sleeve, cc, len(wv))
    res["position_cap"] = bool(np.all(np.abs(wv) <= np.asarray(caps) + tol))
    res["gross"] = bool(np.sum(np.abs(wv)) <= cc["gross_cap"] + tol)
    res["net"] = bool(abs(np.sum(wv)) <= cc["net_band"] + tol)

    if w_prev is not None and cc.get("turnover_cap") is not None:
        wp = np.asarray(w_prev, dtype=float)
        res["turnover"] = bool(np.sum(np.abs(wv - wp)) <= cc["turnover_cap"] + tol)

    if betas is not None and cc.get("beta_neutral_band") is not None:
        B = _beta_matrix(betas)
        res["beta"] = bool(np.all(np.abs(B @ wv) <= cc["beta_neutral_band"] + tol))

    if sectors is not None and cc.get("sector_band") is not None:
        band = cc["sector_band"]
        sec = pd.Series(sectors).to_numpy()
        ok = True
        for s in pd.unique(sec):
            mask = (sec == s).astype(float)
            if abs(mask @ wv) > band + tol:
                ok = False
        res["sector"] = ok

    return res
