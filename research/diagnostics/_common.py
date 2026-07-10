"""Shared helpers for the Kalshi research diagnostics (research-only; production
never imports from research/).

Fee model — Kalshi taker fee, Whelan-imputed per-contract rate: the exchange
charges ceil-to-cent(0.07 * C * p * (1-p)) dollars on C contracts at price p;
following Buergi-Deng-Whelan we impute the per-contract fee from a 100-lot,
c(p) = ceil(700 * p * (1-p)) / 10000, which makes the 50c fee 1.75c and the 5c
fee 0.34c per contract. Maker fee is 25% of taker (2026-07-07 schedule) — the
diagnostics price the TAKER side (the documented bias is taker-side).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd


def taker_fee(p: float) -> float:
    """Per-contract taker fee in dollars at price p, imputed on a 100-lot."""
    return math.ceil(700.0 * p * (1.0 - p)) / 10000.0


def post_fee_return(entry: float, terminal: float) -> float:
    """Whelan eq. (3): r = (y - p - c) / (p + c) — commission counts as investment."""
    c = taker_fee(entry)
    return (terminal - entry - c) / (entry + c)


def pre_fee_return(entry: float, terminal: float) -> float:
    """Whelan eq. (2): r = (y - p) / p."""
    return (terminal - entry) / entry


def cluster_ols(y: np.ndarray, X: np.ndarray, clusters: np.ndarray) -> dict:
    """OLS with CR1 cluster-robust standard errors (no statsmodels dependency).

    Returns {beta, se, t, n, n_clusters}; X should already include a constant
    column if an intercept is wanted.
    """
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    n, k = X.shape
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    resid = y - X @ beta
    labels = pd.factorize(pd.Series(clusters))[0]
    g = int(labels.max()) + 1
    meat = np.zeros((k, k))
    for lab in range(g):
        idx = labels == lab
        Xg = X[idx]
        eg = resid[idx]
        s = Xg.T @ eg
        meat += np.outer(s, s)
    # CR1 small-sample correction
    corr = (g / (g - 1)) * ((n - 1) / (n - k)) if g > 1 and n > k else 1.0
    V = corr * XtX_inv @ meat @ XtX_inv
    se = np.sqrt(np.diag(V))
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(se > 0, beta / se, np.nan)
    return {"beta": beta, "se": se, "t": t, "n": n, "n_clusters": g}


def cluster_mean_test(returns: pd.Series, clusters: pd.Series) -> dict:
    """Mean with cluster-robust t-stat (intercept-only cluster OLS)."""
    r = cluster_ols(returns.to_numpy(), np.ones((len(returns), 1)),
                    clusters.to_numpy())
    return {"mean": float(r["beta"][0]), "se": float(r["se"][0]),
            "t": float(r["t"][0]), "n": r["n"], "n_clusters": r["n_clusters"]}
