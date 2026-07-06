"""Sleeve-level capital allocation — equal risk contribution with risk caps + warm-up.

`erc_weights` solves the equal-risk-contribution portfolio by a sqrt-damped fixed point;
`apply_risk_caps` clips individual sleeves' risk-contribution shares; `sleeve_allocation`
wires them together over a trailing EWMA covariance, with an inverse-vol warm-up rule for
sleeves that lack enough history to trust their covariance. Everything is trailing-only:
`sleeve_allocation` uses returns strictly before the decision date.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _ewma_cov(returns: pd.DataFrame, halflife: float) -> pd.DataFrame:
    """Exponentially-weighted covariance over complete-case rows (newest weighted most)."""
    df = returns.dropna(how="any")
    T = len(df)
    if T == 0:
        return pd.DataFrame(np.zeros((returns.shape[1], returns.shape[1])),
                            index=returns.columns, columns=returns.columns)
    lam = 0.5 ** (1.0 / halflife)
    w = lam ** np.arange(T - 1, -1, -1)  # oldest -> smallest weight, newest -> 1
    w = w / w.sum()
    X = df.to_numpy(dtype=float)
    mu = np.average(X, axis=0, weights=w)
    Xc = X - mu
    cov = (Xc * w[:, None]).T @ Xc
    cov = 0.5 * (cov + cov.T)
    return pd.DataFrame(cov, index=df.columns, columns=df.columns)


def _risk_contrib_shares(w: np.ndarray, Sigma: np.ndarray) -> np.ndarray:
    m = Sigma @ w
    rc = w * m
    tot = rc.sum()
    if tot <= 0:
        return np.full_like(w, 1.0 / len(w))
    return rc / tot


def erc_weights(cov: pd.DataFrame, tol: float = 1e-8, max_iter: int = 200) -> pd.Series:
    """Equal-risk-contribution weights via the sqrt-damped fixed point.

    Iteration x_i <- sqrt(x_i / (Sigma x)_i), renormalized. Its fixed point satisfies
    x_i (Sigma x)_i = const (equal risk contributions); the sqrt damping avoids the
    oscillation of the raw x_i ∝ 1/(Sigma x)_i map. For a diagonal covariance this
    collapses to inverse-vol weights. Converges when the dispersion of normalized risk
    contributions (max - min) falls below `tol`.
    """
    idx = cov.index
    Sigma = cov.to_numpy(dtype=float)
    n = Sigma.shape[0]
    x = np.full(n, 1.0 / n)
    for _ in range(max_iter):
        m = np.maximum(Sigma @ x, 1e-16)
        x = np.sqrt(x / m)
        x = x / x.sum()
        shares = _risk_contrib_shares(x, Sigma)
        if shares.max() - shares.min() < tol:
            break
    return pd.Series(x, index=idx)


def apply_risk_caps(w, cov, crypto_cap: float = 0.20, max_sleeve: float = 0.40,
                    max_iter: int = 500, tol: float = 1e-9) -> pd.Series:
    """Cap each sleeve's *risk-contribution* share, then renormalize weights to sum 1.

    Convergence: each pass finds the single worst violator j and shrinks w_j by the
    damped factor sqrt(cap_j / share_j) (< 1), which strictly lowers share_j; renormalizing
    lifts the others. Because the total over-cap risk share is monotonically drained toward
    the feasible simplex, the iteration converges; `max_iter` is a safety bound.
    """
    w = pd.Series(w, dtype=float).copy()
    idx = w.index
    Sigma = pd.DataFrame(cov).reindex(index=idx, columns=idx).to_numpy(dtype=float)
    caps = np.array([crypto_cap if s == "crypto" else max_sleeve for s in idx], dtype=float)

    wv = np.clip(w.to_numpy(dtype=float), 0.0, None)
    if wv.sum() <= 0:
        return w
    wv = wv / wv.sum()

    for _ in range(max_iter):
        shares = _risk_contrib_shares(wv, Sigma)
        viol = shares - caps
        if np.all(viol <= tol):
            break
        j = int(np.argmax(viol))
        wv[j] *= np.sqrt(caps[j] / max(shares[j], 1e-12))
        wv = wv / wv.sum()

    return pd.Series(wv, index=idx)


def sleeve_allocation(sleeve_returns: pd.DataFrame, t, cfg: dict) -> pd.Series:
    """Allocate across sleeves at date t from trailing sleeve returns (strictly before t).

    Warm sleeves (>= warmup_days observations) are allocated by ERC on an EWMA covariance
    (halflife from cfg). Cold sleeves (fewer obs) fall back to an inverse-vol weight with a
    0.5 haircut — half of their inverse-vol fair share — because their covariance is not yet
    trustworthy. Warm ERC weights and haircut cold weights are pooled and normalized to 1.
    """
    alloc = cfg["allocation"]
    halflife = float(alloc["sleeve_cov_halflife_days"])
    warmup = int(alloc["warmup_days"])

    r = sleeve_returns[sleeve_returns.index < t]
    cols = list(sleeve_returns.columns)
    counts = r.notna().sum()

    cov = _ewma_cov(r, halflife)
    vol = pd.Series(np.sqrt(np.clip(np.diag(cov.to_numpy(dtype=float)), 0.0, None)),
                    index=cov.index)

    warm = [s for s in cols if counts.get(s, 0) >= warmup]
    cold = [s for s in cols if 0 < counts.get(s, 0) < warmup]

    raw = pd.Series(0.0, index=cols)

    # Cold sleeves: inverse-vol fair share (over all sleeves), halved.
    inv = pd.Series(0.0, index=cols)
    for s in cols:
        v = vol.get(s, np.nan)
        if np.isfinite(v) and v > 0:
            inv[s] = 1.0 / v
    inv_frac = inv / inv.sum() if inv.sum() > 0 else inv
    for s in cold:
        raw[s] = 0.5 * inv_frac[s]

    # Warm sleeves: ERC (single warm sleeve trivially gets full weight within its block).
    if len(warm) >= 2:
        w_erc = erc_weights(cov.reindex(index=warm, columns=warm))
        for s in warm:
            raw[s] = w_erc[s]
    elif len(warm) == 1:
        raw[warm[0]] = 1.0

    total = raw.sum()
    if total > 0:
        raw = raw / total
    return raw
