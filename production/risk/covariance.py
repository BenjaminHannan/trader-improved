"""EWMA covariance estimation with diagonal shrinkage.

Used for two things in the risk model: the factor-return covariance ``F`` in the
structural sleeves, and the full instrument covariance ``_cov`` in the small
sleeves (fx_etf / commodity_etf) where an explicit factor structure would just be
fitting noise on a handful of names.

PIT note: this function is a pure estimator over whatever rows the caller passes.
The caller is responsible for only handing it returns computed from ``obs_date <=
as_of`` rows (see ``model.RiskModel.build`` and ``exposures``). Nothing here looks
ahead; the most-recent row carries the most weight, oldest the least.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_JITTER = 1e-12


class RiskError(Exception):
    """Raised when an estimator cannot be formed (e.g. too few observations)."""


def _ewma_weights(n: int, halflife: float) -> np.ndarray:
    """Exponential weights, most-recent row (index n-1) heaviest.

    ``decay = 0.5 ** (1/halflife)`` so the weight halves every ``halflife`` rows.
    Weights are returned oldest->newest to match a chronologically sorted frame.
    """
    decay = 0.5 ** (1.0 / float(halflife))
    ages = np.arange(n - 1, -1, -1)  # oldest row has the largest age
    return decay ** ages


def ewma_cov(returns: pd.DataFrame, halflife: float, shrink: float = 0.3,
             min_obs: int = 60) -> pd.DataFrame:
    """EWMA covariance of ``returns`` (rows=dates, cols=entities), diagonally shrunk.

    Steps: drop all-NaN rows, require ``>= min_obs`` observations (else raise),
    compute the exponentially-weighted covariance around the weighted mean, apply
    shrinkage ``(1-s)*C + s*diag(C)``, symmetrize, and add a tiny diagonal jitter
    so the result is numerically PSD.

    ``shrink=0`` returns the raw EWMA covariance (plus jitter); ``shrink=1``
    returns a pure diagonal matrix — both exact, which the unit tests pin.
    """
    if returns.shape[1] == 0:
        raise RiskError("ewma_cov: no columns to estimate")
    # Keep rows where at least one entity has a return; within a sleeve the
    # calendar is uniform, so this is normally a no-op.
    r = returns.dropna(how="all").sort_index()
    n = len(r)
    if n < min_obs:
        raise RiskError(f"ewma_cov: {n} observations < min_obs={min_obs}")

    cols = list(r.columns)
    X = r.to_numpy(dtype=float)
    w = _ewma_weights(n, halflife)

    # Weighted mean, ignoring NaNs per column via masked weight renormalization.
    mask = ~np.isnan(X)
    Xz = np.where(mask, X, 0.0)
    wcol = mask * w[:, None]
    wsum = wcol.sum(axis=0)
    if np.any(wsum <= 0):
        raise RiskError("ewma_cov: a column has no valid observations")
    mean = (wcol * Xz).sum(axis=0) / wsum
    D = np.where(mask, X - mean[None, :], 0.0)  # centred, NaNs -> 0 contribution

    # Weighted cross-products; pairwise weight normalization keeps it well-defined
    # even with the (rare) ragged NaN. C_ij = Σ w D_i D_j / Σ w (over rows both present).
    WD = D * w[:, None]
    num = WD.T @ D
    pair_w = (mask.astype(float) * w[:, None]).T @ mask.astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        C = num / pair_w
    C = np.nan_to_num(C, nan=0.0)

    if shrink:
        diag = np.diag(np.diag(C))
        C = (1.0 - shrink) * C + shrink * diag
    C = 0.5 * (C + C.T)
    C[np.diag_indices_from(C)] += _JITTER
    return pd.DataFrame(C, index=cols, columns=cols)
