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


def ledoit_wolf_shrinkage(
    returns: pd.DataFrame,
    target: str = "diagonal",
    ewma_halflife: float | None = None,
    min_obs: int = 60,
) -> tuple[pd.DataFrame, float]:
    """Ledoit-Wolf linear-shrinkage covariance with an analytic intensity.

    ``Sigma = delta* F + (1 - delta*) S`` where ``S`` is the (optionally
    EWMA-weighted) sample covariance and ``F`` is a structured shrinkage target:

    - ``target="diagonal"``    -> ``F = diag(S)`` (off-diagonals shrunk toward 0);
    - ``target="constant_correlation"`` -> Ledoit & Wolf (2003) "honey" target:
      ``r_bar`` = mean pairwise sample correlation, ``f_ij = r_bar*sqrt(s_ii*s_jj)``,
      ``f_ii = s_ii``.

    The intensity is ``delta* = clip(kappa / T_eff, 0, 1)`` with
    ``kappa = (pi_hat - rho_hat) / gamma_hat`` (standard LW component estimators):

    - ``pi_hat`` — sum over all entries of the asymptotic variances of ``S``;
    - ``rho_hat`` — asymptotic covariance between ``F`` and ``S``. For the diagonal
      target only the diagonal ``pi_ii`` terms survive (``F`` off-diagonals are a
      constant 0, independent of ``S``). For the constant-correlation target the
      off-diagonal ``r_bar/2`` correction with the ``theta`` cross-moment terms is
      added on top of the diagonal ``pi_ii``;
    - ``gamma_hat`` — squared Frobenius distance ``||F - S||^2``.

    Under EWMA weights the effective sample size is Kish's
    ``T_eff = (sum w)^2 / sum(w^2)`` (materially smaller than the raw window, so
    EWMA estimators warrant more shrinkage); with equal weights ``T_eff = T``.
    A degenerate target (``gamma_hat ~ 0``, i.e. ``F`` already equals ``S``) yields
    ``delta = 1``.

    Demeaning uses the SAME weights as the moments. Returns ``(Sigma_df, delta)``
    with ``Sigma`` symmetrized and tiny-diagonal-jittered exactly like
    :func:`ewma_cov`.
    """
    if returns.shape[1] == 0:
        raise RiskError("ledoit_wolf_shrinkage: no columns to estimate")
    if target not in ("diagonal", "constant_correlation"):
        raise RiskError(f"ledoit_wolf_shrinkage: unknown target {target!r}")

    r = returns.dropna(how="all").sort_index()
    n = len(r)
    if n < min_obs:
        raise RiskError(
            f"ledoit_wolf_shrinkage: {n} observations < min_obs={min_obs}")

    cols = list(r.columns)
    X = r.to_numpy(dtype=float)
    if ewma_halflife is not None:
        w = _ewma_weights(n, float(ewma_halflife))
    else:
        w = np.ones(n, dtype=float)

    # Weighted mean with per-column masked-weight renormalization (as ewma_cov).
    mask = ~np.isnan(X)
    M = mask.astype(float)
    Xz = np.where(mask, X, 0.0)
    wcol = M * w[:, None]
    wsum = wcol.sum(axis=0)
    if np.any(wsum <= 0):
        raise RiskError("ledoit_wolf_shrinkage: a column has no valid observations")
    mean = (wcol * Xz).sum(axis=0) / wsum
    # Centred returns; missing entries contribute 0 to every moment below.
    Y = np.where(mask, X - mean[None, :], 0.0)

    # Pairwise weight totals: pair_w[i,j] = sum_t w_t * present_it * present_jt.
    Mw = M * w[:, None]
    pair_w = Mw.T @ M
    if np.any(pair_w <= 0):
        raise RiskError("ledoit_wolf_shrinkage: a pair has no overlapping obs")

    Yw = Y * w[:, None]
    A = Yw.T @ Y            # A[i,j] = sum_t w_t Y_it Y_jt  (numerator of S)
    S = A / pair_w          # weighted sample covariance (1/T_eff-style scaling)
    s = np.diag(S).copy()
    if np.any(s <= 0):
        raise RiskError("ledoit_wolf_shrinkage: non-positive sample variance")

    # Kish effective sample size under the weights actually used.
    if ewma_halflife is not None:
        t_eff = float(w.sum() ** 2 / np.square(w).sum())
    else:
        t_eff = float(n)

    # --- pi_hat: sum_t w_t (Y_it Y_jt - S_ij)^2 / pair_w  (closed form).
    Y2 = Y * Y
    pi_mat = (Y2 * w[:, None]).T @ Y2 / pair_w - S * S
    pi_hat = float(pi_mat.sum())

    # --- gamma_hat and target F.
    std = np.sqrt(s)
    if target == "diagonal":
        F = np.diag(s)
        rho_hat = float(np.trace(pi_mat))  # diagonal pi_ii only
    else:  # constant_correlation
        corr = S / np.outer(std, std)
        off = ~np.eye(len(cols), dtype=bool)
        r_bar = float(corr[off].mean()) if off.any() else 0.0
        F = r_bar * np.outer(std, std)
        np.fill_diagonal(F, s)

        # theta_ii_ij = sum_t w (Y_it^2 - s_ii)(Y_it Y_jt - s_ij) / pair_w.
        Y3 = Y2 * Y
        P3 = (Y3 * w[:, None]).T @ Y          # P3[i,j] = sum_t w Y_it^3 Y_jt
        Q = (Y2 * w[:, None]).T @ M           # Q[i,j]  = sum_t w Y_it^2 present_jt
        theta_ii = (P3 - S * Q - s[:, None] * A
                    + (s[:, None] * S) * pair_w) / pair_w
        # theta_jj_ij = sum_t w (Y_jt^2 - s_jj)(Y_it Y_jt - s_ij) / pair_w.
        theta_jj = (P3.T - S * Q.T - s[None, :] * A
                    + (s[None, :] * S) * pair_w) / pair_w
        ratio = np.outer(std, 1.0 / std)      # ratio[i,j] = sqrt(s_ii/s_jj)
        rho_off = (r_bar / 2.0) * (ratio.T * theta_ii + ratio * theta_jj)
        np.fill_diagonal(rho_off, 0.0)
        rho_hat = float(np.trace(pi_mat) + rho_off.sum())

    diff = F - S
    gamma_hat = float(np.sum(diff * diff))

    if not np.isfinite(gamma_hat) or gamma_hat <= 1e-30:
        delta = 1.0
    else:
        kappa = (pi_hat - rho_hat) / gamma_hat
        delta = float(np.clip(kappa / t_eff, 0.0, 1.0))

    Sigma = delta * F + (1.0 - delta) * S
    Sigma = 0.5 * (Sigma + Sigma.T)
    Sigma[np.diag_indices_from(Sigma)] += _JITTER
    return pd.DataFrame(Sigma, index=cols, columns=cols), delta
