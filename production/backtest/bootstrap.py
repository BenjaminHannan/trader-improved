"""Stationary bootstrap (Politis & Romano, 1994) confidence intervals for the Sharpe.

A single realized Sharpe is a point estimate off one path; its sampling error is large
and — because daily returns are serially dependent (vol clustering, momentum) — an i.i.d.
bootstrap understates it. The stationary bootstrap resamples *blocks* of geometrically
random length (mean ``avg_block``), wrapping circularly, which preserves short-range
dependence while keeping the resampled series stationary. Recomputing the Sharpe on each
resample traces out its sampling distribution, from which a percentile CI is read off.

The one tuning knob is the expected block length. ``avg_block="auto"`` selects it
data-drivenly via :func:`politis_white_block_length` (Politis & White 2004, corrected by
Patton, Politis & White 2009): too short understates autocorrelation (CIs too tight), too
long wastes power. Passing a number keeps the old fixed-block behaviour exactly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from production.backtest.metrics import sharpe


def _flat_top_lambda(x: np.ndarray) -> np.ndarray:
    """Politis flat-top lag window: 1 for |x|<=1/2, 2(1-|x|) for 1/2<|x|<=1, else 0."""
    ax = np.abs(x)
    w = np.zeros_like(ax, dtype=float)
    w[ax <= 0.5] = 1.0
    taper = (ax > 0.5) & (ax <= 1.0)
    w[taper] = 2.0 * (1.0 - ax[taper])
    return w


def politis_white_block_length(returns) -> float:
    """Data-driven optimal expected block length for the *stationary* bootstrap.

    Patton, Politis & White (2009) correction of Politis & White (2004):
    ``b_opt = (2 G^2 / D_SB)^(1/3) * T^(1/3)`` where, over lags ``|k| <= M`` with flat-top
    weights ``lambda(k/M)`` and sample autocovariances ``R(k)``,
    ``G = sum lambda(k/M)|k|R(k)`` and ``D_SB = 2 (sum lambda(k/M)R(k))^2``. The bandwidth
    ``M`` follows the last-significant-lag rule: the largest lag ``k <= K_n`` (with
    ``K_n = max(5, sqrt(log10 T))``) whose ``|rho(k)| > 2 sqrt(log10(T)/T)``, then
    ``M = min(2 k_hat, T-1)``. Result clipped to ``[1, ceil(T/3)]``; a degenerate
    (constant / too-short) series returns ``1.0``.
    """
    r = pd.Series(returns).astype(float).dropna().to_numpy()
    T = len(r)
    if T < 4:
        return 1.0
    x = r - r.mean()
    r0 = float(np.dot(x, x) / T)  # R(0) = sample variance (biased, /T)
    # Degenerate/constant guard, scale-relative so float round-off on a constant series
    # (variance ~ 1e-36 vs mean^2) does not masquerade as genuine dependence.
    scale = float(np.mean(r * r))
    if r0 <= 0.0 or (scale > 0.0 and r0 <= 1e-10 * scale):
        return 1.0

    # Sample autocovariances R(k), k = 0..T-1 (biased 1/T normalization).
    def autocov(k: int) -> float:
        return float(np.dot(x[: T - k], x[k:]) / T)

    log10T = np.log10(T)
    thresh = 2.0 * np.sqrt(log10T / T)
    kn = int(np.floor(max(5.0, np.sqrt(log10T))))
    kn = min(kn, T - 1)

    # last-significant-lag rule: largest k <= kn with |rho(k)| > threshold.
    k_hat = 0
    for k in range(1, kn + 1):
        if abs(autocov(k) / r0) > thresh:
            k_hat = k
    M = min(2 * k_hat, T - 1)

    if M <= 0:
        return 1.0

    ks = np.arange(1, M + 1)
    lam = _flat_top_lambda(ks / M)
    rk = np.array([autocov(int(k)) for k in ks])

    # symmetric sums over |k| <= M (R(-k) = R(k)); the k=0 term contributes 0 to G.
    G = 2.0 * float(np.sum(lam * ks * rk))
    inner = r0 + 2.0 * float(np.sum(lam * rk))
    D_SB = 2.0 * inner * inner
    if D_SB <= 0.0 or G <= 0.0:
        return 1.0

    b_opt = (2.0 * G * G / D_SB) ** (1.0 / 3.0) * T ** (1.0 / 3.0)
    if not np.isfinite(b_opt):
        return 1.0
    return float(np.clip(b_opt, 1.0, np.ceil(T / 3.0)))


def _resolve_block(avg_block, r: np.ndarray) -> float:
    """Turn ``avg_block`` into a numeric expected block length ('auto' -> PPW rule)."""
    if isinstance(avg_block, str):
        if avg_block.lower() != "auto":
            raise ValueError(f"avg_block must be a number or 'auto', got {avg_block!r}")
        return politis_white_block_length(r)
    return float(avg_block)


def stationary_bootstrap(returns, n_boot: int = 1000, avg_block: float | str = "auto",
                         seed: int = 0, freq: int = 252) -> np.ndarray:
    """Return an array of ``n_boot`` annualized Sharpes from stationary-bootstrap resamples.

    Block lengths are geometric with mean ``avg_block`` (restart probability
    ``p = 1/avg_block``); indices wrap circularly so every start point is valid.
    ``avg_block="auto"`` (the default) picks the mean block length via
    :func:`politis_white_block_length`; a number reproduces the fixed-block behaviour.
    """
    r = pd.Series(returns).astype(float).dropna().to_numpy()
    n = len(r)
    if n < 2:
        return np.array([])
    block = _resolve_block(avg_block, r)
    rng = np.random.default_rng(seed)
    p = 1.0 / max(block, 1.0)
    out = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = np.empty(n, dtype=int)
        i = rng.integers(0, n)
        for k in range(n):
            idx[k] = i
            if rng.random() < p:
                i = rng.integers(0, n)  # start a new block
            else:
                i = (i + 1) % n         # extend the current block (circular)
        out[b] = sharpe(r[idx], freq=freq)
    return out


def sharpe_ci(returns, level: float = 0.95, n_boot: int = 1000,
              avg_block: float | str = "auto", seed: int = 0,
              freq: int = 252) -> tuple[float, float]:
    """Percentile ``level`` confidence interval for the annualized Sharpe.

    Returns ``(lo, hi)``; an all-NaN or too-short series yields ``(nan, nan)``.
    ``avg_block="auto"`` (default) selects the block length data-drivenly; a number keeps
    the prior fixed-block CI unchanged.
    """
    boots = stationary_bootstrap(returns, n_boot=n_boot, avg_block=avg_block,
                                 seed=seed, freq=freq)
    boots = boots[np.isfinite(boots)]
    if len(boots) == 0:
        return (float("nan"), float("nan"))
    alpha = (1.0 - level) / 2.0
    lo = float(np.quantile(boots, alpha))
    hi = float(np.quantile(boots, 1.0 - alpha))
    return (lo, hi)
