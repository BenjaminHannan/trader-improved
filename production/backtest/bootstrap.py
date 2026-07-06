"""Stationary bootstrap (Politis & Romano, 1994) confidence intervals for the Sharpe.

A single realized Sharpe is a point estimate off one path; its sampling error is large
and — because daily returns are serially dependent (vol clustering, momentum) — an i.i.d.
bootstrap understates it. The stationary bootstrap resamples *blocks* of geometrically
random length (mean ``avg_block``), wrapping circularly, which preserves short-range
dependence while keeping the resampled series stationary. Recomputing the Sharpe on each
resample traces out its sampling distribution, from which a percentile CI is read off.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from production.backtest.metrics import sharpe


def stationary_bootstrap(returns, n_boot: int = 1000, avg_block: int = 21,
                         seed: int = 0, freq: int = 252) -> np.ndarray:
    """Return an array of ``n_boot`` annualized Sharpes from stationary-bootstrap resamples.

    Block lengths are geometric with mean ``avg_block`` (restart probability
    ``p = 1/avg_block``); indices wrap circularly so every start point is valid.
    """
    r = pd.Series(returns).astype(float).dropna().to_numpy()
    n = len(r)
    if n < 2:
        return np.array([])
    rng = np.random.default_rng(seed)
    p = 1.0 / max(avg_block, 1)
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
              avg_block: int = 21, seed: int = 0, freq: int = 252) -> tuple[float, float]:
    """Percentile ``level`` confidence interval for the annualized Sharpe.

    Returns ``(lo, hi)``; an all-NaN or too-short series yields ``(nan, nan)``.
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
