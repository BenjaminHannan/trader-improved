"""Probabilistic and Deflated Sharpe ratios (Bailey & Lopez de Prado, 2014).

The problem these guard against: after trying ``n_trials`` factor configurations and
keeping the best backtest Sharpe, that Sharpe is upward-biased by selection alone. The
Deflated Sharpe deflates the observed Sharpe by the Sharpe you would *expect* to see as
the maximum of ``n_trials`` independent trials, then reports the probability the deflated
edge is real given the sample's own non-normality (skew and kurtosis).

All Sharpes here are in PER-OBSERVATION units (mean/std of the return series, NOT
annualized). ``n`` is the number of observations. The report layer passes the
non-annualized Sharpe and ``n = len(returns)`` so the algebra is self-consistent.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

_EULER_GAMMA = 0.5772156649015329  # Euler-Mascheroni constant
_E = np.e


def probabilistic_sharpe(sr: float, sr_star: float, n: int,
                         skew: float, kurt: float) -> float:
    """PSR: probability the true Sharpe exceeds the benchmark ``sr_star``.

    ``PSR = Phi( ((sr - sr_star) * sqrt(n-1)) / sqrt(1 - skew*sr + ((kurt-1)/4)*sr^2) )``

    ``kurt`` is the (non-excess) kurtosis (3 for a normal). The denominator is the
    standard error of the Sharpe estimator under non-normality; its radicand is clipped
    to a tiny positive floor so a pathological (skew, kurt, sr) triple degrades to a
    saturated 0/1 answer instead of a NaN.
    """
    if not np.isfinite(sr) or n < 2:
        return float("nan")
    radicand = 1.0 - skew * sr + ((kurt - 1.0) / 4.0) * sr * sr
    radicand = max(radicand, 1e-12)
    se = np.sqrt(radicand)
    z = (sr - sr_star) * np.sqrt(n - 1) / se
    return float(norm.cdf(z))


def expected_max_sharpe(n_trials: int, var_trials_sr: float) -> float:
    """Expected maximum Sharpe across ``n_trials`` i.i.d. trials (the deflation target).

    ``sr_star = sqrt(var_trials_sr) * [ (1-gamma) * Z(1 - 1/n) + gamma * Z(1 - 1/(n*e)) ]``
    where ``Z = Phi^-1`` and ``gamma`` is the Euler-Mascheroni constant — the standard
    extreme-value approximation for the expected max of Gaussian trials. With a single
    trial (or non-positive trial variance) there is nothing to deflate: returns 0.
    """
    n = int(n_trials)
    if n <= 1 or not np.isfinite(var_trials_sr) or var_trials_sr <= 0:
        return 0.0
    z1 = norm.ppf(1.0 - 1.0 / n)
    z2 = norm.ppf(1.0 - 1.0 / (n * _E))
    return float(np.sqrt(var_trials_sr) * ((1.0 - _EULER_GAMMA) * z1 + _EULER_GAMMA * z2))


def deflated_sharpe(sr: float, n_trials: int, var_trials_sr: float,
                    n: int, skew: float, kurt: float) -> float:
    """Deflated Sharpe: PSR evaluated against the expected-max-of-trials benchmark.

    Deflates the observed (per-observation) Sharpe ``sr`` by ``expected_max_sharpe`` and
    returns the probability the deflated edge is genuine. Near 1 => the Sharpe survives
    the multiple-testing correction; near 0 => it is consistent with selection luck.
    """
    sr_star = expected_max_sharpe(n_trials, var_trials_sr)
    return probabilistic_sharpe(sr, sr_star, n, skew, kurt)
