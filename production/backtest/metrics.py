"""Return-series performance metrics.

All functions take a daily return ``pd.Series`` (simple returns, one row per day) and
are pure — no state, no I/O. ``freq`` is the annualization factor (252 trading days).
NaNs are dropped up front so a structural gap (a sleeve that has not started trading)
never poisons a statistic; an empty series yields ``nan`` rather than raising.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _clean(r) -> pd.Series:
    return pd.Series(r).astype(float).dropna()


def sharpe(r, freq: int = 252) -> float:
    """Annualized Sharpe ratio ``mean/std * sqrt(freq)`` (sample std, ddof=1)."""
    s = _clean(r)
    if len(s) < 2:
        return float("nan")
    sd = s.std(ddof=1)
    if sd == 0 or not np.isfinite(sd):
        return float("nan")
    return float(s.mean() / sd * np.sqrt(freq))


def ann_return(r, freq: int = 252) -> float:
    """Geometric annualized return: ``prod(1+r) ** (freq/n) - 1``."""
    s = _clean(r)
    if len(s) == 0:
        return float("nan")
    growth = float((1.0 + s).prod())
    if growth <= 0:  # a wipe-out; annualization of a non-positive terminal is undefined
        return -1.0
    return float(growth ** (freq / len(s)) - 1.0)


def ann_vol(r, freq: int = 252) -> float:
    """Annualized volatility ``std * sqrt(freq)`` (sample std, ddof=1)."""
    s = _clean(r)
    if len(s) < 2:
        return float("nan")
    return float(s.std(ddof=1) * np.sqrt(freq))


def max_drawdown(r) -> float:
    """Maximum peak-to-trough drawdown of the compounded equity curve.

    Returned as a POSITIVE fraction (0.2 == a 20% drawdown). A flat/empty series -> 0.
    """
    s = _clean(r)
    if len(s) == 0:
        return 0.0
    equity = (1.0 + s).cumprod()
    peak = equity.cummax()
    dd = 1.0 - equity / peak
    return float(dd.max())


def hit_rate(r) -> float:
    """Fraction of strictly-positive return days."""
    s = _clean(r)
    if len(s) == 0:
        return float("nan")
    return float((s > 0).mean())


def information_ratio(r, bench) -> float:
    """Annualized IR of active return ``r - bench`` (Sharpe of the active series)."""
    a = pd.Series(r).astype(float)
    b = pd.Series(bench).astype(float)
    active = (a - b).dropna()
    return sharpe(active)


def turnover(weights_history: pd.DataFrame, freq: int = 252) -> float:
    """Average per-rebalance one-way turnover ``mean(sum_i |w_t - w_{t-1}|)``, annualized.

    ``weights_history`` is a rebalance-dated DataFrame (index = rebalance dates, columns
    = instrument ids). The per-rebalance turnover is annualized by the rebalance
    frequency inferred from the median spacing of the index (a weekly grid -> ~52x).
    """
    if weights_history is None or len(weights_history) < 2:
        return 0.0
    W = weights_history.fillna(0.0)
    dw = W.diff().abs().sum(axis=1)
    per_rebalance = float(dw.iloc[1:].mean())
    idx = pd.DatetimeIndex(W.index)
    gaps = idx.to_series().diff().dropna().dt.days
    med_gap = float(gaps.median()) if len(gaps) else 7.0
    ann_factor = freq / med_gap if med_gap > 0 else 1.0
    return per_rebalance * ann_factor
