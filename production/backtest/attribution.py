"""Return attribution — systematic (risk-factor) and per-alpha decomposition.

Two complementary views:

- ``factor_attribution`` regresses realized portfolio returns on the risk model's
  estimated factor returns (no intercept) and reports each factor's annualized return
  contribution ``beta_f * mean(factor_f) * 252`` plus a ``specific`` residual term
  (``mean(resid) * 252``) — the part of the P&L the risk factors do not explain.

- ``per_alpha_contribution`` estimates how much each *alpha factor* contributed, by
  holding that factor's single-factor target weights through each rebalance and dotting
  with realized instrument returns. This is an APPROXIMATION: the true joint optimizer
  couples factors through the risk penalty and constraints, so the single-factor books
  do not sum exactly to the combined book. It is a directional attribution, not an exact
  P&L split — documented as such.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_TRADING_DAYS = 252


def factor_attribution(port_returns: pd.Series, factor_returns: pd.DataFrame) -> dict:
    """Regress portfolio returns on risk-factor returns; annualized contributions.

    Returns a dict:
        ``contributions`` : {factor: beta_f * mean(factor_f) * 252}
        ``betas``         : {factor: OLS loading}
        ``specific``      : mean(residual) * 252
        ``r_squared``     : regression R^2

    The regression has no intercept, so any constant alpha lands in ``specific``.
    Aligned on the intersection of dates; empty overlap -> zeros.
    """
    y = pd.Series(port_returns).astype(float)
    X = pd.DataFrame(factor_returns).astype(float)
    if X.empty or X.shape[1] == 0:
        # an empty/rank-deficient estimation (e.g. cross-section thinner than the
        # exposure count every day) must degrade to zeros, not crash the report
        return {"contributions": {}, "betas": {},
                "specific": float(y.mean() * 252) if len(y) else 0.0,
                "r_squared": float("nan")}
    joined = pd.concat([y.rename("_y"), X], axis=1, join="inner").dropna()
    factors = list(X.columns)
    if len(joined) <= len(factors) or not factors:
        return {"contributions": {f: 0.0 for f in factors},
                "betas": {f: 0.0 for f in factors},
                "specific": 0.0, "r_squared": float("nan")}

    yv = joined["_y"].to_numpy()
    Xv = joined[factors].to_numpy()
    beta, *_ = np.linalg.lstsq(Xv, yv, rcond=None)
    resid = yv - Xv @ beta
    means = Xv.mean(axis=0)

    contributions = {f: float(beta[i] * means[i] * _TRADING_DAYS)
                     for i, f in enumerate(factors)}
    betas = {f: float(beta[i]) for i, f in enumerate(factors)}
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((yv - yv.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"contributions": contributions, "betas": betas,
            "specific": float(resid.mean() * _TRADING_DAYS), "r_squared": r2}


def per_alpha_contribution(weights_by_factor: dict[str, pd.DataFrame],
                           inst_returns: pd.DataFrame) -> dict:
    """Approximate annualized P&L contribution of each alpha factor.

    ``weights_by_factor[f]`` is a rebalance-dated DataFrame (index = rebalance dates,
    columns = instrument ids) of that factor's stand-alone target weights. Each is
    forward-filled to daily and lagged one day (positions earn from t+1), multiplied by
    ``inst_returns`` (daily dates x ids), and summed to a daily contribution series whose
    annualized mean is reported. See module docstring for the approximation caveat.
    """
    out: dict[str, float] = {}
    daily_idx = inst_returns.index
    for factor, W in weights_by_factor.items():
        if W is None or W.empty:
            out[factor] = 0.0
            continue
        cols = [c for c in W.columns if c in inst_returns.columns]
        if not cols:
            out[factor] = 0.0
            continue
        w_daily = (W[cols].reindex(daily_idx, method="ffill").shift(1).fillna(0.0))
        contrib = (w_daily * inst_returns[cols].fillna(0.0)).sum(axis=1)
        out[factor] = float(contrib.mean() * _TRADING_DAYS)
    return out
