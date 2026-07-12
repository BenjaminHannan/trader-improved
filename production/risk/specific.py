"""Specific (idiosyncratic) risk from factor-model residuals.

For each name we take the EWMA volatility of its regression residuals, then shrink
cross-sectionally toward the sleeve median. Thin histories are the motivation:
a name with only a few weeks of residuals has a noisy own-vol estimate, so we pull
it a quarter of the way to the cohort median (a light Bayesian shrink).

PIT note: pure estimator over the residual rows the caller supplies. Residuals
themselves are produced from ``obs_date <= as_of`` data upstream.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def specific_vol(residuals: pd.DataFrame, halflife: float = 42,
                 shrink_weight: float = 0.25, min_obs: int = 63) -> pd.Series:
    """EWMA daily vol per name, cross-sectionally shrunk toward the median.

    ``residuals`` is dates x instruments. Returns a Series (index=instrument) of
    daily volatilities: ``(1-w)*v_i + w*median(v)`` with ``w = shrink_weight``.

    ``min_obs`` sets the ``min_periods`` for the EWMA — a name with fewer non-NaN
    residuals yields NaN own-vol, which is then replaced by the cohort median so
    it contributes a sensible (fully-shrunk) risk rather than dropping out.
    """
    if residuals.shape[1] == 0:
        return pd.Series(dtype=float)
    r = residuals.sort_index()
    # EWMA variance -> vol. min_periods guards against near-empty columns.
    var = r.ewm(halflife=halflife, min_periods=min(min_obs, len(r))).var()
    v = np.sqrt(var.iloc[-1])

    med = v.median(skipna=True)
    if pd.isna(med):
        med = 0.0
    v = v.fillna(med)  # names without enough history default to the cohort median
    out = (1.0 - shrink_weight) * v + shrink_weight * med
    out.name = "specific_vol"
    return out
