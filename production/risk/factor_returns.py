"""Cross-sectional factor-return estimation (Fama-MacBeth style WLS).

At each month start we refit the exposure matrix using data STRICTLY BEFORE that
date, then hold it fixed through the month. On each day d we regress that day's
cross-section of instrument returns on the (prior) exposures via weighted least
squares — weights ``1/sigma_i^2`` with ``sigma_i`` the trailing 63d vol as of the
refit date — to back out the day's factor returns and residuals.

PIT discipline: exposures use ``obs_date < refit_date`` only, weights use vol as
of the refit date, and the day-d return is the realized cross-section — nothing
in the estimation of factor return f_d uses information after day d except that
the exposures were frozen before the month began (a look-BACK, never a look-ahead).
No statsmodels: we solve the weighted normal equations with ``numpy.linalg.lstsq``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from production.core.config import risk_config
from production.risk.exposures import build_exposures

_VOL_WINDOW = 63
_EPS = 1e-8


def _wide_close(prices: pd.DataFrame, ids, end) -> pd.DataFrame:
    end = pd.Timestamp(end)
    df = prices[prices["instrument_id"].isin(list(ids))]
    df = df[pd.to_datetime(df["obs_date"]) <= end]
    if df.empty:
        return pd.DataFrame(columns=list(ids))
    wide = df.pivot_table(index="obs_date", columns="instrument_id",
                          values="close", aggfunc="last").sort_index()
    return wide.reindex(columns=list(ids))


def estimate_factor_returns(prices: pd.DataFrame, sleeve: str, start, end,
                            cfg: dict | None = None, sectors: pd.Series | None = None,
                            refit: str = "M") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Estimate daily factor returns and residuals over ``[start, end]``.

    Returns ``(factor_returns, residuals)``: factor_returns is dates x K (the
    exposure columns), residuals is dates x N (the instruments in ``ids``).
    ``ids`` is taken as every instrument present for ``sleeve`` in ``prices``.
    """
    if cfg is None:
        cfg = risk_config()
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)

    ids = sorted(prices["instrument_id"].unique().tolist())
    close = _wide_close(prices, ids, end)
    rets = close.pct_change()
    trail_vol = rets.rolling(_VOL_WINDOW, min_periods=max(2, _VOL_WINDOW // 2)).std()

    all_dates = rets.index
    in_window = all_dates[(all_dates >= start) & (all_dates <= end)]
    if len(in_window) == 0:
        return pd.DataFrame(), pd.DataFrame()

    periods = in_window.to_period(refit)
    fr_rows: dict[pd.Timestamp, pd.Series] = {}
    resid_rows: dict[pd.Timestamp, pd.Series] = {}

    for period in pd.unique(periods):
        pmask = periods == period
        pdates = in_window[pmask]
        refit_date = pdates[0]
        # Exposures from data strictly before the refit date.
        as_of_before = refit_date - pd.Timedelta(nanoseconds=1)
        B = build_exposures(prices, sleeve, as_of_before, ids, cfg, sectors)

        # Weights: 1/sigma^2 with sigma = trailing 63d vol as of the refit date
        # (last vol observation strictly before the refit date).
        prior_vol = trail_vol[trail_vol.index < refit_date]
        sigma = prior_vol.iloc[-1] if len(prior_vol) else pd.Series(index=ids, dtype=float)
        sigma = sigma.reindex(ids)

        Bvals = B.reindex(ids).to_numpy(dtype=float)
        cols = list(B.columns)

        for d in pdates:
            r_d = rets.loc[d].reindex(ids).to_numpy(dtype=float)
            s_d = sigma.to_numpy(dtype=float)
            valid = np.isfinite(r_d) & np.isfinite(s_d) & (s_d > 0) & \
                np.isfinite(Bvals).all(axis=1)
            if valid.sum() <= Bvals.shape[1]:
                continue  # under-determined cross-section on this day
            sw = 1.0 / np.maximum(s_d[valid], _EPS)  # sqrt weight = 1/sigma
            Aw = Bvals[valid] * sw[:, None]
            bw = r_d[valid] * sw
            f, *_ = np.linalg.lstsq(Aw, bw, rcond=None)
            fr_rows[d] = pd.Series(f, index=cols)
            resid_full = pd.Series(np.nan, index=ids)
            resid_full.iloc[np.where(valid)[0]] = r_d[valid] - Bvals[valid] @ f
            resid_rows[d] = resid_full

    if not fr_rows:
        return pd.DataFrame(), pd.DataFrame()
    factor_returns = pd.DataFrame(fr_rows).T.sort_index()
    residuals = pd.DataFrame(resid_rows).T.sort_index().reindex(columns=ids)
    return factor_returns, residuals
