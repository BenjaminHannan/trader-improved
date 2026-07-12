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

Estimation-quality floor (``min_history_days``): ``research/wiki/log.md``
2026-07-11 ("The illiquid/drift incident" and "Iterations 6-7 + the MI incident")
traced the French momentum validation sitting at 0.565-0.581 against the 0.6 gate
to a VERIFIED benign cause, not a data defect: roughly 87 alpaca-only delisted-name
fragments (real FRC/SIVB-class collapse histories) enter the equity factor-return
cross-sections mid-sample. Those fragments are short and ramping — a handful of
days of price history — and a name with almost no trailing history contributes a
noisy row (unstable trailing vol weight, unstable beta/momentum/size exposures) to
the WLS at exactly the dates it first appears. Dropping the fragments from the lake
would reintroduce survivorship bias for every OTHER estimate that benefits from
them (yfinance-only, no fragments, scores 0.635), so the fix is not exclusion from
the lake but exclusion from the regression until an instrument has earned enough
trailing observations to estimate reliably: a name only enters a given date's
cross-section once it has at least ``min_history_days`` prior price observations
in the panel (trailing-only, counted strictly before the regression date — never a
look-ahead). Names that never reach the floor simply never enter; the regression
proceeds with whatever names remain, subject to the existing under-determined-day
guard below.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from production.core.config import risk_config
from production.risk.exposures import build_exposures

_VOL_WINDOW = 63
_EPS = 1e-8

# Shipped default per research/wiki/log.md 2026-07-11 — see module docstring.
_DEFAULT_MIN_HISTORY_DAYS = 126


def _wide_close(prices: pd.DataFrame, ids, end) -> pd.DataFrame:
    end = pd.Timestamp(end)
    df = prices[prices["instrument_id"].isin(list(ids))]
    df = df[pd.to_datetime(df["obs_date"]) <= end]
    if df.empty:
        return pd.DataFrame(columns=list(ids))
    wide = df.pivot_table(index="obs_date", columns="instrument_id",
                          values="close", aggfunc="last").sort_index()
    return wide.reindex(columns=list(ids))


def _prior_obs_count(close: pd.DataFrame) -> pd.DataFrame:
    """Trailing, PIT-clean count of non-null price observations per instrument.

    Row ``d`` holds, per column, the number of non-null ``close`` observations
    at rows strictly before ``d`` — never counts day ``d`` itself, so it can be
    compared against ``min_history_days`` without a look-ahead.
    """
    return close.notna().cumsum().shift(1).fillna(0).astype(int)


def estimate_factor_returns(prices: pd.DataFrame, sleeve: str, start, end,
                            cfg: dict | None = None, sectors: pd.Series | None = None,
                            refit: str = "M",
                            min_history_days: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Estimate daily factor returns and residuals over ``[start, end]``.

    Returns ``(factor_returns, residuals)``: factor_returns is dates x K (the
    exposure columns), residuals is dates x N (the instruments in ``ids``).
    ``ids`` is taken as every instrument present for ``sleeve`` in ``prices``.

    ``min_history_days``: an instrument only enters a given date's
    cross-sectional regression once it has at least this many prior price
    observations in the panel (trailing-only, counted strictly before that
    date — see ``_prior_obs_count``). ``None`` (the default) draws from
    ``cfg["factor_returns"]["min_history_days"]`` if present, else falls back
    to the shipped default ``_DEFAULT_MIN_HISTORY_DAYS`` (126) — see the module
    docstring for the fragment-ramp rationale. Pass ``0`` to disable the floor
    entirely.
    """
    if cfg is None:
        cfg = risk_config()
    if min_history_days is None:
        min_history_days = int(
            cfg.get("factor_returns", {}).get("min_history_days", _DEFAULT_MIN_HISTORY_DAYS))
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)

    ids = sorted(prices["instrument_id"].unique().tolist())
    close = _wide_close(prices, ids, end)
    rets = close.pct_change()
    trail_vol = rets.rolling(_VOL_WINDOW, min_periods=max(2, _VOL_WINDOW // 2)).std()
    prior_obs = _prior_obs_count(close)

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
            h_d = prior_obs.loc[d].reindex(ids).to_numpy(dtype=int)
            valid = np.isfinite(r_d) & np.isfinite(s_d) & (s_d > 0) & \
                np.isfinite(Bvals).all(axis=1) & (h_d >= min_history_days)
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
