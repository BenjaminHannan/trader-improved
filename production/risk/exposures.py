"""Point-in-time factor exposures (the ``B`` matrix) for one sleeve.

Columns: ``market`` (raw beta), ``size``, ``momentum``, ``vol`` — the style
factors are z-scored cross-sectionally; ``market`` is left as the raw beta so it
carries an interpretable unit. Sector dummies are appended only when a ``sectors``
Series is supplied (equity sleeve; current GICS labels — deliberately NOT PIT, a
documented v1 caveat in configs/risk.yaml).

PIT discipline (enforced by the corruption test):
- Only rows with ``obs_date <= as_of`` are ever read. The caller passes
  ``as_of = decision date``; every trailing statistic (beta window, 63d ADV,
  63d vol, 12-1 momentum) therefore ends at ``as_of`` and feeds a same-day
  decision. The ADV-weighted sleeve-index weights are ``shift(1)``-ed so the
  index return on day d uses cap-proxy weights known as of d-1, never same-day
  dollar volume; the beta window is the last 252 return observations ending at
  ``as_of``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from production.core.config import risk_config

_BETA_MIN_OBS = 126
_BETA_DEFAULT = 1.0


def _wide(prices: pd.DataFrame, as_of, ids, value_col: str) -> pd.DataFrame:
    """Strictly-PIT wide pivot (index=obs_date, cols=instrument_id) of ``value_col``.

    Only ``obs_date <= as_of`` rows and only the requested ``ids`` are kept.
    """
    as_of = pd.Timestamp(as_of)
    df = prices[prices["instrument_id"].isin(list(ids))]
    obs = pd.to_datetime(df["obs_date"])
    df = df[obs <= as_of]
    if df.empty:
        return pd.DataFrame(columns=list(ids))
    wide = df.pivot_table(index="obs_date", columns="instrument_id",
                          values=value_col, aggfunc="last").sort_index()
    # Preserve caller id order / include ids absent from the panel as all-NaN.
    return wide.reindex(columns=list(ids))


def _zscore(s: pd.Series) -> pd.Series:
    """Cross-sectional z-score with a zero-std guard; NaN -> 0 after standardizing."""
    x = s.astype(float)
    mu = x.mean(skipna=True)
    sd = x.std(skipna=True, ddof=0)
    if not np.isfinite(sd) or sd == 0:
        z = x - mu  # all-equal (or single name) -> all zeros
    else:
        z = (x - mu) / sd
    return z.fillna(0.0)


def _sleeve_index_returns(rets: pd.DataFrame, dollar_volume: pd.DataFrame,
                          adv_window: int) -> pd.Series:
    """ADV-weighted sleeve index daily return.

    Weights ∝ trailing ``adv_window``-day mean dollar volume (a cap proxy — no
    free PIT market cap), renormalized each day, then ``shift(1)`` so day-d index
    return uses weights knowable at d-1. Returns a Series indexed by date.
    """
    adv = dollar_volume.rolling(adv_window, min_periods=1).mean()
    wsum = adv.sum(axis=1)
    weights = adv.div(wsum.where(wsum > 0, np.nan), axis=0)
    weights = weights.shift(1)  # decision-day weights come from the prior day
    idx = (weights * rets).sum(axis=1, min_count=1)
    return idx


def _beta(inst_ret: pd.Series, idx_ret: pd.Series, window: int) -> float:
    """OLS slope of instrument returns on the sleeve index over the last ``window``
    joint observations; ``_BETA_DEFAULT`` when fewer than ``_BETA_MIN_OBS`` remain."""
    df = pd.concat([inst_ret, idx_ret], axis=1, keys=["y", "x"]).dropna()
    if len(df) > window:
        df = df.iloc[-window:]
    if len(df) < _BETA_MIN_OBS:
        return _BETA_DEFAULT
    x = df["x"].to_numpy()
    y = df["y"].to_numpy()
    var = x.var()
    if var == 0:
        return _BETA_DEFAULT
    return float(np.cov(y, x, ddof=0)[0, 1] / var)


def build_exposures(prices: pd.DataFrame, sleeve: str, as_of, ids,
                    cfg: dict | None = None, sectors: pd.Series | None = None) -> pd.DataFrame:
    """Build the point-in-time exposure matrix ``B`` for ``ids`` in ``sleeve``.

    index = instrument_id (in ``ids`` order); columns = [market, size, momentum,
    vol] (+ ``sector_<label>`` dummies if ``sectors`` given). See module docstring
    for the PIT contract.
    """
    if cfg is None:
        cfg = risk_config()
    ids = list(ids)
    scfg = cfg.get("exposures", {}).get(sleeve, {})
    beta_window = int(scfg.get("beta_window_days", 252))
    adv_window = int(scfg.get("size_adv_window_days", 63))
    vol_window = int(scfg.get("vol_window_days", 63))

    close = _wide(prices, as_of, ids, "close")
    dollar_volume = _wide(prices, as_of, ids, "dollar_volume")
    rets = close.pct_change()

    idx_ret = _sleeve_index_returns(rets, dollar_volume, adv_window)

    # market beta (raw), per instrument, over the trailing window ending at as_of.
    market = pd.Series(
        {iid: _beta(rets[iid], idx_ret, beta_window) if iid in rets else _BETA_DEFAULT
         for iid in ids},
        name="market",
    )

    # size = log trailing adv-window mean dollar volume, evaluated at as_of.
    adv = dollar_volume.rolling(adv_window, min_periods=1).mean()
    last_adv = adv.iloc[-1] if len(adv) else pd.Series(index=ids, dtype=float)
    size_raw = np.log(last_adv.reindex(ids).where(last_adv.reindex(ids) > 0))

    # momentum = 12-1 return (close[t-21]/close[t-252] - 1), evaluated at as_of.
    mom_raw = (close.shift(21) / close.shift(252) - 1.0)
    mom_raw = mom_raw.iloc[-1].reindex(ids) if len(mom_raw) else pd.Series(index=ids, dtype=float)

    # vol = trailing vol-window std of daily returns, evaluated at as_of.
    vol_raw = rets.rolling(vol_window, min_periods=max(2, vol_window // 2)).std()
    vol_raw = vol_raw.iloc[-1].reindex(ids) if len(vol_raw) else pd.Series(index=ids, dtype=float)

    out = pd.DataFrame(index=pd.Index(ids, name="instrument_id"))
    out["market"] = market
    out["size"] = _zscore(size_raw)
    out["momentum"] = _zscore(mom_raw)
    out["vol"] = _zscore(vol_raw)

    if sectors is not None:
        labels = pd.Series(sectors).reindex(ids)
        dummies = pd.get_dummies(labels, prefix="sector", dtype=float)
        dummies.index = out.index
        out = pd.concat([out, dummies], axis=1)

    return out
