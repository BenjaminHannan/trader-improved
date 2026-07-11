"""Point-in-time factor exposures (the ``B`` matrix) for one sleeve.

Columns: ``market`` (raw beta), ``size``, ``momentum``, ``vol`` — the style
factors are z-scored cross-sectionally; ``market`` is left as the raw beta so it
carries an interpretable unit. Sector dummies are appended only when a ``sectors``
Series is supplied (equity sleeve; current GICS labels — deliberately NOT PIT, a
documented v1 caveat in configs/risk.yaml).

Two additional, OPT-IN style factors (research/wiki/questions/research-risk-model-
validation.md Q5 — risk exposures only, no alpha-gate/n_trials contact):
- ``liquidity``: USE4-style turnover, ``log(trailing size_adv_window_days-day mean
  dollar ADV / market cap)`` with ``mcap = shares_outstanding(PIT) * close(as_of)``.
  Shares outstanding comes from the ``fundamentals`` panel via the exact same
  filing-``available_from`` step-function convention ``production/signals/value.py``
  uses for EPS (see ``_pit_shares_asof`` below, built on that module's
  ``_avail_step_by_instrument``).
- ``earnings_yield``: trailing-12m EPS / close(as_of), reusing
  ``production/signals/value.py``'s ``EarningsYield`` arithmetic verbatim
  (``_eps_vintages`` + ``EarningsYield._ttm``) — the same filing-date-gated,
  latest-vintage-wins TTM walk. Deliberately NOT sector-demeaned: as a risk
  exposure its job is to explain cross-sectional co-movement (including
  sector-driven co-movement), not to isolate a sector-neutral alpha signal —
  that the underlying EPS/price arithmetic was once evaluated (and rejected) as
  an *alpha* is irrelevant to its use here as a *risk* exposure.

Both new factors are gated on ``cfg["exposures"][sleeve]["factors"]`` naming
them explicitly — unlike ``market``/``size``/``momentum``/``vol``, which this
module has always built unconditionally regardless of that list. Since
``configs/risk.yaml`` does not list ``liquidity``/``earnings_yield`` today,
``build_exposures`` output is BIT-IDENTICAL to before this change for every
existing caller (model.py, factor_returns.py, backtest/engine.py), even when a
``fundamentals`` panel is supplied — see ``tests/test_exposures_q5.py``'s
config-off equality test. When requested but the ``fundamentals`` panel cannot
support the factor at all (absent/empty, missing the required field, or zero
overlap with ``ids``), the column is SKIPPED (not added, never NaN-filled) with
a ``warnings.warn`` — consistent with "never silently NaN-poison B". Partial
coverage (some but not all ``ids``) leaves those names NaN pre-zscore, which
``_zscore``'s existing NaN->0 fill absorbs exactly like any other factor's
missing-instrument case (e.g. a name too short-lived for the beta window).

PIT discipline (enforced by the corruption test):
- Only rows with ``obs_date <= as_of`` are ever read. The caller passes
  ``as_of = decision date``; every trailing statistic (beta window, 63d ADV,
  63d vol, 12-1 momentum) therefore ends at ``as_of`` and feeds a same-day
  decision. The ADV-weighted sleeve-index weights are ``shift(1)``-ed so the
  index return on day d uses cap-proxy weights known as of d-1, never same-day
  dollar volume; the beta window is the last 252 return observations ending at
  ``as_of``. ``liquidity``/``earnings_yield`` extend the same discipline to the
  fundamentals lake: a shares/EPS vintage is usable at ``as_of`` only once its
  filing's ``available_from`` (normalized to its UTC date) is ``<= as_of``.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from production.alpha.zscore import winsorize as _mad_winsorize
from production.core.config import risk_config
from production.signals.value import EarningsYield as _EarningsYieldAlpha
from production.signals.value import _avail_step_by_instrument, _eps_vintages

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


def _winsorize_then_zscore(raw: pd.Series) -> pd.Series:
    """Shared pipeline for the two Q5 factors: MAD-winsorize the finite values
    (``production/alpha/zscore.py``'s ``winsorize`` — the codebase's one existing
    winsorize convention; this module's four legacy factors were never
    winsorized, only z-scored, so there is nothing "existing" to match there),
    then this module's own ``_zscore``. NaNs pass through winsorization
    untouched (median/MAD computed over the finite subset only) and are handled
    by ``_zscore``'s NaN->0 fill same as every other factor.
    """
    x = raw.astype(float)
    finite = np.isfinite(x.to_numpy())
    if finite.any():
        clipped = x.to_numpy(copy=True)
        clipped[finite] = _mad_winsorize(clipped[finite])
        x = pd.Series(clipped, index=x.index)
    return _zscore(x)


def _pit_shares_asof(fundamentals: pd.DataFrame | None, as_of, ids: list) -> pd.Series | None:
    """PIT shares-outstanding per instrument at ``as_of``.

    Reuses ``production/signals/value.py``'s ``_avail_step_by_instrument`` — the
    same per-filing ``available_from``-normalized-to-date step function used
    there for mcap/TVL/EPS — applied to the fundamentals lake's ``field ==
    "shares"`` rows. Returns ``None`` (never a some-NaN Series) when the input
    cannot support the factor for ANY requested id: no panel, no ``shares``
    field, or zero id overlap after applying the ``as_of`` availability cutoff.
    A Series aligned to ``ids`` (NaN for uncovered names) otherwise.
    """
    if fundamentals is None or fundamentals.empty:
        return None
    if "field" not in fundamentals.columns:
        return None
    rows = fundamentals[fundamentals["field"] == "shares"]
    if rows.empty:
        return None
    steps = _avail_step_by_instrument(rows.rename(columns={"value": "shares"}), "shares")
    cutoff = np.datetime64(pd.Timestamp(as_of).normalize())
    out: dict = {}
    for iid in ids:
        st = steps.get(iid)
        if st is None or st.empty:
            continue
        mask = st["avail_date"].to_numpy() <= cutoff
        if mask.any():
            out[iid] = float(st["shares"].to_numpy()[mask][-1])
    if not out:
        return None
    return pd.Series(out, dtype=float).reindex(ids)


def _pit_ttm_eps_asof(fundamentals: pd.DataFrame | None, as_of, ids: list) -> pd.Series | None:
    """PIT trailing-12m EPS per instrument at ``as_of``.

    Reuses ``production/signals/value.py``'s ``EarningsYield`` arithmetic
    verbatim: ``_eps_vintages`` (per-instrument filing-vintage walk) feeding
    ``EarningsYield._ttm`` (latest-vintage-per-period, sum of the four most
    recent periods filed by ``as_of``, NaN until four quarters have filed).
    Returns ``None`` (never a some-NaN Series) when no id has a usable TTM value
    at ``as_of`` at all — absent/empty panel, no ``eps`` field, zero id overlap,
    or (for every id) fewer than four quarters filed by ``as_of``. A Series
    aligned to ``ids`` (NaN for uncovered names) otherwise.
    """
    if fundamentals is None or fundamentals.empty:
        return None
    if "field" not in fundamentals.columns or "eps" not in set(fundamentals["field"]):
        return None
    cutoff = np.datetime64(pd.Timestamp(as_of).normalize())
    out: dict = {}
    for iid in ids:
        ev = _eps_vintages(fundamentals, iid)
        if ev.empty:
            continue
        ttm = _EarningsYieldAlpha._ttm(ev, [cutoff])[0]
        if np.isfinite(ttm):
            out[iid] = float(ttm)
    if not out:
        return None
    return pd.Series(out, dtype=float).reindex(ids)


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
                    cfg: dict | None = None, sectors: pd.Series | None = None,
                    fundamentals: pd.DataFrame | None = None) -> pd.DataFrame:
    """Build the point-in-time exposure matrix ``B`` for ``ids`` in ``sleeve``.

    index = instrument_id (in ``ids`` order); columns = [market, size, momentum,
    vol] (+ ``sector_<label>`` dummies if ``sectors`` given) (+ ``liquidity`` /
    ``earnings_yield`` if named in ``cfg["exposures"][sleeve]["factors"]`` AND a
    usable ``fundamentals`` panel is supplied — see module docstring). ``cfg``
    is unchanged today (``configs/risk.yaml`` lists neither), so every existing
    caller (which never passes ``fundamentals`` either) gets byte-identical
    output to before this parameter existed. See module docstring for the PIT
    contract.
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

    # Q5 opt-in risk exposures — additive, config-gated (see module docstring).
    requested = list(scfg.get("factors", []))
    if "liquidity" in requested or "earnings_yield" in requested:
        last_close = close.iloc[-1].reindex(ids) if len(close) else pd.Series(index=ids, dtype=float)

        if "liquidity" in requested:
            shares = _pit_shares_asof(fundamentals, as_of, ids)
            if shares is None:
                warnings.warn(
                    f"build_exposures: 'liquidity' requested for sleeve {sleeve!r} at "
                    f"{as_of!r} but the fundamentals panel has no usable PIT "
                    "shares-outstanding coverage (missing/empty panel, no 'shares' "
                    "field, or zero id overlap as of this date) -- skipping the "
                    "factor rather than NaN-poisoning B.", stacklevel=2)
            else:
                mcap = shares.reindex(ids) * last_close
                adv_now = last_adv.reindex(ids)
                liq_raw = np.log((adv_now / mcap).where((mcap > 0) & (adv_now > 0)))
                out["liquidity"] = _winsorize_then_zscore(liq_raw)

        if "earnings_yield" in requested:
            ttm_eps = _pit_ttm_eps_asof(fundamentals, as_of, ids)
            if ttm_eps is None:
                warnings.warn(
                    f"build_exposures: 'earnings_yield' requested for sleeve {sleeve!r} "
                    f"at {as_of!r} but the fundamentals panel has no usable PIT "
                    "trailing-12m EPS coverage (missing/empty panel, no 'eps' field, "
                    "zero id overlap, or fewer than four quarters filed by this date "
                    "for every id) -- skipping the factor rather than NaN-poisoning B.",
                    stacklevel=2)
            else:
                ey_raw = (ttm_eps.reindex(ids) / last_close).where(last_close > 0)
                out["earnings_yield"] = _winsorize_then_zscore(ey_raw)

    if sectors is not None:
        labels = pd.Series(sectors).reindex(ids)
        dummies = pd.get_dummies(labels, prefix="sector", dtype=float)
        dummies.index = out.index
        out = pd.concat([out, dummies], axis=1)

    return out
