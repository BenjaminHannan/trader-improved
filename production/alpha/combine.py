"""Combine per-factor alpha panels into a single blended alpha.

Alphas from different factors cover overlapping but not identical instrument/date sets
(a crypto-only carry factor has no view on equities; a factor may lack history for a
recently listed name). The blend is an *outer* alignment: a name/date present in at
least one factor survives, and factors with no view there contribute 0. A cell that no
factor touches never appears — we never invent a zero alpha out of thin air.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from production.alpha.refine import refine_alpha


def combine_alphas(alpha_panels: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Outer-aligned sum of per-factor alpha panels.

    Each value in ``alpha_panels`` is a long ``[obs_date, instrument_id, value]`` panel.
    Returns the same long format, where each ``(obs_date, instrument_id)`` value is the
    sum over factors that have a value there (missing => treated as 0). The key set is
    the union across factors, so no all-missing cells are fabricated.
    """
    frames = [p[["obs_date", "instrument_id", "value"]]
              for p in alpha_panels.values() if p is not None and not p.empty]
    if not frames:
        return pd.DataFrame(columns=["obs_date", "instrument_id", "value"])
    stacked = pd.concat(frames, ignore_index=True)
    out = stacked.groupby(["obs_date", "instrument_id"], as_index=False)["value"].sum()
    return out


def factor_momentum_tilt(factor_returns_trailing: pd.Series, gamma: float) -> float:
    """Multiplicative factor-momentum tilt on a factor's IC weight.

    Ehsani-Linnainmaa (JF 2022) / Gupta-Kelly (JPM 2019): single-factor returns are
    positively autocorrelated — a factor that has been winning over the trailing year
    tends to keep winning. We express this as a bounded tilt on the factor's combination
    weight: ``clip(1 + gamma * sign(sum trailing returns), 0, 2)``. A factor with a
    positive trailing sum is scaled up (toward ``1 + gamma``), a persistently losing one
    scaled down (toward ``1 - gamma``); the clip keeps the tilt in ``[0, 2]`` so it can
    never flip a factor's sign or blow its weight up. ``gamma = 0`` returns exactly 1.0
    (feature off, bit-identical). An empty/all-NaN trailing series also returns 1.0 (no
    evidence -> no tilt).
    """
    if gamma == 0.0:
        return 1.0
    s = pd.Series(factor_returns_trailing).dropna()
    if s.empty:
        return 1.0
    sign = float(np.sign(float(s.sum())))
    return float(np.clip(1.0 + gamma * sign, 0.0, 2.0))


def single_factor_returns(z_panel: pd.DataFrame, prices: pd.DataFrame, sleeve_ids,
                          quantile: float = 0.2) -> pd.Series:
    """Weekly long-short top-minus-bottom-quintile return series for one factor/sleeve.

    On each date the factor emits scores, rank the sleeve's names cross-sectionally,
    go long the top ``quantile`` and short the bottom ``quantile`` (equal-weight, each
    leg summing to 1), hold with the engine's one-day implementation lag, and compound
    the daily long-short return into a weekly (``W-FRI``) series.

    PIT / positional: the weight formed from the score at date ``d`` earns the return of
    day ``d+1`` onward (``shift(1)``), never day ``d``'s own return — the same one-day
    effect lag the engine applies to live weights. A weekly return dated ``w`` is the
    compounded long-short return over the week *ending* at ``w`` and uses only prices
    ``<= w``; corrupting any price after ``w`` cannot move it. Callers embargo the tail
    (use weeks strictly before the decision date) exactly as with the IC window.

    Returns a Series indexed by week-ending date (empty if the factor/sleeve has no data).
    """
    ids = list(sleeve_ids)
    zc = z_panel[z_panel["instrument_id"].isin(ids)]
    pc = prices[prices["instrument_id"].isin(ids)]
    if zc.empty or pc.empty:
        return pd.Series(dtype=float)

    zw = (zc.pivot_table(index="obs_date", columns="instrument_id", values="value",
                         aggfunc="last").sort_index())
    close = (pc.pivot_table(index="obs_date", columns="instrument_id", values="close",
                            aggfunc="last").sort_index())
    zw.index = pd.DatetimeIndex(zw.index)
    close.index = pd.DatetimeIndex(close.index)
    daily_ret = close.pct_change()

    def _ls_weights(row: pd.Series) -> pd.Series:
        r = row.dropna()
        nn = len(r)
        k = int(np.floor(quantile * nn))
        w = pd.Series(0.0, index=row.index)
        if k < 1 or nn < 2 * k or nn < 2:
            return w
        order = r.sort_values()
        w[order.index[-k:]] = 1.0 / k     # long the top quantile
        w[order.index[:k]] = -1.0 / k     # short the bottom quantile
        return w

    W = zw.apply(_ls_weights, axis=1)
    # Hold each date's weights until the next scored date, effective one day later.
    W_daily = (W.reindex(daily_ret.index, method="ffill").shift(1).fillna(0.0))
    ls_daily = (W_daily * daily_ret.reindex(columns=W_daily.columns)).sum(axis=1, min_count=1)
    weekly = (1.0 + ls_daily.fillna(0.0)).resample("W-FRI").prod() - 1.0
    return weekly.dropna()


def score_correlation(z_panels: dict[str, pd.DataFrame], as_of,
                      window: int = 252, min_obs: int = 60) -> pd.DataFrame:
    """PIT estimate of the K x K correlation matrix of factor *scores*.

    The Grinold-Kahn best-linear-predictor needs the correlation ``C_jk`` between two
    factors' standardized scores. We estimate it *cross-sectionally per date* and average
    over time, using only history knowable at ``as_of``:

    - For each date ``d <= as_of`` (restricted to the trailing ``window`` dates the two
      factors share), correlate the two factors' cross-sectional score vectors over their
      common instruments. A date with fewer than 5 common names is skipped (a correlation
      over <5 points is noise, matching the IC ``_MIN_NAMES`` convention).
    - Average the per-date correlations over the window.
    - A pair with fewer than ``min_obs`` usable dates falls back to 0 correlation — the
      identity prior, i.e. "assume independence until we have enough evidence otherwise".

    The result is symmetric with a unit diagonal, indexed/columned by factor name over the
    full ``z_panels`` key set (factors that never co-occur keep the 0 prior). Because every
    input row is filtered to ``obs_date <= as_of``, corrupting any future score cannot move
    a single entry — the PIT invariant the whole system depends on.
    """
    factor_names = list(z_panels.keys())
    C = pd.DataFrame(np.eye(len(factor_names)), index=factor_names, columns=factor_names)
    as_of = pd.Timestamp(as_of)

    # Wide (date x instrument) score matrix per factor, restricted to knowable dates.
    wide: dict[str, pd.DataFrame] = {}
    for f in factor_names:
        p = z_panels[f]
        if p is None or p.empty:
            continue
        p = p[p["obs_date"] <= as_of]
        if p.empty:
            continue
        wide[f] = (p.pivot_table(index="obs_date", columns="instrument_id",
                                 values="value", aggfunc="last").sort_index())

    names = list(wide.keys())
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            fj, fk = names[a], names[b]
            wj, wk = wide[fj], wide[fk]
            common_dates = wj.index.intersection(wk.index)
            if len(common_dates) == 0:
                continue
            common_dates = common_dates[-window:]           # trailing window of shared dates
            corrs: list[float] = []
            for d in common_dates:
                vj, vk = wj.loc[d], wk.loc[d]
                ids = vj.dropna().index.intersection(vk.dropna().index)
                if len(ids) < 5:
                    continue
                x = vj.loc[ids].to_numpy(dtype=float)
                y = vk.loc[ids].to_numpy(dtype=float)
                if x.std() == 0 or y.std() == 0:
                    continue
                r = float(np.corrcoef(x, y)[0, 1])
                if np.isfinite(r):
                    corrs.append(r)
            rho = float(np.mean(corrs)) if len(corrs) >= min_obs else 0.0
            C.loc[fj, fk] = rho
            C.loc[fk, fj] = rho
    return C


def combine_alphas_grinold(z_panels: dict[str, pd.DataFrame],
                           ic_by_factor: dict[str, float],
                           resid_vol: pd.DataFrame,
                           score_corr: pd.DataFrame,
                           ridge: float = 0.10) -> pd.DataFrame:
    """Correlation-aware multi-factor alpha (Grinold-Kahn best linear predictor).

    Given K standardized score panels, per-factor ICs and the score-correlation matrix
    ``C``, the optimal combination weights on the scores are ``w = C^{-1} ic`` and the
    combined forecast is ``alpha_i = sigma_i * sum_k w_k z_{k,i}``. This down-weights
    redundant (correlated) factors instead of double-counting them; when ``C = I`` it
    reduces exactly to the plain ``sigma * IC * z`` sum of :func:`combine_alphas`.

    ``C`` estimated on trailing cross-sectional scores is noisy, so it is ridge-shrunk
    toward the identity before inversion: ``C_r = (1 - ridge) * C + ridge * I`` (the same
    Ledoit-Wolf-style conditioning used elsewhere in the risk stack).

    Alignment mirrors the rest of the alpha layer: the combined score is an *outer* union
    over factors (a name scored by >=1 factor survives, factors with no view there
    contribute 0 — never a fabricated all-missing cell), and the final ``sigma *`` scaling
    is an inner join on residual vol (a name we cannot risk-scale is dropped), exactly as
    :func:`refine_alpha`.

    Returns a long ``[obs_date, instrument_id, value]`` alpha panel.
    """
    factors = [f for f in z_panels if z_panels[f] is not None and not z_panels[f].empty]
    if not factors:
        return pd.DataFrame(columns=["obs_date", "instrument_id", "value"])

    # C over exactly the live factors, defaulting to the identity prior for any missing cell.
    K = len(factors)
    C = np.eye(K)
    for i, fi in enumerate(factors):
        for j, fj in enumerate(factors):
            if fi in score_corr.index and fj in score_corr.columns:
                v = score_corr.loc[fi, fj]
                if pd.notna(v):
                    C[i, j] = float(v)

    C_r = (1.0 - ridge) * C + ridge * np.eye(K)
    ic = np.array([float(ic_by_factor.get(f, 0.0)) for f in factors], dtype=float)
    w = np.linalg.solve(C_r, ic)

    # Combined score s_i = sum_k w_k z_{k,i}, outer-aligned over the union of (date, name).
    frames = []
    for i, f in enumerate(factors):
        p = z_panels[f][["obs_date", "instrument_id", "value"]].copy()
        p["value"] = p["value"] * w[i]
        frames.append(p)
    stacked = pd.concat(frames, ignore_index=True)
    score = stacked.groupby(["obs_date", "instrument_id"], as_index=False)["value"].sum()

    # alpha_i = sigma_i * s_i (unit IC): identical alignment semantics to refine_alpha.
    return refine_alpha(score, 1.0, resid_vol)
