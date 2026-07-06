"""Information-coefficient machinery (Spearman rank IC) with PIT-safe rolling stats.

The IC measures how well a signal ranks *future* returns. This makes it structurally
forward-looking, and that is the whole trap this module is built to avoid:

- ``forward_returns`` deliberately looks forward: ``fwd_ret`` at date ``t`` is the
  return realized over ``(t, t+h]``. It is only ever used to *score* a signal after the
  fact, never as an input to a same-day decision.
- ``rolling_shrunk_ic`` is the bridge to live decisions: the rolling IC used to weight a
  factor at decision date ``t`` may only include IC dates ``d`` whose forward window has
  already resolved, i.e. ``d <= t - horizon_days``. The unresolved tail is embargoed.
  A corruption test relies on this: mutating IC values in ``(t - horizon, t]`` must not
  change the rolling value at ``t``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

_MIN_NAMES = 5  # Spearman on fewer than 5 names is noise; drop the group.


def forward_returns(prices: pd.DataFrame, horizon_days: int) -> pd.DataFrame:
    """Positional forward return per instrument: ``close[t+h]/close[t] - 1``.

    Shifts are positional (``groupby(instrument).shift(-h)``) on each instrument's own
    trading-day series, so calendar gaps between instruments do not leak across names.
    The last ``h`` rows of every instrument have no resolved forward window and are
    dropped (NaN).
    """
    df = prices[["obs_date", "instrument_id", "close"]].copy()
    df = df.sort_values(["instrument_id", "obs_date"])
    fwd_close = df.groupby("instrument_id")["close"].shift(-horizon_days)
    df["fwd_ret"] = fwd_close / df["close"] - 1.0
    df = df.dropna(subset=["fwd_ret"])
    return df[["obs_date", "instrument_id", "fwd_ret"]].reset_index(drop=True)


def rank_ic(scores: pd.DataFrame, fwd: pd.DataFrame,
            sleeve_map: pd.Series, min_names: int = _MIN_NAMES) -> pd.DataFrame:
    """Spearman rank IC between scores and forward returns per ``(obs_date, sleeve)``.

    Returns ``[obs_date, sleeve, rank_ic, n_names]``. Groups with fewer than
    ``min_names`` names, or with no dispersion in either scores or returns (correlation
    undefined), are dropped.
    """
    m = scores[["obs_date", "instrument_id", "value"]].merge(
        fwd[["obs_date", "instrument_id", "fwd_ret"]],
        on=["obs_date", "instrument_id"], how="inner")
    m["sleeve"] = m["instrument_id"].map(sleeve_map)
    m = m.dropna(subset=["sleeve"])

    rows: list[tuple] = []
    for (obs_date, sleeve), g in m.groupby(["obs_date", "sleeve"], sort=False):
        if len(g) < min_names:
            continue
        x = g["value"].to_numpy(dtype=float)
        y = g["fwd_ret"].to_numpy(dtype=float)
        if np.std(x) == 0 or np.std(y) == 0:
            continue
        rho = spearmanr(x, y)[0]  # [0] = correlation, version-agnostic
        if np.isnan(rho):
            continue
        rows.append((obs_date, sleeve, float(rho), int(len(g))))

    return pd.DataFrame(rows, columns=["obs_date", "sleeve", "rank_ic", "n_names"])


def ic_tstat(ic) -> float:
    """t-statistic of an IC series: ``mean / std * sqrt(n)`` (sample std, ddof=1)."""
    s = pd.Series(ic).dropna()
    n = len(s)
    if n < 2:
        return float("nan")
    sd = s.std(ddof=1)
    if sd == 0:
        return float("nan")
    return float(s.mean() / sd * np.sqrt(n))


def shrunk_ic(ic, n0: int = 126) -> float:
    """Shrink a mean IC toward zero by the sample size: ``mean * n / (n + n0)``.

    With few observations the mean IC is unreliable, so it is pulled toward 0; as ``n``
    grows past ``n0`` the shrinkage relaxes. ``n0=126`` ~ half a trading year.
    """
    s = pd.Series(ic).dropna()
    n = len(s)
    if n == 0:
        return float("nan")
    return float(s.mean() * n / (n + n0))


def rolling_shrunk_ic(ic_by_date: pd.Series, window: int = 252,
                      n0: int = 126, horizon_days: int = 1) -> pd.Series:
    """Trailing shrunk IC usable at each date, embargoing the unresolved forward window.

    ``ic_by_date`` is an IC series indexed by date. The value at date ``t`` is the shrunk
    mean of the (up to ``window``) most recent IC observations whose ``horizon_days``
    forward-return window has fully resolved by ``t``. The embargo is POSITIONAL on the
    IC index: ``horizon_days`` is a trading-day horizon and the IC index consists of
    trading dates, so the IC at position ``j`` is resolved at position ``i`` only when
    ``j <= i - horizon_days``. (A calendar ``Timedelta`` cutoff would under-embargo:
    21 calendar days < 21 trading days, letting unresolved ICs leak in.) On an index
    sparser than daily this over-embargoes — the conservative direction.
    """
    s = pd.Series(ic_by_date).dropna()
    s = s.sort_index()
    out: dict = {}
    for i, t in enumerate(s.index):
        j_max = i - horizon_days
        if j_max < 0:
            out[t] = float("nan")
            continue
        eligible = s.iloc[:j_max + 1]
        w = eligible.iloc[-window:]
        out[t] = shrunk_ic(w, n0=n0)
    return pd.Series(out)


def ic_decay(scores: pd.DataFrame, prices: pd.DataFrame, sleeve_map: pd.Series,
             horizons=(1, 2, 5, 10, 21, 42)) -> pd.DataFrame:
    """Mean rank IC of a score panel at a range of forward horizons, per sleeve.

    Returns long ``[sleeve, horizon, ic]``. A signal with genuine short-horizon edge
    that fades will show ``|ic|`` shrinking as the horizon grows.
    """
    rows: list[tuple] = []
    for h in horizons:
        fwd = forward_returns(prices, h)
        ic = rank_ic(scores, fwd, sleeve_map)
        if ic.empty:
            continue
        for sleeve, g in ic.groupby("sleeve"):
            rows.append((sleeve, int(h), float(g["rank_ic"].mean())))
    return pd.DataFrame(rows, columns=["sleeve", "horizon", "ic"])


def decay_halflife(decay: pd.DataFrame) -> float:
    """Horizon at which ``|IC|`` first falls to half of ``|IC|`` at the shortest horizon.

    The per-sleeve decay table is collapsed to a single ``|IC|`` curve by averaging the
    absolute IC across sleeves at each horizon. Returns the horizon (linearly
    interpolated between the two bracketing horizons) where the curve first crosses half
    its base level, or ``np.inf`` if it never does.
    """
    if decay.empty:
        return np.inf
    curve = (decay.assign(abs_ic=decay["ic"].abs())
                  .groupby("horizon")["abs_ic"].mean()
                  .sort_index())
    horizons = curve.index.to_numpy(dtype=float)
    vals = curve.to_numpy(dtype=float)
    if len(vals) == 0 or vals[0] <= 0:
        return np.inf
    half = 0.5 * vals[0]
    for i in range(1, len(vals)):
        if vals[i] <= half:
            x0, x1 = horizons[i - 1], horizons[i]
            y0, y1 = vals[i - 1], vals[i]
            if y0 == y1:
                return float(x1)
            frac = (y0 - half) / (y0 - y1)
            return float(x0 + frac * (x1 - x0))
    return np.inf
