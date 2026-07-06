"""Cross-sectional score normalization.

The z-score of a signal is computed *within a single (obs_date, sleeve) group* — it
is a same-day cross-sectional statistic, so it is PIT-safe by construction: the value
for instrument i on date D depends only on the other instruments' signal values on the
same date D, never on any past or future date. There is no rolling window here and
therefore nothing to ``shift(1)``.

Winsorization uses the median/MAD (robust to the fat tails that wreck a mean/std clip)
scaled by 1.4826 so that, for Gaussian data, the scaled MAD equals the standard
deviation and a ``3*MAD`` clip is a genuine 3-sigma clip.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_MAD_SCALE = 1.4826  # makes scaled-MAD a consistent estimator of sigma under normality
_ZERO_DISPERSION = 1e-12


def mad(x: np.ndarray) -> float:
    """Median absolute deviation about the median (unscaled)."""
    x = np.asarray(x, dtype=float)
    med = np.median(x)
    return float(np.median(np.abs(x - med)))


def winsorize(x: np.ndarray, n_mad: float = 3.0, scale: float = _MAD_SCALE) -> np.ndarray:
    """Clip values to ``median ± n_mad * scale * MAD``.

    Returns a copy. If the scaled MAD is zero (no dispersion) the input is returned
    unchanged — there is nothing to clip to.
    """
    x = np.asarray(x, dtype=float)
    med = np.median(x)
    s = scale * np.median(np.abs(x - med))
    if s <= 0:
        return x.copy()
    return np.clip(x, med - n_mad * s, med + n_mad * s)


def zscore_scores(signal_panel: pd.DataFrame, sleeve_map: pd.Series) -> pd.DataFrame:
    """Winsorize then standardize signal scores per ``(obs_date, sleeve)`` group.

    Parameters
    ----------
    signal_panel : long panel ``[obs_date, instrument_id, value]``.
    sleeve_map   : ``pd.Series`` mapping ``instrument_id -> sleeve``.

    Returns
    -------
    Long panel ``[obs_date, instrument_id, value]`` of z-scores. Groups with fewer than
    3 names, or with ~zero dispersion after winsorization (a degenerate group where a
    z-score is meaningless), are dropped entirely.
    """
    df = signal_panel[["obs_date", "instrument_id", "value"]].copy()
    df["sleeve"] = df["instrument_id"].map(sleeve_map)
    df = df.dropna(subset=["sleeve", "value"])

    out: list[pd.DataFrame] = []
    for (obs_date, _sleeve), g in df.groupby(["obs_date", "sleeve"], sort=False):
        if len(g) < 3:
            continue
        xw = winsorize(g["value"].to_numpy())
        std = xw.std()  # population std (ddof=0): z then has std exactly 1
        if std < _ZERO_DISPERSION:
            continue
        z = (xw - xw.mean()) / std
        out.append(pd.DataFrame({
            "obs_date": obs_date,
            "instrument_id": g["instrument_id"].to_numpy(),
            "value": z,
        }))

    if not out:
        return pd.DataFrame(columns=["obs_date", "instrument_id", "value"])
    return pd.concat(out, ignore_index=True)
