"""Turn normalized scores into an alpha (expected active return) forecast.

Grinold-Kahn refinement: ``alpha_i = volatility_i * IC * score_i``. The z-score carries
the cross-sectional ranking, the shrunk IC scales it to the strength of the historical
signal, and the residual volatility converts a unitless standardized score into a return
forecast in the instrument's own risk units. All three are aligned on
``(obs_date, instrument_id)``; a name missing a residual-vol estimate is dropped (we do
not forecast alpha we cannot risk-scale).
"""
from __future__ import annotations

import numbers

import pandas as pd


def _find_sleeve(*frames: pd.DataFrame) -> pd.Series | None:
    for f in frames:
        if "sleeve" in f.columns:
            return f["sleeve"]
    return None


def refine_alpha(z_panel: pd.DataFrame,
                 ic_shrunk_by_sleeve,
                 resid_vol: pd.DataFrame) -> pd.DataFrame:
    """Compute ``alpha_i = sigma_i * IC* * z_i`` as an aligned long panel.

    Parameters
    ----------
    z_panel : long ``[obs_date, instrument_id, value]`` z-scores (an optional ``sleeve``
        column is honored if present).
    ic_shrunk_by_sleeve : the shrunk IC to apply. Either a scalar (applied to every
        row), or a ``dict``/``pd.Series`` keyed by sleeve (requires a ``sleeve`` column
        on ``z_panel`` or ``resid_vol``; a single-entry mapping is broadcast).
    resid_vol : long ``[obs_date, instrument_id, value]`` daily residual volatility.

    Returns
    -------
    Long ``[obs_date, instrument_id, value]`` alpha panel. Inner-joined on
    ``(obs_date, instrument_id)`` so names missing either a score or a vol are dropped.
    """
    z = z_panel.rename(columns={"value": "_z"})
    v = resid_vol.rename(columns={"value": "_sigma"})
    z_cols = ["obs_date", "instrument_id", "_z"] + (["sleeve"] if "sleeve" in z.columns else [])
    v_cols = ["obs_date", "instrument_id", "_sigma"] + (["sleeve"] if "sleeve" in v.columns else [])
    m = z[z_cols].merge(v[v_cols], on=["obs_date", "instrument_id"], how="inner",
                        suffixes=("", "_v"))

    if isinstance(ic_shrunk_by_sleeve, numbers.Real):
        m["_ic"] = float(ic_shrunk_by_sleeve)
    else:
        ic_map = pd.Series(dict(ic_shrunk_by_sleeve), dtype=float)
        sleeve = _find_sleeve(z, v)
        if sleeve is not None:
            m["_ic"] = m["sleeve"].map(ic_map) if "sleeve" in m.columns \
                else m["instrument_id"].map(lambda _i: None)
        if "_ic" not in m.columns or m["_ic"].isna().all():
            if len(ic_map) == 1:
                m["_ic"] = float(ic_map.iloc[0])
            elif "sleeve" not in m.columns:
                raise ValueError(
                    "ic_shrunk_by_sleeve is a per-sleeve mapping but no 'sleeve' column "
                    "was found on z_panel or resid_vol to align it")

    m["value"] = m["_sigma"] * m["_ic"] * m["_z"]
    out = m[["obs_date", "instrument_id", "value"]].dropna(subset=["value"])
    return out.reset_index(drop=True)
