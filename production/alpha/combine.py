"""Combine per-factor alpha panels into a single blended alpha.

Alphas from different factors cover overlapping but not identical instrument/date sets
(a crypto-only carry factor has no view on equities; a factor may lack history for a
recently listed name). The blend is an *outer* alignment: a name/date present in at
least one factor survives, and factors with no view there contribute 0. A cell that no
factor touches never appears — we never invent a zero alpha out of thin air.
"""
from __future__ import annotations

import pandas as pd


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
