"""Signal purification — strip unpaid risk exposure from a raw alpha score.

Grinold-Kahn (Ch. 14): an alpha should carry *only* the view you are paid for.
Incidental exposure to risk factors on which you have no forecast (market beta,
size, residual vol, sectors, …) adds variance without return and mechanically
dilutes the information coefficient. The fix is a pure-factor construction: on each
rebalance date, cross-sectionally regress the score on the point-in-time risk
exposure matrix ``B`` and keep the residual — the part of the signal orthogonal to
those exposures — then re-standardize it.

The signal's *own* style must never be regressed out (that would strip the very
view being expressed): a momentum-family signal excludes the ``momentum`` column of
``B``, a low-vol signal excludes ``vol``, etc. Carry/positioning/value families have
no style twin among the risk columns and neutralize against all of ``B``.

PIT: this is a *same-date* cross-sectional statistic, exactly like the z-score. The
score for name ``i`` on date ``t`` depends only on the other names' scores and
exposures on the same date ``t`` — both of which are already ``obs_date <= t``. There
is no rolling window and nothing to ``shift(1)``; corrupting any future row cannot
move a single purified value.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Map a factor name -> the risk-exposure style that IS its own view and must therefore
# survive purification (excluded from the neutralization set). Anything absent defaults
# to ``None`` = neutralize against every column of B (no own-style twin to protect).
FACTOR_STYLE: dict[str, str | None] = {
    "mom_12_1": "momentum",
    "tsmom": "momentum",
    "str_reversal_1m": "momentum",
    "low_vol": "vol",
    # carry / positioning / value families -> None (neutralize against all of B):
    "carry_funding": None,
    "carry_rate_diff": None,
    "basis_carry": None,
    "cot_positioning": None,
    "mcap_tvl": None,
}


def exclude_for(factor_name: str) -> list[str]:
    """The ``exclude`` list for a factor: its own-style B column, or [] if none."""
    style = FACTOR_STYLE.get(factor_name)
    return [style] if style else []


def _as_series(z_wide: pd.DataFrame | pd.Series) -> pd.Series:
    """Coerce a one-date cross-section to a ``Series`` indexed by instrument_id."""
    if isinstance(z_wide, pd.Series):
        return z_wide.astype(float)
    if isinstance(z_wide, pd.DataFrame):
        if "instrument_id" in z_wide.columns and "value" in z_wide.columns:
            return z_wide.set_index("instrument_id")["value"].astype(float)
        if z_wide.shape[0] == 1:            # single-row wide frame: cols = instrument_id
            return z_wide.iloc[0].astype(float)
        return z_wide.squeeze().astype(float)
    return pd.Series(z_wide, dtype=float)


def purify_scores(z_wide: pd.DataFrame | pd.Series, B: pd.DataFrame,
                  exclude=("momentum",), min_excess: int = 3) -> pd.Series:
    """Neutralize a one-date score cross-section against the risk exposures ``B``.

    Parameters
    ----------
    z_wide : the score cross-section for a single date — a ``Series`` indexed by
        instrument_id, a single-row wide frame, or a long ``[obs_date?, instrument_id,
        value]`` frame.
    B : exposure matrix, index = instrument_id, columns = risk styles (market/size/
        momentum/vol + optional ``sector_*`` dummies). The SAME PIT ``B`` the risk
        model built as of this date.
    exclude : column names to leave in place (the signal's own style). Defaults to
        ``("momentum",)``; pass ``[]`` to neutralize against every column of ``B``.
    min_excess : require at least ``K_used + min_excess`` usable names before purifying
        — a regression with too few excess degrees of freedom over-fits and would
        manufacture spurious residual. Below that, the input is returned unchanged.

    Returns
    -------
    ``pd.Series`` indexed like the input: the residual of the cross-sectional OLS of
    the score on ``B[used_cols]`` (with an intercept), re-standardized to mean 0 /
    std 1 over the regressed names. Names that were dropped (missing score or missing
    any used exposure) keep their original score. When the sample is too small, or no
    columns remain to neutralize against, the input is returned unchanged.
    """
    z = _as_series(z_wide)
    exclude_set = set(exclude or ())
    used_cols = [c for c in B.columns if c not in exclude_set]
    if not used_cols:                       # nothing to neutralize against
        return z

    Bc = B.loc[B.index.intersection(z.index), used_cols].astype(float)
    sub = pd.concat([z.rename("_z"), Bc], axis=1, join="inner").dropna()
    n = len(sub)
    k_used = len(used_cols)
    if n < k_used + int(min_excess):        # documented fallback: too few excess d.o.f.
        return z

    y = sub["_z"].to_numpy(dtype=float)
    X = np.column_stack([np.ones(n), sub[used_cols].to_numpy(dtype=float)])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta

    mu = resid.mean()
    sd = resid.std(ddof=0)
    if not np.isfinite(sd) or sd < 1e-12:   # degenerate residual -> nothing meaningful
        return z
    resid_std = (resid - mu) / sd

    out = z.copy()
    out.loc[sub.index] = resid_std
    return out
