"""Transaction-cost analysis (TCA) feedback — realized shortfall calibrates the cost model.

Standard Perold implementation-shortfall practice: pre-trade cost *estimates* are only
honest if the post-trade *realized* numbers feed back and recalibrate them. A one-way
estimate that never learns from fills stays optimistic forever (the alpha=0.15 concern).

This module closes that loop. Given a frame of realized implementation shortfall
(``production/execution/shortfall.py`` output — columns include ``instrument_id``,
``shortfall_bps``, ``notional_usd``), :func:`calibrate_overrides` turns per-instrument
realized cost into a ``half_spread_bps`` override candidate:

    candidate = median(|shortfall_bps|) * safety

An override is emitted for an instrument ONLY when it has enough fills (``>= min_fills``)
AND the candidate *exceeds* the instrument's sleeve-default half-spread. Overrides may
therefore only ever RAISE the effective cost above the sleeve default — floors are floors,
never lowered (CLAUDE.md, non-negotiable). Candidates are capped at the cost model's
``cap_bps``. The result round-trips through a lake reference table (``cost_overrides``)
carrying a calibration timestamp and per-instrument fill counts.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from production.core.config import costs_config
from production.core.lake import Lake, LakeError
from production.signals.base import sleeve_from_id

_OVERRIDES_TABLE = "cost_overrides"
_OVERRIDE_COLUMNS = ["instrument_id", "half_spread_bps", "n_fills",
                     "median_abs_shortfall_bps", "calibrated_at"]

# Per-run accumulation target for --live fills (backlog #15). Columns per order: the
# realized fill alongside the decision it was measured against, plus enough provenance
# (``filled_at``, ``run_id``) to audit which daily_run invocation produced the row. This
# is the table --calibrate-tca reads by default (see scripts/daily_run.py:_calibrate_tca).
SHORTFALL_LOG_TABLE = "shortfall_log"
SHORTFALL_LOG_COLUMNS = ["instrument_id", "side", "qty", "decision_price", "fill_price",
                         "shortfall_bps", "filled_at", "run_id"]


def calibrate_overrides(shortfall_df: pd.DataFrame, min_fills: int = 20,
                        safety: float = 1.25, cfg: dict | None = None) -> dict[str, dict]:
    """Per-instrument ``half_spread_bps`` overrides calibrated from realized shortfall.

    Parameters
    ----------
    shortfall_df : frame with at least ``instrument_id`` and ``shortfall_bps`` (the
                   ``implementation_shortfall`` per-order output).
    min_fills    : minimum number of finite-shortfall fills required before an
                   instrument is eligible for calibration.
    safety       : multiplier applied to the realized median absolute shortfall.
    cfg          : cost config (defaults to ``costs_config()``); used for the sleeve
                   default half-spreads and the ``cap_bps`` ceiling.

    Returns ``{instrument_id: {"half_spread_bps", "n_fills", "median_abs_shortfall_bps"}}``,
    including only instruments that clear the fill gate AND whose candidate strictly
    exceeds their sleeve-default half-spread (overrides only ever raise cost; floors stay
    floors). Candidates are capped at ``cap_bps``.
    """
    cfg = cfg if cfg is not None else costs_config()
    sleeves = cfg["sleeves"]
    cap_bps = float(cfg["cap_bps"])

    out: dict[str, dict] = {}
    if shortfall_df is None or shortfall_df.empty:
        return out
    if "instrument_id" not in shortfall_df.columns or "shortfall_bps" not in shortfall_df.columns:
        return out

    df = shortfall_df[["instrument_id", "shortfall_bps"]].copy()
    df["shortfall_bps"] = pd.to_numeric(df["shortfall_bps"], errors="coerce")
    df = df[np.isfinite(df["shortfall_bps"])]
    if df.empty:
        return out

    for iid, g in df.groupby("instrument_id", sort=True):
        n = int(len(g))
        if n < min_fills:
            continue
        median_abs = float(np.median(np.abs(g["shortfall_bps"].to_numpy())))
        candidate = median_abs * float(safety)
        # Sleeve default half-spread — the floor below which we never calibrate down.
        sleeve = sleeve_from_id(str(iid))
        sleeve_hs = float(sleeves[sleeve]["half_spread_bps"])
        if candidate <= sleeve_hs:
            continue                      # never lower cost — leave the sleeve default in place
        candidate = min(candidate, cap_bps)
        out[str(iid)] = {
            "half_spread_bps": candidate,
            "n_fills": n,
            "median_abs_shortfall_bps": median_abs,
        }
    return out


def write_overrides(overrides: dict, lake: Lake) -> Path:
    """Persist calibrated overrides to the ``cost_overrides`` lake reference table.

    Each row is stamped with a UTC ``calibrated_at``. Writing an empty ``overrides``
    still creates a (schema-only) table so downstream reads are deterministic.
    """
    calibrated_at = pd.Timestamp.utcnow()
    rows = [
        {"instrument_id": iid,
         "half_spread_bps": float(o["half_spread_bps"]),
         "n_fills": int(o["n_fills"]),
         "median_abs_shortfall_bps": float(o["median_abs_shortfall_bps"]),
         "calibrated_at": calibrated_at}
        for iid, o in sorted(overrides.items())
    ]
    df = pd.DataFrame(rows, columns=_OVERRIDE_COLUMNS)
    return lake.write_reference(df, _OVERRIDES_TABLE)


def read_overrides(lake: Lake) -> dict[str, dict]:
    """Read the ``cost_overrides`` table into a CostModel-ready override dict.

    Returns ``{instrument_id: {"half_spread_bps", "n_fills", "median_abs_shortfall_bps"}}``,
    or ``{}`` when the table is absent (never calibrated yet).
    """
    try:
        df = lake.read_reference(_OVERRIDES_TABLE)
    except LakeError:
        return {}
    out: dict[str, dict] = {}
    for r in df.itertuples(index=False):
        out[str(r.instrument_id)] = {
            "half_spread_bps": float(r.half_spread_bps),
            "n_fills": int(r.n_fills),
            "median_abs_shortfall_bps": float(r.median_abs_shortfall_bps),
        }
    return out


def append_shortfall_log(rows: pd.DataFrame, lake: Lake) -> Path | None:
    """Read-modify-write append onto the lake's ``shortfall_log`` reference table.

    ``Lake.write_reference`` REPLACES the underlying parquet file wholesale — there is no
    native append — so any prior table is read back first and the new rows are concatenated
    AFTER it; nothing already accumulated is ever lost. ``rows`` must already carry
    :data:`SHORTFALL_LOG_COLUMNS` (``scripts/daily_run.py`` builds it from
    :func:`production.execution.shortfall.implementation_shortfall` plus ``qty``,
    ``filled_at``, ``run_id``).

    An empty/``None`` ``rows`` is a no-op — nothing is written and any existing table is
    left untouched (this is what makes ``--dry-run`` safe: the caller simply never
    constructs rows to pass in, but this guard makes the function safe standalone too).
    Returns the written path, or ``None`` when nothing was written.
    """
    if rows is None or rows.empty:
        return None
    try:
        existing = lake.read_reference(SHORTFALL_LOG_TABLE)
    except LakeError:
        existing = pd.DataFrame(columns=SHORTFALL_LOG_COLUMNS)
    combined = pd.concat([existing[SHORTFALL_LOG_COLUMNS], rows[SHORTFALL_LOG_COLUMNS]],
                         ignore_index=True)
    return lake.write_reference(combined, SHORTFALL_LOG_TABLE)
