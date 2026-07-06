"""THE point-in-time utility. Every join between data and a decision date routes
through here — backtest and live paths alike.

The contract: a row is visible at `as_of_ts` iff `available_from <= as_of_ts`.
When several vintages of the same (obs_date, entity[, field]) exist, the latest
visible vintage wins. There is no other sanctioned way to read the lake as-of a date.
"""
from __future__ import annotations

import pandas as pd


def _entity_col(df: pd.DataFrame) -> str:
    if "instrument_id" in df.columns:
        return "instrument_id"
    if "series_id" in df.columns:
        return "series_id"
    raise ValueError("panel has neither instrument_id nor series_id")


def asof_panel(df: pd.DataFrame, as_of_ts) -> pd.DataFrame:
    """Rows knowable at `as_of_ts`, latest vintage per (obs_date, entity[, field]).

    `as_of_ts` is coerced to a UTC timestamp; naive input is assumed UTC.
    """
    if df.empty:
        return df
    as_of = pd.Timestamp(as_of_ts)
    if as_of.tzinfo is None:
        as_of = as_of.tz_localize("UTC")
    avail = pd.to_datetime(df["available_from"], utc=True)
    visible = df.loc[avail <= as_of].copy()
    if visible.empty:
        return visible
    entity = _entity_col(visible)
    keys = ["obs_date", entity] + (["field"] if "field" in visible.columns else [])
    sort_cols = ["available_from"] + (["ingested_at"] if "ingested_at" in visible.columns else [])
    visible = visible.sort_values(sort_cols, kind="stable")
    return visible.drop_duplicates(subset=keys, keep="last").reset_index(drop=True)


def asof_wide(df: pd.DataFrame, as_of_ts, value_col: str = "value") -> pd.DataFrame:
    """Convenience: PIT filter then pivot to obs_date x entity wide matrix.

    Wide frames are in-memory helpers only — never persisted (lake is long format).
    """
    snap = asof_panel(df, as_of_ts)
    if snap.empty:
        return pd.DataFrame()
    entity = _entity_col(snap)
    return snap.pivot(index="obs_date", columns=entity, values=value_col).sort_index()
