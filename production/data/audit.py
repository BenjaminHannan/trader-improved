"""Data-quality audit run on every curated write.

Some failures are *fatal* — they must block the write, because letting the row into
the lake would silently corrupt every downstream signal (a missing schema column, a
negative price, a duplicated key, or an implausibly small pull). Others are
*warnings* — a mildly elevated null fraction or a sleeve that is temporarily missing
a few names is worth recording but not worth halting an ingest over. The distinction
is deliberate: fatal checks defend the invariants the rest of the system assumes;
warnings surface drift for a human to look at later.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from production.core.lake import MANDATORY_COLUMNS

DEFAULT_MAX_NULL_FRAC = 0.02
FATAL_NULL_MULTIPLE = 5.0  # null/coverage warnings escalate to fatal past 5x threshold

# Columns that are structural (keys / provenance) and not subject to null/summary
# scanning as "values".
_STRUCTURAL = set(MANDATORY_COLUMNS) | {
    "instrument_id", "series_id", "asset_class", "field", "realtime_start",
}


@dataclass
class AuditReport:
    fatal: bool
    checks: dict
    summary: dict


def _entity_col(df: pd.DataFrame) -> str | None:
    if "instrument_id" in df.columns:
        return "instrument_id"
    if "series_id" in df.columns:
        return "series_id"
    return None


def audit(df: pd.DataFrame, expectations: dict | None = None) -> AuditReport:
    """Validate a canonical long frame against `expectations`.

    expectations keys (all optional):
      columns:            extra required columns beyond the mandatory set
      max_null_frac:      per-column null tolerance (default 0.02)
      ranges:             {col: (lo, hi)} inclusive bounds, None = unbounded
      min_rows:           minimum acceptable row count
      expected_entities:  expected distinct entity count (coverage, warn-only)
    """
    expectations = expectations or {}
    checks: dict[str, dict] = {}
    entity = _entity_col(df)

    # -- schema: mandatory columns + an entity key + any extra required ------
    required = set(MANDATORY_COLUMNS) | set(expectations.get("columns", []))
    missing = sorted(required - set(df.columns))
    entity_ok = entity is not None
    schema_ok = not missing and entity_ok
    checks["schema"] = {
        "ok": schema_ok, "fatal": not schema_ok,
        "missing": missing, "has_entity": entity_ok,
    }
    if not schema_ok:
        # Without a valid schema the remaining checks are meaningless.
        return AuditReport(fatal=True, checks=checks, summary=_summary(df, entity))

    # -- duplicate keys (after collapsing identical vintages) ---------------
    key = ["obs_date", entity] + (["field"] if "field" in df.columns else [])
    key_avail = key + ["available_from"]
    deduped = df.drop_duplicates()  # exact re-ingests are idempotent, not dupes
    dup_count = int(deduped.duplicated(subset=key_avail).sum())
    checks["duplicates"] = {"ok": dup_count == 0, "fatal": dup_count > 0, "count": dup_count}

    # -- value ranges -------------------------------------------------------
    range_violations: dict[str, int] = {}
    for col, (lo, hi) in expectations.get("ranges", {}).items():
        if col not in df.columns:
            continue
        vals = pd.to_numeric(df[col], errors="coerce")
        bad = pd.Series(False, index=df.index)
        if lo is not None:
            bad |= vals < lo
        if hi is not None:
            bad |= vals > hi
        n = int(bad.sum())
        if n:
            range_violations[col] = n
    checks["ranges"] = {
        "ok": not range_violations, "fatal": bool(range_violations),
        "violations": range_violations,
    }

    # -- minimum row count --------------------------------------------------
    min_rows = int(expectations.get("min_rows", 0))
    rows_ok = len(df) >= min_rows
    checks["min_rows"] = {"ok": rows_ok, "fatal": not rows_ok,
                          "rows": int(len(df)), "min_rows": min_rows}

    # -- null fractions (warn; fatal only past 5x threshold) ----------------
    thr = float(expectations.get("max_null_frac", DEFAULT_MAX_NULL_FRAC))
    value_cols = [c for c in df.columns if c not in _STRUCTURAL]
    null_fracs = {c: float(df[c].isna().mean()) for c in value_cols}
    over = {c: f for c, f in null_fracs.items() if f > thr}
    fatal_over = {c: f for c, f in null_fracs.items() if f > FATAL_NULL_MULTIPLE * thr}
    checks["null_fraction"] = {
        "ok": not over, "fatal": bool(fatal_over), "threshold": thr,
        "over_threshold": over, "fatal_over": fatal_over,
        "fractions": null_fracs,
    }

    # -- coverage vs expected entity count (warn-only) ----------------------
    expected_entities = expectations.get("expected_entities")
    if expected_entities is not None:
        distinct = int(df[entity].nunique())
        cov_ok = distinct >= int(expected_entities)
        checks["coverage"] = {"ok": cov_ok, "fatal": False,
                              "distinct": distinct, "expected": int(expected_entities)}

    fatal = any(c.get("fatal") for c in checks.values())
    return AuditReport(fatal=fatal, checks=checks, summary=_summary(df, entity))


def _summary(df: pd.DataFrame, entity: str | None) -> dict:
    """Per-numeric-column moments + row count + obs_date span."""
    summary: dict = {"rows": int(len(df))}
    if df.empty:
        return summary
    if "obs_date" in df.columns:
        obs = pd.to_datetime(df["obs_date"])
        summary["date_span"] = [str(obs.min()), str(obs.max())]
    if entity is not None:
        summary["distinct_entities"] = int(df[entity].nunique())
    cols = {}
    for c in df.columns:
        # is_numeric_dtype, not np.issubdtype: extension dtypes (StringDtype,
        # tz-aware datetimes) make issubdtype RAISE rather than return False
        if c in _STRUCTURAL or not pd.api.types.is_numeric_dtype(df[c]):
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        cols[c] = {
            "mean": float(s.mean()), "std": float(s.std()),
            "min": float(s.min()), "max": float(s.max()),
        }
    summary["columns"] = cols
    return summary
