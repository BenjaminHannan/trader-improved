"""Loader framework: availability stamping, the ingest pipeline, and symbol resolution.

Every dataset that lands in the curated lake goes through a `BaseLoader.run`:
    fetch -> write_raw (immutable snapshot) -> transform (canonical long, vendor
    symbols mapped to synthetic instrument_ids) -> stamp_availability -> audit
    (raise on fatal) -> write_curated + write_audit.

The single most important thing a loader does is stamp `available_from` — the UTC
timestamp at which a value was *knowable*. Signals may only ever see rows whose
`available_from <= as_of`, so getting this lag right is what separates an honest
backtest from a look-ahead fantasy. The `AvailabilityRule` kinds below encode the
publication lag of each vendor explicitly rather than guessing.
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

from production.core.lake import Lake, MANDATORY_COLUMNS
from production.data.audit import audit

# Reference resolver is written by another agent; guard the import so our loaders
# (and their tests) never hard-depend on it. When absent we fall back to a local
# resolver that reads the instrument master's vendor_symbols json + validity window.
try:  # pragma: no cover - exercised only once reference.instruments exists
    from production.reference.instruments import resolve_vendor_symbol as _external_resolve
except Exception:  # ModuleNotFoundError while the module is still being written
    _external_resolve = None


class IngestError(Exception):
    """Raised when a fatal audit failure blocks a curated write."""


# --------------------------------------------------------------- availability
@dataclass(frozen=True)
class AvailabilityRule:
    """How to derive `available_from` for a dataset.

    kinds:
      - "obs_offset":       available_from = obs_date (UTC midnight) + params["offset"]
      - "next_weekday_time":first weekday >= obs_date at params[hour,minute] UTC
                            (COT: Tuesday obs -> that week's Friday 20:30 UTC)
      - "ingest_time":      available_from = ingested_at (snapshot datasets)
      - "explicit":         available_from = df[params["column"]] (e.g. ALFRED
                            realtime_start, or a funding-rate publish timestamp)
    """
    kind: str
    params: dict[str, Any] = field(default_factory=dict)


def stamp_availability(df: pd.DataFrame, rule: AvailabilityRule) -> pd.DataFrame:
    """Apply an `AvailabilityRule`, returning a copy with `available_from` (UTC)."""
    df = df.copy()
    p = rule.params
    obs = pd.to_datetime(df["obs_date"])

    if rule.kind == "obs_offset":
        avail = obs.dt.tz_localize("UTC") + p["offset"]
    elif rule.kind == "next_weekday_time":
        weekday = int(p["weekday"])
        hh, mm = int(p.get("hour", 0)), int(p.get("minute", 0))

        def _next(d) -> pd.Timestamp:
            d = pd.Timestamp(d).normalize()
            days_ahead = (weekday - d.weekday()) % 7  # 0 => same day (on or after)
            stamp = d + pd.Timedelta(days=days_ahead)
            return stamp.tz_localize("UTC") + pd.Timedelta(hours=hh, minutes=mm)

        avail = obs.map(_next)
    elif rule.kind == "ingest_time":
        avail = pd.to_datetime(df["ingested_at"], utc=True)
    elif rule.kind == "explicit":
        avail = pd.to_datetime(df[p["column"]], utc=True)
    else:
        raise ValueError(f"unknown availability rule kind: {rule.kind!r}")

    df["available_from"] = avail
    return df


# ----------------------------------------------------------------- resolution
_CLASS_PREFIX = {"EQ": "equity", "CR": "crypto", "FX": "fx", "CO": "commodity"}


def asset_class_from_instrument_id(instrument_id: str) -> str:
    """Derive the lake `asset_class` partition from the synthetic id prefix."""
    prefix = str(instrument_id).split(":", 1)[0]
    return _CLASS_PREFIX.get(prefix, "unknown")


def _local_resolve(master: pd.DataFrame | None, vendor: str, symbol: str,
                   as_of=None) -> str | None:
    """Fallback resolver: match vendor_symbols[vendor] (or the plain symbol) within
    the PIT validity window. Deliberately simple so loader tests don't depend on the
    reference package still being written by another agent."""
    if master is None or len(master) == 0:
        return None
    for _, row in master.iterrows():
        raw = row.get("vendor_symbols")
        try:
            mapping = json.loads(raw) if isinstance(raw, str) else dict(raw or {})
        except (json.JSONDecodeError, TypeError, ValueError):
            mapping = {}
        if mapping.get(vendor) != symbol and row.get("symbol") != symbol:
            continue
        if as_of is not None:
            vf, vt = row.get("valid_from"), row.get("valid_to")
            if vf is not None and pd.notna(vf) and pd.Timestamp(as_of) < pd.Timestamp(vf):
                continue
            if vt is not None and pd.notna(vt) and pd.Timestamp(as_of) > pd.Timestamp(vt):
                continue
        return row["instrument_id"]
    return None


# --------------------------------------------------------------------- result
@dataclass
class IngestResult:
    dataset: str
    vendor: str
    rows: int
    start: pd.Timestamp
    end: pd.Timestamp
    audit: dict
    paths: list


# ----------------------------------------------------------------- raw framing
def _to_raw_frame(payload: object) -> pd.DataFrame:
    """Best-effort flat DataFrame snapshot of any vendor payload for the raw zone.

    Parquet can't store MultiIndex columns or arbitrary python objects, so we
    flatten. The raw zone is an audit trail, not a query surface — fidelity of the
    exact bytes matters less than having *something* immutable per pull.
    """
    if isinstance(payload, pd.DataFrame):
        df = payload.copy()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ["_".join(str(x) for x in c) for c in df.columns]
        return df.reset_index()
    if isinstance(payload, dict):
        vals = list(payload.values())
        if vals and all(isinstance(v, pd.DataFrame) for v in vals):
            frames = []
            for key, sub in payload.items():
                f = sub.copy()
                if isinstance(f.columns, pd.MultiIndex):
                    f.columns = ["_".join(str(x) for x in c) for c in f.columns]
                f = f.reset_index()
                f.insert(0, "__key__", str(key))
                frames.append(f)
            return pd.concat(frames, ignore_index=True)
        return pd.DataFrame([{"payload": json.dumps(payload, default=str)}])
    if isinstance(payload, list):
        try:
            return pd.DataFrame(payload)
        except (ValueError, TypeError):
            return pd.DataFrame([{"payload": json.dumps(payload, default=str)}])
    return pd.DataFrame([{"payload": str(payload)}])


# ------------------------------------------------------------- ohlcv helpers
def prices_to_long(payload, resolve: Callable[[str], str | None]) -> pd.DataFrame:
    """Vendor OHLCV -> canonical long [obs_date, instrument_id, close, volume,
    dollar_volume, asset_class]. Accepts either a dict {symbol: OHLCV frame} or a
    yfinance-style DataFrame with MultiIndex (symbol, field) columns."""
    if isinstance(payload, dict):
        items = list(payload.items())
    elif isinstance(payload, pd.DataFrame) and isinstance(payload.columns, pd.MultiIndex):
        items = [(s, payload[s]) for s in payload.columns.get_level_values(0).unique()]
    else:
        raise IngestError("unrecognized OHLCV payload shape")

    frames = []
    for sym, sub in items:
        iid = resolve(sym)
        if iid is None:
            continue
        cols = {str(c).lower(): c for c in sub.columns}
        if "close" not in cols:
            continue
        close = pd.to_numeric(sub[cols["close"]], errors="coerce").to_numpy()
        if "volume" in cols:
            volume = pd.to_numeric(sub[cols["volume"]], errors="coerce").to_numpy()
        else:
            volume = np.full(len(sub), np.nan)
        f = pd.DataFrame({
            "obs_date": pd.to_datetime(sub.index),
            "instrument_id": iid,
            "close": close,
            "volume": volume,
        })
        f["dollar_volume"] = f["close"] * f["volume"]
        f["asset_class"] = asset_class_from_instrument_id(iid)
        frames.append(f)
    if not frames:
        return pd.DataFrame(columns=["obs_date", "instrument_id", "close", "volume",
                                     "dollar_volume", "asset_class"])
    out = pd.concat(frames, ignore_index=True)
    return out.dropna(subset=["close"]).reset_index(drop=True)


# ------------------------------------------------------------------- pipeline
class BaseLoader(ABC):
    """Base class for every dataset loader. Subclasses set the class attributes and
    implement `fetch`/`transform`; `run` orchestrates the PIT ingest pipeline."""

    dataset: str = ""
    vendor: str = ""
    source: str | None = None
    asset_classes: list[str] = []
    availability_rule: AvailabilityRule | None = None
    expectations: dict = {}
    default_asset_class: str | None = None

    # small overlap re-pull on incremental loads; the lake upsert dedupes so the
    # overlap is harmless and guards against late-arriving revisions at the tail.
    incremental_overlap = pd.Timedelta(days=5)

    def __init__(self, lake: Lake | None = None, instruments: pd.DataFrame | None = None):
        self.lake = lake if lake is not None else Lake()
        self.instruments = instruments
        self.warnings: list[str] = []

    # -- to be implemented by subclasses -----------------------------------
    @abstractmethod
    def fetch(self, start, end) -> object:
        """Return the raw vendor payload (DataFrame, dict, or list of records)."""

    @abstractmethod
    def transform(self, raw) -> pd.DataFrame:
        """Map the raw payload to canonical long format with vendor symbols
        resolved to synthetic instrument_ids (or series_id for macro series)."""

    # -- shared machinery ---------------------------------------------------
    def resolve(self, symbol: str, vendor: str | None = None, as_of=None) -> str | None:
        vendor = vendor or self.vendor
        if _external_resolve is not None:
            try:
                ref = _external_resolve(self.instruments, vendor, symbol,
                                        as_of if as_of is not None else pd.Timestamp.now(tz="UTC"))
                if ref:
                    return ref
            except Exception:
                pass
        return _local_resolve(self.instruments, vendor, symbol, as_of)

    def stamp_availability(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.availability_rule is None:
            raise IngestError(f"{type(self).__name__} has no availability_rule")
        return stamp_availability(df, self.availability_rule)

    def run(self, start, end, incremental=True) -> IngestResult:
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        if incremental:
            wm = self.lake.watermark(self.dataset)
            if wm is not None:
                adjusted = pd.Timestamp(wm) - self.incremental_overlap
                if adjusted > start:
                    start = adjusted

        raw = self.fetch(start, end)
        ingest_ts = pd.Timestamp.now(tz="UTC")
        raw_path = self.lake.write_raw(_to_raw_frame(raw), self.vendor, self.dataset, ingest_ts)

        df = self.transform(raw)
        df = df.copy() if df is not None else pd.DataFrame()
        if not df.empty:
            df["obs_date"] = pd.to_datetime(df["obs_date"])
            if "source" not in df.columns:
                df["source"] = self.source or self.vendor
            if "ingested_at" not in df.columns:
                df["ingested_at"] = ingest_ts
            df = self.stamp_availability(df)

        report = audit(df, self.expectations)
        record = {
            "dataset": self.dataset,
            "vendor": self.vendor,
            "rows": int(len(df)),
            "start": str(start),
            "end": str(end),
            "fatal": report.fatal,
            "checks": report.checks,
            "summary": report.summary,
            "warnings": list(self.warnings),
            "ingested_at": str(ingest_ts),
        }
        audit_name = f"{self.dataset}_{self.vendor}_{ingest_ts.strftime('%Y%m%dT%H%M%S%f')}"
        self.lake.write_audit(record, audit_name)

        if report.fatal:
            raise IngestError(
                f"{self.dataset}/{self.vendor} audit failed: "
                f"{[k for k, v in report.checks.items() if v.get('fatal')]}")

        paths = [raw_path]
        if not df.empty:
            if "asset_class" in df.columns:
                for ac, chunk in df.groupby("asset_class"):
                    paths += self.lake.write_curated(chunk.drop(columns=["asset_class"]),
                                                     self.dataset, str(ac))
            else:
                ac = self.default_asset_class or (
                    self.asset_classes[0] if self.asset_classes else "unknown")
                paths += self.lake.write_curated(df, self.dataset, ac)

        return IngestResult(dataset=self.dataset, vendor=self.vendor, rows=int(len(df)),
                            start=start, end=end, audit=record, paths=paths)
