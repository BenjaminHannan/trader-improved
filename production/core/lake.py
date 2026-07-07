"""Parquet lake read/write with mandatory-column enforcement.

Layout:
  data/raw/vendor=<v>/dataset=<d>/ingest_date=YYYY-MM-DD/part.parquet   (immutable pulls)
  data/curated/dataset=<d>/asset_class=<ac>/year=YYYY/part.parquet      (canonical long)
  data/reference/*.parquet
  data/panels/<kind>/...
  data/audit/*.json

Every curated write MUST carry the mandatory columns — the writer rejects anything
else. `available_from` (UTC, when the value was knowable) is the single most
important column in the project; nothing gets into the lake without it.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from production.core.config import REPO_ROOT

# obs_date + entity key + availability + provenance. Entity key is instrument_id
# for tradables, series_id for macro series — at least one must be present.
MANDATORY_COLUMNS = ("obs_date", "available_from", "source", "ingested_at")
ENTITY_COLUMNS = ("instrument_id", "series_id")


class LakeError(Exception):
    pass


class Lake:
    def __init__(self, root: str | Path | None = None):
        self.root = Path(root) if root is not None else REPO_ROOT / "data"

    # ------------------------------------------------------------------ raw
    def write_raw(self, df: pd.DataFrame, vendor: str, dataset: str,
                  ingest_date: str | pd.Timestamp) -> Path:
        d = pd.Timestamp(ingest_date).strftime("%Y-%m-%d")
        path = self.root / "raw" / f"vendor={vendor}" / f"dataset={dataset}" / f"ingest_date={d}"
        path.mkdir(parents=True, exist_ok=True)
        out = path / "part.parquet"
        df.to_parquet(out, index=False)
        return out

    # -------------------------------------------------------------- curated
    def _validate_curated(self, df: pd.DataFrame) -> None:
        missing = [c for c in MANDATORY_COLUMNS if c not in df.columns]
        if missing:
            raise LakeError(f"curated write rejected: missing mandatory columns {missing}")
        if not any(c in df.columns for c in ENTITY_COLUMNS):
            raise LakeError(
                f"curated write rejected: need one of entity columns {ENTITY_COLUMNS}")
        if df["available_from"].isna().any():
            raise LakeError("curated write rejected: available_from contains nulls")
        if df["obs_date"].isna().any():
            raise LakeError("curated write rejected: obs_date contains nulls")

    def write_curated(self, df: pd.DataFrame, dataset: str, asset_class: str) -> list[Path]:
        """Write canonical long-format data, partitioned by year of obs_date.

        Idempotent per year partition: re-runs overwrite affected year files after
        merging with existing rows (latest vintage kept per key).
        """
        self._validate_curated(df)
        df = df.copy()
        df["obs_date"] = pd.to_datetime(df["obs_date"])
        df["available_from"] = pd.to_datetime(df["available_from"], utc=True)
        entity = "instrument_id" if "instrument_id" in df.columns else "series_id"
        written = []
        for year, chunk in df.groupby(df["obs_date"].dt.year):
            path = (self.root / "curated" / f"dataset={dataset}"
                    / f"asset_class={asset_class}" / f"year={year}")
            path.mkdir(parents=True, exist_ok=True)
            out = path / "part.parquet"
            if out.exists():
                existing = pd.read_parquet(out)
                existing["obs_date"] = pd.to_datetime(existing["obs_date"])
                existing["available_from"] = pd.to_datetime(existing["available_from"], utc=True)
                chunk = pd.concat([existing, chunk], ignore_index=True)
                subset = ["obs_date", entity, "available_from"]
                if "field" in chunk.columns:
                    subset.insert(2, "field")
                chunk = (chunk.sort_values("ingested_at")
                              .drop_duplicates(subset=subset, keep="last"))
            chunk.sort_values(["obs_date", entity]).to_parquet(out, index=False)
            written.append(out)
        return written

    def read_curated(self, dataset: str, asset_class: str | None = None,
                     start=None, end=None) -> pd.DataFrame:
        base = self.root / "curated" / f"dataset={dataset}"
        if not base.exists():
            raise LakeError(f"curated dataset not found: {dataset}")
        pattern = (f"asset_class={asset_class}/year=*/part.parquet" if asset_class
                   else "asset_class=*/year=*/part.parquet")
        files = sorted(base.glob(pattern))
        if start is not None or end is not None:
            lo = pd.Timestamp(start).year if start is not None else -1
            hi = pd.Timestamp(end).year if end is not None else 10_000
            files = [f for f in files
                     if lo <= int(f.parent.name.split("=")[1]) <= hi]
        if not files:
            return pd.DataFrame()
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        df["obs_date"] = pd.to_datetime(df["obs_date"])
        df["available_from"] = pd.to_datetime(df["available_from"], utc=True)
        if start is not None:
            df = df[df["obs_date"] >= pd.Timestamp(start)]
        if end is not None:
            df = df[df["obs_date"] <= pd.Timestamp(end)]
        return df.reset_index(drop=True)

    def watermark(self, dataset: str, asset_class: str | None = None) -> pd.Timestamp | None:
        """Max obs_date in curated — drives incremental loads."""
        try:
            df = self.read_curated(dataset, asset_class)
        except LakeError:
            return None
        if df.empty:
            return None
        return df["obs_date"].max()

    # ------------------------------------------------------------ reference
    def write_reference(self, df: pd.DataFrame, name: str) -> Path:
        path = self.root / "reference"
        path.mkdir(parents=True, exist_ok=True)
        out = path / f"{name}.parquet"
        df.to_parquet(out, index=False)
        return out

    def read_reference(self, name: str) -> pd.DataFrame:
        out = self.root / "reference" / f"{name}.parquet"
        if not out.exists():
            raise LakeError(f"reference table not found: {name}")
        return pd.read_parquet(out)

    # --------------------------------------------------------------- panels
    def write_panel(self, df: pd.DataFrame, kind: str, name: str) -> Path:
        path = self.root / "panels" / kind
        path.mkdir(parents=True, exist_ok=True)
        out = path / f"{name}.parquet"
        df.to_parquet(out, index=False)
        return out

    def read_panel(self, kind: str, name: str) -> pd.DataFrame:
        out = self.root / "panels" / kind / f"{name}.parquet"
        if not out.exists():
            raise LakeError(f"panel not found: {kind}/{name}")
        return pd.read_parquet(out)

    # ---------------------------------------------------------------- audit
    def write_audit(self, record: dict, name: str) -> Path:
        path = self.root / "audit"
        path.mkdir(parents=True, exist_ok=True)
        out = path / f"{name}.json"
        with open(out, "w") as f:
            json.dump(record, f, indent=2, default=str)
        return out


# ------------------------------------------------------- signal input bundle
# The signals consume a bundle keyed by *bundle name* (data["mcap"], data["tvl"], …).
# For most sources the bundle key equals the curated dataset name the loader writes, but
# the two crypto snapshot loaders write differently-named curated datasets whose columns
# also differ from the signal contract:
#   - CoinGecko  -> curated dataset "crypto_meta", market-cap column "market_cap"
#     (keyed by instrument_id) — the mcap_tvl signal wants bundle key "mcap"/column "mcap".
#   - DefiLlama  -> curated dataset "defi_tvl",   TVL column "value"
#     (keyed by series_id "TVL_<chain>") — the signal wants bundle key "tvl"/column "tvl"
#     keyed by instrument_id. DefiLlama emits chain-level TVL as a macro *series*, so it
#     has no instrument_id and cannot feed the per-instrument "tvl" contract without a
#     chain->instrument resolution table that does not exist yet. We still map + rename it
#     best-effort (only when the curated frame carries instrument_id); until an
#     instrument-keyed TVL feed exists the "tvl" bundle key is populated only if such rows
#     are present, otherwise it is skipped like any other absent dataset.
SIGNAL_BUNDLE_DATASETS = (
    "prices", "funding", "basis", "macro", "cot", "fundamentals", "mcap", "tvl",
)

# bundle key -> curated dataset name (identity unless listed here).
_BUNDLE_TO_CURATED = {"mcap": "crypto_meta", "tvl": "defi_tvl"}

# bundle key -> {curated column: signal-contract column} normalization.
_BUNDLE_COLUMN_RENAME = {"mcap": {"market_cap": "mcap"}, "tvl": {"value": "tvl"}}


def _normalize_bundle_frame(key: str, df: pd.DataFrame) -> pd.DataFrame | None:
    """Rename curated columns to the signal-contract columns for a bundle key.

    Returns the normalized frame, or ``None`` when the curated frame cannot satisfy the
    signal contract (e.g. a TVL snapshot with no ``instrument_id`` — chain-level series
    that has no per-instrument mapping).
    """
    rename = _BUNDLE_COLUMN_RENAME.get(key)
    if rename:
        df = df.rename(columns={c: t for c, t in rename.items() if c in df.columns})
        target = next(iter(rename.values()))
        if target not in df.columns:
            return None  # curated frame lacks the value column the signal needs
        if "instrument_id" not in df.columns:
            return None  # per-instrument signal contract; series-keyed data can't feed it
    return df


def read_signal_bundle(lake: "Lake", start=None, end=None) -> dict[str, pd.DataFrame]:
    """Load every available curated dataset into the signal input bundle.

    Missing datasets are skipped silently (a lake with only prices still yields a usable,
    smaller bundle). Bundle keys are normalized to the signal contract via
    ``_BUNDLE_TO_CURATED`` / ``_BUNDLE_COLUMN_RENAME`` so ``data["mcap"]`` / ``data["tvl"]``
    carry the columns the signals expect.
    """
    bundle: dict[str, pd.DataFrame] = {}
    for key in SIGNAL_BUNDLE_DATASETS:
        curated = _BUNDLE_TO_CURATED.get(key, key)
        try:
            df = lake.read_curated(curated, start=start, end=end)
        except LakeError:
            continue
        if df is None or df.empty:
            continue
        df = _normalize_bundle_frame(key, df)
        if df is None or df.empty:
            continue
        bundle[key] = df
    return bundle
