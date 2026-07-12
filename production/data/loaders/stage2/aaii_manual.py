"""AAII Investor Sentiment Survey — manual (paid) drop-in.

The AAII sentiment history is a paid dataset, so we do NOT fetch it over the network.
Instead the user places the spreadsheet at `data/manual/sentiment.xls` and this loader
reads it locally; if the file is absent `fetch` raises `IngestError` with an explicit
instruction rather than silently producing nothing.

The survey closes Wednesday and the weekly result is published Thursday, so we stamp
`available_from` at Thursday 12:00 UTC (the first Thursday on/after the Thursday-dated
obs), keeping the reading off-limits until its release. Emitted as macro series
AAII_BULL / AAII_NEUTRAL / AAII_BEAR (fractions of respondents).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from production.core.config import REPO_ROOT
from production.data.base import AvailabilityRule, BaseLoader, IngestError

DEFAULT_PATH = REPO_ROOT / "data" / "manual" / "sentiment.xls"

# Vendor column token (upper, stripped) -> internal series_id.
COL_MAP = {
    "BULLISH": "AAII_BULL",
    "NEUTRAL": "AAII_NEUTRAL",
    "BEARISH": "AAII_BEAR",
}


class AaiiManualLoader(BaseLoader):
    dataset = "macro"
    vendor = "aaii"
    source = "aaii:sentiment_survey"
    asset_classes = ["macro"]
    default_asset_class = "macro"
    # Thursday-published weekly survey -> first Thursday on/after obs at 12:00 UTC.
    availability_rule = AvailabilityRule("next_weekday_time",
                                         {"weekday": 3, "hour": 12, "minute": 0})
    expectations = {"columns": ["value"], "max_null_frac": 0.05, "min_rows": 1}

    def __init__(self, lake=None, instruments=None, path=DEFAULT_PATH):
        super().__init__(lake, instruments)
        self.path = Path(path)

    def fetch(self, start, end):
        if not self.path.exists():
            raise IngestError(
                f"AAII sentiment file not found at {self.path}. This is a paid dataset; "
                "place the exported spreadsheet at data/manual/sentiment.xls to ingest it.")
        return pd.read_excel(self.path)

    def transform(self, raw):
        df = raw.copy() if isinstance(raw, pd.DataFrame) else pd.DataFrame(raw)
        if df.empty:
            return pd.DataFrame(columns=["obs_date", "series_id", "value", "asset_class"])
        date_col = next((c for c in df.columns if "date" in str(c).lower()), df.columns[0])
        obs = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()

        frames = []
        for col in df.columns:
            sid = COL_MAP.get(str(col).strip().upper())
            if sid is None:
                continue
            frames.append(pd.DataFrame({
                "obs_date": obs,
                "series_id": sid,
                "value": pd.to_numeric(df[col], errors="coerce"),
                "asset_class": "macro",
            }))
        if not frames:
            return pd.DataFrame(columns=["obs_date", "series_id", "value", "asset_class"])
        out = pd.concat(frames, ignore_index=True)
        return out.dropna(subset=["obs_date", "value"]).reset_index(drop=True)
