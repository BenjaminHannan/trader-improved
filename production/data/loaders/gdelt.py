"""GDELT DOC 2.0 average tone by query.

Stage-1-optional. The timeline/tone endpoint returns per-day average tone for a news
query; we emit it as a macro SERIES (series_id "GDELT_TONE_<query>"). Tone dates are
reported after the day closes; a snapshot pull is stamped `available_from =
ingested_at` (conservative for a same-day sentiment read). Minimal but functional.
"""
from __future__ import annotations

import pandas as pd

from production.data.base import AvailabilityRule, BaseLoader


class GdeltLoader(BaseLoader):
    dataset = "gdelt_tone"
    vendor = "gdelt"
    source = "gdelt:doc_tone"
    asset_classes = ["macro"]
    default_asset_class = "macro"
    availability_rule = AvailabilityRule("ingest_time")
    expectations = {"columns": ["value"], "min_rows": 1}

    def __init__(self, lake=None, instruments=None, query="markets"):
        super().__init__(lake, instruments)
        self.query = query

    def fetch(self, start, end) -> dict:
        import requests

        resp = requests.get(
            "https://api.gdeltproject.org/api/v2/doc/doc",
            params={"query": self.query, "mode": "timelinetone",
                    "format": "json",
                    "startdatetime": pd.Timestamp(start).strftime("%Y%m%d000000"),
                    "enddatetime": pd.Timestamp(end).strftime("%Y%m%d000000")},
            timeout=30)
        resp.raise_for_status()
        return resp.json()

    def transform(self, raw) -> pd.DataFrame:
        sid = f"GDELT_TONE_{self.query.replace(' ', '_')}"
        series = raw.get("timeline", [])
        points = series[0].get("data", []) if series else []
        rows = []
        for pt in points:
            rows.append({"obs_date": pd.Timestamp(pt["date"]).normalize(),
                         "series_id": sid, "value": float(pt.get("value", 0.0)),
                         "asset_class": "macro"})
        return pd.DataFrame(rows)
