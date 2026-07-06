"""CBOE daily put/call ratios (total, equity, index).

CBOE publishes end-of-day options volume statistics in a daily market-statistics
archive. The ratio for a trading day is finalized after the close; to stay safely on
the right side of the publication we stamp `available_from = obs_date + 1 day 12:00 UTC`
(an `obs_offset`), i.e. the day-after-observation midday. That is conservative — the
file is usually up sooner — but it can never leak a same-day ratio into a same-day
decision.

Emits macro series CBOE_TOTAL_PC always, plus CBOE_EQUITY_PC / CBOE_INDEX_PC when the
archive carries those columns.
"""
from __future__ import annotations

import pandas as pd

from production.data.base import AvailabilityRule, BaseLoader

CBOE_URL = "https://cdn.cboe.com/api/global/us_indices/daily_statistics/put_call_ratios.csv"

# Normalized (upper, stripped) column token -> internal series_id.
COL_MAP = {
    "TOTAL": "CBOE_TOTAL_PC",
    "TOTAL P/C": "CBOE_TOTAL_PC",
    "TOTAL PUT/CALL RATIO": "CBOE_TOTAL_PC",
    "EQUITY": "CBOE_EQUITY_PC",
    "EQUITY P/C": "CBOE_EQUITY_PC",
    "EQUITY PUT/CALL RATIO": "CBOE_EQUITY_PC",
    "INDEX": "CBOE_INDEX_PC",
    "INDEX P/C": "CBOE_INDEX_PC",
    "INDEX PUT/CALL RATIO": "CBOE_INDEX_PC",
}


class CboePutCallLoader(BaseLoader):
    dataset = "macro"
    vendor = "cboe"
    source = "cboe:put_call"
    asset_classes = ["macro"]
    default_asset_class = "macro"
    # Ratio finalized after the close -> knowable the next day. Stamp day+1 noon UTC.
    availability_rule = AvailabilityRule(
        "obs_offset", {"offset": pd.Timedelta(days=1, hours=12)})
    expectations = {
        "columns": ["value"],
        "ranges": {"value": (0, None)},  # ratios are non-negative
        "max_null_frac": 0.05,
        "min_rows": 1,
    }

    def __init__(self, lake=None, instruments=None, url=CBOE_URL):
        super().__init__(lake, instruments)
        self.url = url

    def fetch(self, start, end):
        import io

        import requests

        resp = requests.get(self.url, timeout=60)
        resp.raise_for_status()
        return pd.read_csv(io.StringIO(resp.text))

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
