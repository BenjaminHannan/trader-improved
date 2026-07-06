"""NAAIM Exposure Index — active manager equity exposure survey.

The National Association of Active Investment Managers polls its members on their
current equity exposure. The survey is dated to **Wednesday** but the aggregate is
not published until **Thursday**, so a Wednesday-dated reading is a look-ahead trap
if used on the Wednesday itself. We stamp `available_from` with a `next_weekday_time`
rule at Thursday 21:00 UTC (the first Thursday on/after the Wednesday obs_date), which
sits safely after the US-afternoon release.

Emitted as macro series NAAIM_EXPOSURE (mean/average member exposure, in percent).
"""
from __future__ import annotations

from production.data.base import AvailabilityRule, BaseLoader

NAAIM_URL = "https://www.naaim.org/programs/naaim-exposure-index/"
SERIES_ID = "NAAIM_EXPOSURE"
# Candidate column names for the headline mean-exposure figure, most specific first.
_VALUE_KEYS = ("NAAIM Number", "NAAIM Exposure Index", "Mean/Average", "Mean", "Average")


class NaaimLoader(BaseLoader):
    dataset = "macro"
    vendor = "naaim"
    source = "naaim:exposure_index"
    asset_classes = ["macro"]
    default_asset_class = "macro"
    # Wednesday obs, Thursday release -> first Thursday on/after obs at 21:00 UTC.
    availability_rule = AvailabilityRule("next_weekday_time",
                                         {"weekday": 3, "hour": 21, "minute": 0})
    expectations = {"columns": ["value"], "max_null_frac": 0.05, "min_rows": 1}

    def __init__(self, lake=None, instruments=None, url=NAAIM_URL):
        super().__init__(lake, instruments)
        self.url = url

    def fetch(self, start, end):
        import io

        import pandas as pd
        import requests

        resp = requests.get(self.url, timeout=60)
        resp.raise_for_status()
        # The page exposes a CSV/XLS export; fall back to parsing HTML tables.
        try:
            return pd.read_csv(io.StringIO(resp.text))
        except Exception:
            return pd.read_html(resp.text)[0]

    def transform(self, raw):
        import pandas as pd

        df = raw.copy() if isinstance(raw, pd.DataFrame) else pd.DataFrame(raw)
        if df.empty:
            return pd.DataFrame(columns=["obs_date", "series_id", "value", "asset_class"])
        date_col = next((c for c in df.columns if "date" in str(c).lower()), df.columns[0])
        val_col = next((c for c in df.columns if str(c).strip() in _VALUE_KEYS), None)
        if val_col is None:  # last resort: last numeric-looking column
            val_col = df.columns[-1]
        obs = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()
        value = pd.to_numeric(df[val_col], errors="coerce")
        out = pd.DataFrame({"obs_date": obs, "series_id": SERIES_ID,
                            "value": value, "asset_class": "macro"})
        return out.dropna(subset=["obs_date", "value"]).reset_index(drop=True)
