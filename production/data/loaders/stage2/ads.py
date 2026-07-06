"""Philadelphia Fed Aros-Diebold-Scotti (ADS) business-conditions index.

REVISION CAVEAT — READ THIS. The ADS index is a filtered state estimate: **every**
value in the whole history is re-estimated on **every** update. The xlsx the Philly
Fed publishes is a single "most current vintage" file, so the numbers we pull today
are NOT what a market participant saw on the historical dates they are stamped to.
Only the vintage we actually pulled is point-in-time trustworthy.

Because the entire series is revised at each release and the file carries no per-row
vintage, the honest availability stamp is `ingest_time`: every row became "knowable"
(in this revised form) only at the moment we downloaded it. That deliberately prevents
the revised history from being used as if it had been available in the past. A truly
PIT ADS backtest would need the archived vintage files; that is a Stage-3 upgrade.
"""
from __future__ import annotations

from production.data.base import AvailabilityRule, BaseLoader

ADS_URL = ("https://www.philadelphiafed.org/-/media/frbp/assets/surveys-and-data/"
           "ads/ads_index_most_current_vintage.xlsx")
SERIES_ID = "ADS_INDEX"


class AdsLoader(BaseLoader):
    dataset = "macro"
    vendor = "philadelphia_fed"
    source = "philadelphia_fed:ads"
    asset_classes = ["macro"]
    default_asset_class = "macro"
    # Whole history revised each release + no per-row vintage -> only the pull instant
    # is trustworthy. See the module docstring.
    availability_rule = AvailabilityRule("ingest_time")
    expectations = {"columns": ["value"], "max_null_frac": 0.05, "min_rows": 1}

    def __init__(self, lake=None, instruments=None, url=ADS_URL):
        super().__init__(lake, instruments)
        self.url = url

    def fetch(self, start, end):
        import io

        import pandas as pd
        import requests

        resp = requests.get(self.url, timeout=60)
        resp.raise_for_status()
        return pd.read_excel(io.BytesIO(resp.content))

    def transform(self, raw):
        import pandas as pd

        df = raw.copy() if isinstance(raw, pd.DataFrame) else pd.DataFrame(raw)
        if df.empty:
            return pd.DataFrame(columns=["obs_date", "series_id", "value", "asset_class"])
        # First column is the observation date; the ADS value column contains "ads".
        date_col = df.columns[0]
        val_col = next((c for c in df.columns if "ads" in str(c).lower()), df.columns[-1])
        obs = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()
        value = pd.to_numeric(df[val_col], errors="coerce")
        out = pd.DataFrame({"obs_date": obs, "series_id": SERIES_ID,
                            "value": value, "asset_class": "macro"})
        return out.dropna(subset=["obs_date", "value"]).reset_index(drop=True)
