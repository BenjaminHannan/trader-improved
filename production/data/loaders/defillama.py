"""DefiLlama chain TVL snapshot.

Stage-1-optional. Snapshot semantics -> `available_from = ingested_at`. Emitted as
macro SERIES (series_id "TVL_<chain>") — a risk-appetite / crypto-flows context input,
never a tradable. Minimal but functional.
"""
from __future__ import annotations

import pandas as pd

from production.data.base import AvailabilityRule, BaseLoader


class DefiLlamaLoader(BaseLoader):
    dataset = "defi_tvl"
    vendor = "defillama"
    source = "defillama:chains"
    asset_classes = ["macro"]
    default_asset_class = "macro"
    availability_rule = AvailabilityRule("ingest_time")
    expectations = {"columns": ["value"], "ranges": {"value": (0, None)}, "min_rows": 1}

    def fetch(self, start, end) -> list[dict]:
        import requests

        resp = requests.get("https://api.llama.fi/v2/chains", timeout=30)
        resp.raise_for_status()
        return resp.json()

    def transform(self, raw) -> pd.DataFrame:
        today = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
        rows = []
        for rec in raw:
            name = rec.get("name")
            tvl = rec.get("tvl")
            if name is None or tvl is None:
                continue
            rows.append({"obs_date": today, "series_id": f"TVL_{name}",
                         "value": float(tvl), "asset_class": "macro"})
        return pd.DataFrame(rows)
