"""CoinGecko market snapshot (market cap + 24h volume).

Stage-1-optional. A point-in-time snapshot has no natural observation lag — the
numbers are current as of the API call — so `available_from = ingested_at`. Used later
for crypto market-cap weighting and a value-style mcap signal; kept minimal here.
"""
from __future__ import annotations

import os

import pandas as pd

from production.data.base import AvailabilityRule, BaseLoader


class CoinGeckoLoader(BaseLoader):
    dataset = "crypto_meta"
    vendor = "coingecko"
    source = "coingecko:markets"
    asset_classes = ["crypto"]
    default_asset_class = "crypto"
    availability_rule = AvailabilityRule("ingest_time")
    expectations = {"columns": ["market_cap", "volume"], "min_rows": 1}

    def __init__(self, lake=None, instruments=None, vs_currency="usd", per_page=100,
                 api_key=None):
        super().__init__(lake, instruments)
        self.vs_currency = vs_currency
        self.per_page = per_page
        # Optional demo API key: raises the free-tier rate limit. Read from env only,
        # never hardcoded; absence just means unauthenticated (public) access.
        self.api_key = api_key if api_key is not None else os.environ.get("COINGECKO_API_KEY")

    def fetch(self, start, end) -> list[dict]:
        import requests

        headers = {"x-cg-demo-api-key": self.api_key} if self.api_key else None
        resp = requests.get(
            "https://api.coingecko.com/api/v3/coins/markets",
            params={"vs_currency": self.vs_currency, "order": "market_cap_desc",
                    "per_page": self.per_page, "page": 1},
            headers=headers, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def transform(self, raw) -> pd.DataFrame:
        today = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
        rows = []
        for rec in raw:
            sym = str(rec.get("symbol", "")).upper()
            iid = self.resolve(sym, "coingecko") or self.resolve(sym)
            if iid is None:
                continue
            rows.append({
                "obs_date": today,
                "instrument_id": iid,
                "market_cap": rec.get("market_cap"),
                "volume": rec.get("total_volume"),
                "asset_class": "crypto",
            })
        return pd.DataFrame(rows)
