"""ECB reference FX rates via the Frankfurter API (free, keyless, ECB-sourced).

These are macro SERIES (series_id like "ECB_EURUSD"), not tradables — the FX sleeve
trades ETF proxies, and these reference rates are signal inputs (rate-differential
carry, macro context). The ECB publishes its daily reference rates around 16:00 CET;
we stamp a conservative `available_from = obs_date 15:00 UTC`.

Frankfurter with `from=USD` returns units of the counter currency per USD; we invert
to the conventional CCYUSD quote (USD per unit of the foreign currency).
"""
from __future__ import annotations

import pandas as pd

from production.data.base import AvailabilityRule, BaseLoader


class FrankfurterFxLoader(BaseLoader):
    dataset = "macro"
    vendor = "frankfurter"
    source = "frankfurter:ecb_fx"
    asset_classes = ["macro"]
    default_asset_class = "macro"
    availability_rule = AvailabilityRule("obs_offset", {"offset": pd.Timedelta(hours=15)})
    expectations = {
        "columns": ["value"],
        "ranges": {"value": (0, None)},
        "max_null_frac": 0.02,
        "min_rows": 1,
    }

    def __init__(self, lake=None, instruments=None, currencies=None):
        super().__init__(lake, instruments)
        self.currencies = currencies or ["EUR", "JPY", "GBP", "AUD", "CAD", "CHF"]

    def fetch(self, start, end) -> dict:
        import requests

        s, e = pd.Timestamp(start).date(), pd.Timestamp(end).date()
        url = f"https://api.frankfurter.app/{s}..{e}"
        resp = requests.get(url, params={"from": "USD", "to": ",".join(self.currencies)},
                            timeout=30)
        resp.raise_for_status()
        return resp.json()

    def transform(self, raw) -> pd.DataFrame:
        rates = raw.get("rates", {})
        rows = []
        for date, per_ccy in rates.items():
            for ccy, usd_per_unit in per_ccy.items():
                if not usd_per_unit:
                    continue
                rows.append({
                    "obs_date": pd.Timestamp(date),
                    "series_id": f"ECB_{ccy}USD",
                    "value": 1.0 / float(usd_per_unit),  # -> USD per unit of CCY
                })
        df = pd.DataFrame(rows)
        if not df.empty:
            df["asset_class"] = "macro"
        return df
