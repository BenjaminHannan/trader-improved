"""FINRA bi-monthly consolidated short-interest.

FINRA publishes short interest twice a month. The number is observed as of a
**settlement date** but is not disseminated until roughly nine business days later.
Getting that embargo wrong is a textbook look-ahead error, so we stamp `available_from`
conservatively at `settlement date + 13 calendar days 22:00 UTC` — comfortably past
the ~9-business-day publication lag regardless of how weekends and holidays fall in
the window.

Each symbol becomes its own macro-style series (series_id "FINRA_SI:<symbol>", value =
current short-position quantity in shares), so the point-in-time entity model (series_id
keyed rows) carries them without needing tradable instrument resolution.
"""
from __future__ import annotations

import pandas as pd

from production.data.base import AvailabilityRule, BaseLoader

FINRA_URL = "https://api.finra.org/data/group/otcMarket/name/consolidatedShortInterest"


def _first(rec: dict, *keys):
    for k in keys:
        if k in rec and rec[k] not in (None, ""):
            return rec[k]
    return None


class FinraShortInterestLoader(BaseLoader):
    dataset = "macro"
    vendor = "finra"
    source = "finra:short_interest"
    asset_classes = ["macro"]
    default_asset_class = "macro"
    # Settlement-date obs, ~9 business day publication lag -> conservative 13 calendar
    # days + 22:00 UTC so no weekend/holiday arrangement can leak the print early.
    availability_rule = AvailabilityRule(
        "obs_offset", {"offset": pd.Timedelta(days=13, hours=22)})
    expectations = {
        "columns": ["value"],
        "ranges": {"value": (0, None)},  # short shares are non-negative
        "max_null_frac": 0.05,
        "min_rows": 1,
    }

    def __init__(self, lake=None, instruments=None, symbols=None, url=FINRA_URL):
        super().__init__(lake, instruments)
        self.symbols = symbols  # optional filter; None = keep everything returned
        self.url = url

    def fetch(self, start, end):
        import requests

        resp = requests.get(self.url, timeout=60)
        resp.raise_for_status()
        return resp.json()

    def transform(self, raw):
        records = raw.to_dict("records") if isinstance(raw, pd.DataFrame) else list(raw)
        keep = {s.upper() for s in self.symbols} if self.symbols else None
        rows = []
        for rec in records:
            sym = _first(rec, "symbolCode", "symbol", "issueSymbolIdentifier")
            date = _first(rec, "settlementDate", "settlement_date")
            qty = _first(rec, "currentShortPositionQuantity",
                         "current_short_position_quantity", "shortInterest")
            if sym is None or date is None or qty is None:
                continue
            sym = str(sym).upper()
            if keep is not None and sym not in keep:
                continue
            rows.append({
                "obs_date": pd.Timestamp(str(date)).normalize(),
                "series_id": f"FINRA_SI:{sym}",
                "value": pd.to_numeric(qty, errors="coerce"),
                "asset_class": "macro",
            })
        if not rows:
            return pd.DataFrame(columns=["obs_date", "series_id", "value", "asset_class"])
        return pd.DataFrame(rows).dropna(subset=["value"]).reset_index(drop=True)
