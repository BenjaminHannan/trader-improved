"""Fallback daily prices from Stooq.

Used when Yahoo is unavailable or a symbol is missing there. Same canonical schema
and the same 21:30 UTC availability stamp as the yfinance loader — Stooq's daily bar
for session D is likewise a post-close figure. Fetch tries pandas_datareader's stooq
reader first, then falls back to the direct CSV endpoint per symbol.
"""
from __future__ import annotations

import io

import pandas as pd

from production.data.base import AvailabilityRule, BaseLoader, prices_to_long


class StooqPricesLoader(BaseLoader):
    dataset = "prices"
    vendor = "stooq"
    source = "stooq:prices"
    asset_classes = ["equity", "fx", "commodity"]
    availability_rule = AvailabilityRule("obs_offset",
                                         {"offset": pd.Timedelta(hours=21, minutes=30)})
    expectations = {
        "columns": ["close", "volume", "dollar_volume"],
        "ranges": {"close": (0, None), "volume": (0, None)},
        "max_null_frac": 0.02,
        "min_rows": 1,
    }

    def __init__(self, lake=None, instruments=None, symbols=None):
        super().__init__(lake, instruments)
        # Stooq US symbols carry a ".us" suffix, e.g. "aapl.us".
        self.symbols = symbols or []

    def fetch(self, start, end) -> dict[str, pd.DataFrame]:
        out: dict[str, pd.DataFrame] = {}
        for sym in self.symbols:
            df = self._fetch_one(sym, start, end)
            if df is not None and not df.empty:
                out[sym] = df
        return out

    def _fetch_one(self, sym, start, end) -> pd.DataFrame | None:
        try:
            import pandas_datareader.data as web

            df = web.DataReader(sym, "stooq", pd.Timestamp(start), pd.Timestamp(end))
            return df.sort_index()
        except Exception:
            pass
        try:  # direct CSV endpoint
            import requests

            url = f"https://stooq.com/q/d/l/?s={sym}&i=d"
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            df = pd.read_csv(io.StringIO(resp.text), parse_dates=["Date"]).set_index("Date")
            return df.loc[str(pd.Timestamp(start).date()):str(pd.Timestamp(end).date())]
        except Exception:
            return None

    def transform(self, raw) -> pd.DataFrame:
        return prices_to_long(raw, lambda s: self.resolve(s, "stooq"))
