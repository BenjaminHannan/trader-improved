"""Fallback daily prices from Stooq.

Used when Yahoo is unavailable or a symbol is missing there. Same canonical schema
as the yfinance loader, stamped 21:15 UTC — deliberately EARLIER than yfinance's
21:30. Both are honest post-close times, and the 15-minute gap encodes feed
priority in the availability algebra itself: the curated lake dedups on
(obs_date, instrument_id, available_from), so both vendors' rows coexist (enabling
production/data/cross_check.py), while asof_panel's latest-visible-vintage rule
makes yfinance win whenever both are present and stooq fill the gaps — exactly
fallback semantics, with no special-case code. Fetch tries pandas_datareader's
stooq reader first, then falls back to the direct CSV endpoint per symbol.
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
                                         {"offset": pd.Timedelta(hours=21, minutes=15)})
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
