"""Primary daily equity/ETF prices from Yahoo Finance via yfinance.

Availability: Yahoo's adjusted daily bar for session D is settled after the US close;
we stamp `available_from = obs_date 21:30 UTC` (~16:30 ET), a conservative post-close
time by which the consolidated close is known. auto_adjust=True so `close` is already
total-return adjusted (splits + dividends), which is what every price-based signal
wants.
"""
from __future__ import annotations

import pandas as pd

from production.data.base import (AvailabilityRule, BaseLoader, prices_to_long)


class YFinancePricesLoader(BaseLoader):
    dataset = "prices"
    vendor = "yfinance"
    source = "yfinance:prices"
    asset_classes = ["equity", "fx", "commodity"]
    availability_rule = AvailabilityRule("obs_offset",
                                         {"offset": pd.Timedelta(hours=21, minutes=30)})
    expectations = {
        "columns": ["close", "volume", "dollar_volume"],
        "ranges": {"close": (0, None), "volume": (0, None)},
        "max_null_frac": 0.02,
        "min_rows": 1,
    }

    def __init__(self, lake=None, instruments=None, symbols=None, batch_size=50):
        super().__init__(lake, instruments)
        self.symbols = symbols or []
        self.batch_size = batch_size

    def fetch(self, start, end):
        import yfinance as yf

        frames = []
        for i in range(0, len(self.symbols), self.batch_size):
            batch = self.symbols[i:i + self.batch_size]
            data = yf.download(batch, start=pd.Timestamp(start), end=pd.Timestamp(end),
                               auto_adjust=True, group_by="ticker", progress=False,
                               threads=True)
            frames.append(data)
        if not frames:
            return pd.DataFrame()
        return frames[0] if len(frames) == 1 else pd.concat(frames, axis=1)

    def transform(self, raw) -> pd.DataFrame:
        return prices_to_long(raw, lambda s: self.resolve(s, "yfinance"))
