"""Daily crypto prices from a centralized exchange via ccxt.

Crypto trades a 7-day week on midnight-UTC daily bars. The bar labelled obs_date D
covers [D 00:00, D+1 00:00) UTC and is only *complete* at its close, so the value is
knowable at D + 24h: `available_from = obs_date + 1 day`. Default venue Kraken, with
Coinbase as a fallback if Kraken lacks a pair.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from production.data.base import (AvailabilityRule, BaseLoader,
                                  asset_class_from_instrument_id,
                                  fetch_ohlcv_paginated)

# A venue whose earliest returned bar is this far after the requested start is
# serving a truncated retention window (kraken: last ~720 bars regardless of
# `since`), so the fallback venue is consulted for deeper history.
_DEPTH_SLACK_MS = 90 * 24 * 3600 * 1000


class CcxtPricesLoader(BaseLoader):
    dataset = "prices"
    vendor = "ccxt"
    source = "ccxt:prices"
    asset_classes = ["crypto"]
    availability_rule = AvailabilityRule("obs_offset", {"offset": pd.Timedelta(hours=24)})
    expectations = {
        "columns": ["close", "volume", "dollar_volume"],
        "ranges": {"close": (0, None), "volume": (0, None)},
        "max_null_frac": 0.02,
        "min_rows": 1,
    }

    def __init__(self, lake=None, instruments=None, symbols=None,
                 exchange="kraken", fallback="coinbase"):
        super().__init__(lake, instruments)
        self.symbols = symbols or []          # e.g. ["BTC/USD", "ETH/USD"]
        self.exchange = exchange
        self.fallback = fallback

    def fetch(self, start, end) -> dict[str, list]:
        import ccxt

        since = int(pd.Timestamp(start).tz_localize("UTC").timestamp() * 1000)
        primary = getattr(ccxt, self.exchange)()
        backup = getattr(ccxt, self.fallback)() if self.fallback else None
        out: dict[str, list] = {}
        for sym in self.symbols:
            best: list = []
            for ex in (primary, backup):
                if ex is None:
                    continue
                try:
                    bars = fetch_ohlcv_paginated(ex, sym, since)
                except Exception:
                    continue
                # Keep the deepest series across venues: a truncated-retention venue
                # (kraken serves only its last ~720 daily bars) must not shadow a
                # fallback that can reach the requested start.
                if bars and (not best or bars[0][0] < best[0][0]):
                    best = bars
                if best and best[0][0] <= since + _DEPTH_SLACK_MS:
                    break            # deep enough — no need to consult the fallback
            if best:
                out[sym] = best
        return out

    def transform(self, raw) -> pd.DataFrame:
        frames = []
        for sym, bars in raw.items():
            iid = self.resolve(sym, "ccxt")
            if iid is None or not len(bars):
                continue
            arr = pd.DataFrame(bars, columns=["ts", "open", "high", "low", "close", "volume"])
            obs = pd.to_datetime(arr["ts"], unit="ms", utc=True).dt.tz_localize(None).dt.normalize()
            f = pd.DataFrame({
                "obs_date": obs,
                "instrument_id": iid,
                "close": pd.to_numeric(arr["close"], errors="coerce"),
                "volume": pd.to_numeric(arr["volume"], errors="coerce"),
            })
            f["dollar_volume"] = f["close"] * f["volume"]
            f["asset_class"] = asset_class_from_instrument_id(iid)
            frames.append(f)
        if not frames:
            return pd.DataFrame(columns=["obs_date", "instrument_id", "close", "volume",
                                         "dollar_volume", "asset_class"])
        return pd.concat(frames, ignore_index=True).dropna(subset=["close"]).reset_index(drop=True)
