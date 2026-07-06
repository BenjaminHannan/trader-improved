"""Perpetual-swap funding rates from a derivatives exchange via ccxt.

Funding is paid on a cycle (typically every 8h) and each payment carries a settlement
timestamp. We aggregate the intra-day cycles to a daily mean `funding_rate` keyed on
the funding date, and stamp `available_from` explicitly from the *last* funding
timestamp of that day — the moment the day's funding is fully known. This is more
honest than a fixed offset because the exact cycle times vary by venue. Symbols that
a venue doesn't list are skipped (degrade gracefully) rather than failing the pull.
"""
from __future__ import annotations

import pandas as pd

from production.data.base import AvailabilityRule, BaseLoader


class CcxtFundingLoader(BaseLoader):
    dataset = "funding"
    vendor = "ccxt"
    source = "ccxt:funding"
    asset_classes = ["crypto"]
    availability_rule = AvailabilityRule("explicit", {"column": "available_from"})
    expectations = {
        "columns": ["funding_rate"],
        "ranges": {"funding_rate": (-0.05, 0.05)},  # sane per-cycle band
        "max_null_frac": 0.02,
        "min_rows": 1,
    }

    def __init__(self, lake=None, instruments=None, symbols=None, exchange="binanceusdm"):
        super().__init__(lake, instruments)
        self.symbols = symbols or []          # e.g. ["BTC/USDT:USDT"]
        self.exchange = exchange

    def fetch(self, start, end) -> dict[str, list]:
        import ccxt

        since = int(pd.Timestamp(start).tz_localize("UTC").timestamp() * 1000)
        ex = getattr(ccxt, self.exchange)()
        out: dict[str, list] = {}
        for sym in self.symbols:
            try:
                out[sym] = ex.fetch_funding_rate_history(sym, since=since)
            except Exception:
                continue  # perpetual not listed for this symbol
        return out

    def transform(self, raw) -> pd.DataFrame:
        frames = []
        for sym, hist in raw.items():
            iid = self.resolve(sym, "ccxt")
            if iid is None or not hist:
                continue
            d = pd.DataFrame(hist)
            ts = pd.to_datetime(d["timestamp"], unit="ms", utc=True)
            d = d.assign(_ts=ts, _date=ts.dt.tz_localize(None).dt.normalize(),
                         _fr=pd.to_numeric(d["fundingRate"], errors="coerce"))
            g = d.groupby("_date").agg(funding_rate=("_fr", "mean"),
                                       available_from=("_ts", "max")).reset_index()
            g = g.rename(columns={"_date": "obs_date"})
            g["instrument_id"] = iid
            g["asset_class"] = "crypto"
            frames.append(g)
        if not frames:
            return pd.DataFrame(columns=["obs_date", "instrument_id", "funding_rate",
                                         "available_from", "asset_class"])
        return pd.concat(frames, ignore_index=True).dropna(subset=["funding_rate"])
