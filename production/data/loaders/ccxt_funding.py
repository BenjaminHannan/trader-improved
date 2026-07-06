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

from production.data.base import AvailabilityRule, BaseLoader, IngestError

# Tried in order when no explicit exchange is given. binanceusdm is deliberately NOT
# in the default chain: it (and several other venues) geo-block US IPs, degrading
# every symbol and producing a column-less frame that trips the schema audit with an
# unhelpful message. bybit/okx/kraken publish perpetual funding history and are
# reachable from more regions; the first venue that returns data for a symbol wins.
DEFAULT_EXCHANGE_CHAIN = ["bybit", "okx", "kraken"]


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

    def __init__(self, lake=None, instruments=None, symbols=None, exchange=None,
                 exchanges=None):
        super().__init__(lake, instruments)
        self.symbols = symbols or []          # e.g. ["BTC/USDT:USDT"]
        # An explicit `exchange` pins the venue (overriding the chain); otherwise we
        # walk `exchanges` (default chain) trying each unresolved symbol in turn.
        self.exchange = exchange
        self.exchanges = ([exchange] if exchange
                          else (exchanges or list(DEFAULT_EXCHANGE_CHAIN)))

    def fetch(self, start, end) -> dict[str, list]:
        import ccxt

        since = int(pd.Timestamp(start).tz_localize("UTC").timestamp() * 1000)
        out: dict[str, list] = {}
        remaining = list(self.symbols)
        tried: list[str] = []
        for name in self.exchanges:
            if not remaining:
                break
            try:
                ex = getattr(ccxt, name)()
            except Exception as exc:  # unknown/broken exchange id — try the next one
                self.warnings.append(f"funding: exchange {name!r} unavailable: {exc!r}")
                continue
            tried.append(name)
            still: list[str] = []
            for sym in remaining:
                try:
                    hist = ex.fetch_funding_rate_history(sym, since=since)
                except Exception:
                    still.append(sym)   # not listed / geo-blocked here — retry next venue
                    continue
                if hist:
                    out[sym] = hist
                else:
                    still.append(sym)
            remaining = still

        if not out:
            raise IngestError(
                "funding/ccxt: every symbol failed across exchanges "
                f"{tried or self.exchanges}. Perpetual-funding venues are frequently "
                "geo-blocked from US IPs (binanceusdm in particular). Retry from an "
                "unblocked network/VPN, pass a reachable --exchange, or run "
                "`python scripts/diagnose_vendors.py` to see which venues respond. "
                "See `python scripts/ingest.py --help`.")
        if remaining:  # partial success: name the symbols no venue could serve
            self.warnings.append(
                f"funding: no exchange in {tried} served {remaining}")
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
