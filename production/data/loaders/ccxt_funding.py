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
        until = int(pd.Timestamp(end).tz_localize("UTC").timestamp() * 1000)
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
                    hist = self._fetch_paginated(ex, sym, since, until)
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

    @staticmethod
    def _fetch_paginated(ex, sym, since: int, until: int,
                         max_pages: int = 400) -> list:
        """Walk fetch_funding_rate_history forward from `since` until `until`.

        Venues cap a single call (bybit ~200 cycles ≈ 66 days at 8h), so one call from
        2016 silently returns only the most recent window. Advance `since` past the
        last returned timestamp each page; stop on an empty page, no forward progress,
        or the `until` bound. First page failing raises (caller falls through to the
        next venue); a later page failing returns what we have (partial > nothing)."""
        import time

        _probe_step_ms = 180 * 24 * 3600 * 1000   # 180d: pre-listing gap probe stride
        all_rows: list = []
        cursor = since
        for page in range(max_pages):
            # Pace beyond ccxt's own throttle: a universe of symbols × pages/probes
            # adds up fast, and both bybit and okx IP-ban aggressive callers
            # (learned the hard way on the 2026-07-07 live ingest).
            time.sleep(0.25)
            try:
                batch = ex.fetch_funding_rate_history(sym, since=cursor)
            except Exception:
                if page == 0:
                    raise
                break
            # Drop boundary overlap / venues that ignore `since`; only forward rows
            # count as progress, so a stale repeat page terminates the walk.
            new = [r for r in batch if int(r["timestamp"]) >= cursor]
            if not new:
                if all_rows:
                    break             # walked off the end of the history
                # Venues (bybit) return EMPTY when `since` predates the perp's listing
                # instead of clamping. Probe forward until the series starts.
                cursor += _probe_step_ms
                if cursor >= until:
                    break
                continue
            all_rows.extend(new)
            last = max(int(r["timestamp"]) for r in new)
            if last >= until:
                break
            cursor = last + 1
        return all_rows

    def transform(self, raw) -> pd.DataFrame:
        frames = []
        for sym, hist in raw.items():
            # The master's ccxt vendor symbol is the spot pair (e.g. "BTC/USD"); the
            # funding fetch addresses the perp ("BTC/USDT:USDT"). Try the raw symbol
            # first (tests/fixtures may key on it), then the base's spot pair.
            iid = self.resolve(sym, "ccxt")
            if iid is None and "/" in sym:
                iid = self.resolve(f"{sym.split('/', 1)[0]}/USD", "ccxt")
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
