"""Daily perp-vs-spot basis for crypto via ccxt.

The cross-sectional crypto carry signal is *basis*: how far the perpetual swap trades
above (contango) or below (backwardation) spot. For each configured base symbol we
fetch daily spot bars (``SYM/USD``, falling back to ``SYM/USDT``) and daily perpetual
swap bars (``SYM/USDT:USDT``) from the same venue, align them on the midnight-UTC bar
date, and compute

    basis_D = perp_close_D / spot_close_D - 1.

A daily bar labelled ``obs_date`` D covers [D 00:00, D+1 00:00) UTC and is only
*complete* at its close, so the value is knowable at D + 24h — the same availability
rule as ccxt daily prices. Default venue bybit, with okx as a fallback. A symbol whose
spot or swap leg a venue does not list is skipped with an audit warning (degrade
per-symbol) rather than failing the whole pull.
"""
from __future__ import annotations

import pandas as pd

from production.data.base import AvailabilityRule, BaseLoader, asset_class_from_instrument_id

# A candidate quote-symbol whose earliest returned bar is this far after the
# requested start is deep enough to stop trying further candidates (mirrors
# ccxt_prices.CcxtPricesLoader's _DEPTH_SLACK_MS early-exit).
_DEPTH_SLACK_MS = 90 * 24 * 3600 * 1000


class CcxtPerpBasisLoader(BaseLoader):
    dataset = "basis"
    vendor = "ccxt"
    source = "ccxt:basis"
    asset_classes = ["crypto"]
    availability_rule = AvailabilityRule("obs_offset", {"offset": pd.Timedelta(hours=24)})
    expectations = {
        "columns": ["basis"],
        "ranges": {"basis": (-0.5, 0.5)},  # a >50% perp/spot dislocation is not real data
        "max_null_frac": 0.02,
        "min_rows": 1,
    }

    def __init__(self, lake=None, instruments=None, symbols=None,
                 exchange="bybit", fallback="okx"):
        super().__init__(lake, instruments)
        self.symbols = symbols or []          # base symbols, e.g. ["BTC", "ETH"]
        self.exchange = exchange
        self.fallback = fallback

    # ------------------------------------------------------------------ fetch
    def fetch(self, start, end) -> dict[str, dict]:
        import ccxt

        since = int(pd.Timestamp(start).tz_localize("UTC").timestamp() * 1000)
        primary = getattr(ccxt, self.exchange)()
        backup = getattr(ccxt, self.fallback)() if self.fallback else None
        out: dict[str, dict] = {}
        for sym in self.symbols:
            pair = self._fetch_symbol(sym, since, primary, backup)
            if pair is not None:
                out[sym] = pair
            else:
                self.warnings.append(
                    f"basis: skipped {sym} — spot/swap OHLCV unavailable on "
                    f"{self.exchange}/{self.fallback}")
        return out

    def _fetch_symbol(self, sym, since, primary, backup) -> dict | None:
        """Return {"spot": bars, "swap": bars} from the first venue that lists both."""
        for ex in (primary, backup):
            if ex is None:
                continue
            spot = self._try_ohlcv(ex, [f"{sym}/USD", f"{sym}/USDT"], since)
            swap = self._try_ohlcv(ex, [f"{sym}/USDT:USDT"], since)
            if spot and swap:
                return {"spot": spot, "swap": swap}
        return None

    @staticmethod
    def _try_ohlcv(ex, symbols, since) -> list | None:
        """Return the DEEPEST bar series across the candidate vendor symbols.

        A venue can list more than one quote-currency pair for the same base asset
        with very different depth — e.g. okx added direct SYM/USD spot markets across
        most alts on 2025-01-15 (~500 daily bars), while the long-standing SYM/USDT
        pair reaches back to ~2020 (~2300+ bars). Live-probed 2026-07-11: taking
        whichever candidate answers FIRST (as this used to) silently locked onto the
        shallow SYM/USD listing for nearly every symbol once bybit (this venue's
        deep-history primary) became geo-blocked and okx carried the whole pull —
        basis fell from 33,567 to 11,347 rows with no warning, because each symbol
        still "succeeded", just on a few hundred days of history. Try every
        candidate and keep the one whose first bar reaches furthest back; stop early
        once a candidate is already deep enough (mirrors CcxtPricesLoader's
        depth-slack early-exit) so a genuinely deep first hit doesn't pay for a
        second full pagination for nothing.
        """
        from production.data.base import fetch_ohlcv_paginated

        best: list | None = None
        for s in symbols:
            try:
                # One fetch_ohlcv call returns a single venue page (~500-1000 bars,
                # anchored at the listing) — which surfaced as basis history frozen at
                # 2021-07..2022-05 on the first live ingest. Stitch the full window.
                bars = fetch_ohlcv_paginated(ex, s, since)
            except Exception:
                continue  # not listed on this venue
            if bars and (best is None or bars[0][0] < best[0][0]):
                best = bars
            if best is not None and best[0][0] <= since + _DEPTH_SLACK_MS:
                break  # already reaches (near) the requested start — good enough
        return best

    # -------------------------------------------------------------- transform
    @staticmethod
    def _closes(bars) -> pd.DataFrame:
        """OHLCV rows -> [obs_date, close] on the midnight-UTC daily bar date."""
        arr = pd.DataFrame(bars, columns=["ts", "open", "high", "low", "close", "volume"])
        obs = pd.to_datetime(arr["ts"], unit="ms", utc=True).dt.tz_localize(None).dt.normalize()
        return pd.DataFrame({"obs_date": obs,
                             "close": pd.to_numeric(arr["close"], errors="coerce")})

    def transform(self, raw) -> pd.DataFrame:
        frames = []
        for sym, pair in raw.items():
            iid = self.resolve(f"{sym}/USD", "ccxt")
            if iid is None or not pair.get("spot") or not pair.get("swap"):
                continue
            spot = self._closes(pair["spot"]).rename(columns={"close": "spot_close"})
            swap = self._closes(pair["swap"]).rename(columns={"close": "swap_close"})
            merged = spot.merge(swap, on="obs_date", how="inner").dropna()
            if merged.empty:
                continue
            f = pd.DataFrame({
                "obs_date": merged["obs_date"],
                "instrument_id": iid,
                "basis": merged["swap_close"] / merged["spot_close"] - 1.0,
            })
            f["asset_class"] = asset_class_from_instrument_id(iid)
            frames.append(f)
        if not frames:
            return pd.DataFrame(columns=["obs_date", "instrument_id", "basis", "asset_class"])
        return pd.concat(frames, ignore_index=True).dropna(subset=["basis"]).reset_index(drop=True)
