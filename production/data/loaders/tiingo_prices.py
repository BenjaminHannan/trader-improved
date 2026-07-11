"""Secondary daily prices from Tiingo's EOD API (replaces the dead stooq feed).

Stooq locked its CSV endpoints behind a JS browser-verification wall in 2026 (see
diagnostics/stooq_verdict.md), which orphaned the cross-check's secondary feed. Tiingo's
daily endpoint serves split+dividend-adjusted bars (``adjClose``/``adjVolume``) — the
same total-return semantics as the yfinance loader's ``auto_adjust=True``.

Availability is stamped 21:15 UTC — deliberately EARLIER than yfinance's 21:30. Both
are honest post-close times, and the 15-minute gap encodes feed priority in the
availability algebra itself: the curated lake dedups on (obs_date, instrument_id,
available_from), so both vendors' rows coexist (enabling
production/data/cross_check.py), while asof_panel's latest-visible-vintage rule makes
yfinance win whenever both are present and tiingo fill the gaps — exactly fallback
semantics, with no special-case code. (Identical contract to the retired stooq loader.)

The free tier is rate-limited (hourly/daily request caps and a unique-symbol
allowance), so ``max_symbols`` bounds a single pull and a 429 stops the fetch early
with a warning instead of burning the quota — the ingest is incremental, re-run to
extend coverage. API key from ``TIINGO_API_KEY`` (env only, never a tracked file).
"""
from __future__ import annotations

import os
import time

import pandas as pd

from production.data.base import AvailabilityRule, BaseLoader, IngestError, prices_to_long
from production.reference.hygiene import apply_hygiene

TIINGO_URL = "https://api.tiingo.com/tiingo/daily/{symbol}/prices"


class TiingoPricesLoader(BaseLoader):
    dataset = "prices"
    vendor = "tiingo"
    source = "tiingo:prices"
    asset_classes = ["equity", "fx", "commodity"]
    availability_rule = AvailabilityRule("obs_offset",
                                         {"offset": pd.Timedelta(hours=21, minutes=15)})
    expectations = {
        "columns": ["close", "volume", "dollar_volume"],
        "ranges": {"close": (0, None), "volume": (0, None)},
        "max_null_frac": 0.02,
        "min_rows": 1,
    }

    def __init__(self, lake=None, instruments=None, symbols=None, api_key=None,
                 max_symbols=None, pause_s=0.4):
        super().__init__(lake, instruments)
        # Tiingo tickers use the dash class-share form (BRK-B) — same as yfinance.
        self.symbols = symbols or []
        self.api_key = api_key if api_key is not None else os.environ.get("TIINGO_API_KEY")
        self.max_symbols = max_symbols
        self.pause_s = pause_s

    def fetch(self, start, end) -> dict[str, pd.DataFrame]:
        import requests

        if not self.api_key:
            raise IngestError(
                "tiingo: TIINGO_API_KEY is not set. Export it in the environment "
                "(never commit it) — free keys at https://www.tiingo.com/.")

        # Uncovered symbols first: the hourly free-tier cap 429s after ~50 requests,
        # and a FIXED list order would re-fetch the same head every run — coverage
        # would never extend (found 2026-07-11: two successive runs fetched the same
        # ~54 names). Deprioritizing symbols that already have tiingo rows in the
        # lake makes each quota window push the frontier instead.
        ordered = self._frontier_order(self.symbols)
        todo = ordered[: self.max_symbols] if self.max_symbols else ordered
        if self.max_symbols and len(self.symbols) > self.max_symbols:
            self.warnings.append(
                f"tiingo: symbol list truncated to max_symbols={self.max_symbols} "
                f"of {len(self.symbols)} (free-tier quota); re-run to extend coverage")
        out: dict[str, pd.DataFrame] = {}
        params = {"startDate": str(pd.Timestamp(start).date()),
                  "endDate": str(pd.Timestamp(end).date()),
                  "token": self.api_key, "format": "json"}
        for sym in todo:
            time.sleep(self.pause_s)
            try:
                resp = requests.get(TIINGO_URL.format(symbol=sym), params=params,
                                    timeout=30)
                if resp.status_code == 429:
                    # Quota exhausted: keep what we have, never burn the key further.
                    self.warnings.append(
                        f"tiingo: rate-limited (429) at symbol {sym!r} — stopping "
                        f"with {len(out)} of {len(todo)} symbols fetched")
                    break
                if resp.status_code == 404:   # unknown/delisted ticker on tiingo
                    continue
                resp.raise_for_status()
                rows = resp.json()
            except Exception as exc:
                self.warnings.append(f"tiingo: fetch failed for {sym!r}: {exc}")
                continue
            df = self._to_ohlcv(rows)
            if df is not None and not df.empty:
                out[sym] = df
        return out

    def _frontier_order(self, symbols: list[str]) -> list[str]:
        """Stable-sort ``symbols`` so those WITHOUT existing tiingo-sourced lake rows
        come first (see fetch: the hourly quota then extends coverage every run).
        Stability preserves the ETF-sleeves-before-equities intent within each group.
        Any lake/resolution hiccup degrades to the original order — never raises."""
        try:
            cur = self.lake.read_curated("prices")
            covered_iids = set(
                cur.loc[cur["source"].astype(str).str.contains("tiingo"),
                        "instrument_id"].unique())
        except Exception:
            return list(symbols)
        if not covered_iids:
            return list(symbols)

        def _is_covered(sym: str) -> bool:
            try:
                return self.resolve(sym, "yfinance") in covered_iids
            except Exception:
                return False

        return sorted(symbols, key=_is_covered)

    @staticmethod
    def _to_ohlcv(rows) -> pd.DataFrame | None:
        """Tiingo JSON rows -> the Date-indexed OHLCV frame prices_to_long expects,
        using the adjusted fields (total-return semantics, like yfinance auto_adjust)."""
        if not rows or not isinstance(rows, list):
            return None
        df = pd.DataFrame(rows)
        if "date" not in df.columns or "adjClose" not in df.columns:
            return None
        out = pd.DataFrame({
            "Date": pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize(),
            "Close": pd.to_numeric(df["adjClose"], errors="coerce"),
            "Volume": pd.to_numeric(df.get("adjVolume", df.get("volume")),
                                    errors="coerce"),
        })
        return out.set_index("Date").sort_index()

    def transform(self, raw) -> pd.DataFrame:
        # Tiingo symbols are the yfinance dash form, so resolve through that vendor key.
        long = prices_to_long(raw, lambda s: self.resolve(s, "yfinance"))
        return self._apply_price_hygiene(long)

    def _apply_price_hygiene(self, long: pd.DataFrame) -> pd.DataFrame:
        """Same backstop + blocklist enforcement as the other price loaders."""
        if long is not None and not long.empty and "symbol" not in long.columns \
                and "instrument_id" in long.columns:
            long = long.copy()
            long["symbol"] = long["instrument_id"].map(
                lambda i: str(i).split(":")[1] if pd.notna(i) and ":" in str(i) else i)
        clean, drops = apply_hygiene(long, symbol_col="symbol")
        clean = clean.drop(columns=["symbol"], errors="ignore")
        self.hygiene_drops = drops
        if any(drops.values()):
            self.warnings.append(
                f"hygiene: dropped {drops['backstop_dropped']} sub-floor price row(s), "
                f"{drops.get('flap_dropped', 0)} flap-outlier row(s), "
                f"and {drops['blocklist_dropped']} blocklisted-ticker row(s)")
        return clean
