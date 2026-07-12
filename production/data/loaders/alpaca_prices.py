"""Secondary daily prices from Alpaca Market Data (IEX feed).

Second leg of the cross-check trio. Free with the owner's Alpaca keys and — unlike
Tiingo's free tier — carries no unique-symbol quota, so it can cover the full universe
in one pull. Empirical constraint (probed 2026-07-07): IEX daily-bar history on the
free plan reaches back to ~2021, so the cross-check overlap window is ~5 years; Tiingo
supplies pre-2021 depth for the symbols its quota admits.

``adjustment=all`` returns split+dividend-adjusted bars — the same total-return
semantics as the yfinance loader's ``auto_adjust=True``.

Availability is stamped 21:10 UTC — earlier than tiingo's 21:15 and yfinance's 21:30,
all honest post-close times. The ordering encodes feed priority in the availability
algebra itself (curated dedup keeps all vendors' rows; asof_panel's latest-visible-
vintage rule makes yfinance > tiingo > alpaca wherever they overlap) with no
special-case code.

Credentials from ``APCA_API_KEY_ID`` / ``APCA_API_SECRET_KEY`` (env only, never a
tracked file).
"""
from __future__ import annotations

import os
import time

import pandas as pd

from production.data.base import AvailabilityRule, BaseLoader, IngestError, prices_to_long
from production.reference.hygiene import apply_hygiene

BARS_URL = "https://data.alpaca.markets/v2/stocks/bars"


class AlpacaPricesLoader(BaseLoader):
    dataset = "prices"
    vendor = "alpaca"
    source = "alpaca:prices"
    asset_classes = ["equity", "fx", "commodity"]
    availability_rule = AvailabilityRule("obs_offset",
                                         {"offset": pd.Timedelta(hours=21, minutes=10)})
    expectations = {
        "columns": ["close", "volume", "dollar_volume"],
        "ranges": {"close": (0, None), "volume": (0, None)},
        "max_null_frac": 0.02,
        "min_rows": 1,
    }

    def __init__(self, lake=None, instruments=None, symbols=None, key_id=None,
                 secret=None, batch_size=100, pause_s=0.35, feed="iex"):
        super().__init__(lake, instruments)
        # Alpaca tickers are the plain exchange form; class shares use dots (BRK.B).
        self.symbols = symbols or []
        self.key_id = key_id if key_id is not None else os.environ.get("APCA_API_KEY_ID")
        self.secret = secret if secret is not None else os.environ.get("APCA_API_SECRET_KEY")
        self.batch_size = batch_size
        self.pause_s = pause_s
        self.feed = feed

    def fetch(self, start, end) -> dict[str, pd.DataFrame]:
        import requests

        if not (self.key_id and self.secret):
            raise IngestError(
                "alpaca: APCA_API_KEY_ID / APCA_API_SECRET_KEY are not set. Export "
                "them in the environment (never commit them).")

        headers = {"APCA-API-KEY-ID": self.key_id, "APCA-API-SECRET-KEY": self.secret}
        base_params = {
            "timeframe": "1Day", "adjustment": "all", "feed": self.feed,
            "limit": 10000,
            "start": pd.Timestamp(start).strftime("%Y-%m-%dT00:00:00Z"),
            "end": pd.Timestamp(end).strftime("%Y-%m-%dT23:59:59Z"),
        }
        bars_by_symbol: dict[str, list] = {}
        for i in range(0, len(self.symbols), self.batch_size):
            batch = self.symbols[i:i + self.batch_size]
            params = dict(base_params, symbols=",".join(batch))
            page_token = None
            while True:
                time.sleep(self.pause_s)   # free tier: 200 req/min — stay far under
                if page_token:
                    params["page_token"] = page_token
                try:
                    resp = requests.get(BARS_URL, params=params, headers=headers,
                                        timeout=60)
                    resp.raise_for_status()
                    payload = resp.json()
                except Exception as exc:
                    self.warnings.append(
                        f"alpaca: batch {i // self.batch_size} failed: {exc}")
                    break
                for sym, rows in (payload.get("bars") or {}).items():
                    bars_by_symbol.setdefault(sym, []).extend(rows or [])
                page_token = payload.get("next_page_token")
                if not page_token:
                    break
        return {sym: df for sym, rows in bars_by_symbol.items()
                if (df := self._to_ohlcv(rows)) is not None and not df.empty}

    @staticmethod
    def _to_ohlcv(rows) -> pd.DataFrame | None:
        """Alpaca bar dicts ({t,o,h,l,c,v,...}) -> Date-indexed OHLCV frame."""
        if not rows:
            return None
        df = pd.DataFrame(rows)
        if "t" not in df.columns or "c" not in df.columns:
            return None
        out = pd.DataFrame({
            "Date": pd.to_datetime(df["t"]).dt.tz_localize(None).dt.normalize(),
            "Close": pd.to_numeric(df["c"], errors="coerce"),
            "Volume": pd.to_numeric(df["v"], errors="coerce"),
        })
        return out.set_index("Date").sort_index()

    def transform(self, raw) -> pd.DataFrame:
        # Alpaca uses the dotted class-share form (BRK.B) = the master's plain symbol;
        # resolve via the explicit alpaca vendor key first, then the plain symbol.
        long = prices_to_long(
            raw, lambda s: self.resolve(s, "alpaca") or self.resolve(s))
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
