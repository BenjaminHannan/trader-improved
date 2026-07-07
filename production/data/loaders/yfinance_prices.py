"""Primary daily equity/ETF prices from Yahoo Finance via yfinance.

Availability: Yahoo's adjusted daily bar for session D is settled after the US close;
we stamp `available_from = obs_date 21:30 UTC` (~16:30 ET), a conservative post-close
time by which the consolidated close is known. auto_adjust=True so `close` is already
total-return adjusted (splits + dividends), which is what every price-based signal
wants.
"""
from __future__ import annotations

import pandas as pd

from production.data.base import (AvailabilityRule, BaseLoader, IngestError,
                                  prices_to_long)
from production.reference.hygiene import apply_hygiene

# Known OHLCV field labels (lower-cased) used to disambiguate which level of a
# yfinance MultiIndex is the *field* level vs the *ticker* level. yfinance has
# shipped both column orderings across versions — (ticker, field) with
# group_by="ticker" and (field, ticker) with the "column" default — so we detect
# the field level by content rather than trusting a fixed position.
_OHLCV_FIELDS = {"open", "high", "low", "close", "adj close", "adjclose", "volume"}


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
        # yfinance's payload shape drifts across versions (group_by default flipped,
        # single-ticker frames go flat, field names change case, 'Adj Close' vanishes
        # under auto_adjust, and 1.x stamps a 'Price'/'Ticker' name on the column
        # levels). Normalise whatever we got into {ticker: field-frame} first, then
        # reuse the canonical long-form builder. Only genuinely unparseable payloads
        # raise — and the message carries the columns so diagnosis is one glance.
        by_symbol = self._normalize(raw)
        if not by_symbol:
            if self._is_empty_payload(raw):
                # A legitimately empty pull (nothing downloaded) is a coverage/min-rows
                # problem for the audit to flag, not a shape error.
                return prices_to_long({}, lambda s: self.resolve(s, "yfinance"))
            cols = self._columns_repr(raw)
            raise IngestError(f"unrecognized OHLCV payload shape; columns={cols}")
        long = prices_to_long(by_symbol, lambda s: self.resolve(s, "yfinance"))
        return self._apply_price_hygiene(long)

    # ------------------------------------------------------------- normalisation
    @staticmethod
    def _is_empty_payload(raw) -> bool:
        if isinstance(raw, pd.DataFrame):
            return raw.empty or len(raw.columns) == 0
        if isinstance(raw, dict):
            return len(raw) == 0
        return raw is None

    @staticmethod
    def _columns_repr(raw) -> str:
        """repr of the first 10 columns of the payload for a failure message."""
        try:
            cols = list(raw.columns) if isinstance(raw, pd.DataFrame) else list(raw)
        except Exception:
            cols = []
        return repr(cols[:10])

    @classmethod
    def _field_level(cls, columns: pd.MultiIndex) -> int:
        """Return which MultiIndex level (0 or 1) holds the OHLCV field names.

        Score each level by how many of its distinct values are recognised OHLCV
        fields; the higher-scoring level is the field level. Ties fall to level 1,
        which is yfinance's historical group_by='ticker' layout (ticker, field)."""
        def score(level: int) -> int:
            vals = {str(v).strip().lower() for v in columns.get_level_values(level)}
            return len(vals & _OHLCV_FIELDS)
        return 0 if score(0) > score(1) else 1

    def _normalize(self, raw) -> dict[str, pd.DataFrame]:
        """Coerce any yfinance payload into {ticker: OHLCV frame with field columns}.

        Per-symbol empty/all-NaN sub-frames (e.g. a ticker that a geo-blocked or
        rate-limited batch failed to download) degrade to a warning and are skipped,
        never raised."""
        if isinstance(raw, dict):  # already {symbol: frame} — defensive passthrough
            return {str(k): v for k, v in raw.items()
                    if isinstance(v, pd.DataFrame) and not self._all_nan(v, str(k))}
        if not isinstance(raw, pd.DataFrame) or raw.empty or len(raw.columns) == 0:
            return {}

        out: dict[str, pd.DataFrame] = {}
        if isinstance(raw.columns, pd.MultiIndex) and raw.columns.nlevels >= 2:
            field_level = self._field_level(raw.columns)
            ticker_level = 1 - field_level
            for tk in pd.unique(raw.columns.get_level_values(ticker_level)):
                sub = raw.xs(tk, axis=1, level=ticker_level)
                if isinstance(sub.columns, pd.MultiIndex):
                    sub.columns = sub.columns.get_level_values(-1)
                if not self._all_nan(sub, str(tk)):
                    out[str(tk)] = sub
            return out

        # Flat columns => a single-ticker download. Attribute to the lone requested
        # symbol; if the request was ambiguous we cannot map the bars, so warn+skip.
        sym = self.symbols[0] if len(self.symbols) == 1 else None
        if sym is None:
            self.warnings.append(
                "yfinance: flat single-ticker frame but the request spanned "
                f"{len(self.symbols)} symbols — cannot attribute bars, skipping")
            return {}
        if not self._all_nan(raw, str(sym)):
            out[str(sym)] = raw
        return out

    def _all_nan(self, sub: pd.DataFrame, symbol: str) -> bool:
        """True (with a warning) if a per-symbol frame carries no usable close data."""
        cols = {str(c).strip().lower(): c for c in sub.columns}
        close_col = cols.get("close") or cols.get("adj close") or cols.get("adjclose")
        empty = sub.empty or close_col is None or \
            pd.to_numeric(sub[close_col], errors="coerce").notna().sum() == 0
        if empty:
            self.warnings.append(f"yfinance: no usable close data for {symbol!r} "
                                 "(empty/geo-blocked download) — skipped")
        return empty

    def _apply_price_hygiene(self, long: pd.DataFrame) -> pd.DataFrame:
        """Enforce the sub-$0.10 backstop + reused-ticker blocklist on the canonical
        long frame before it is returned for stamping/audit. The vendor symbol is
        recovered from the synthetic instrument_id (``class:symbol:first-listing``);
        drop counts are stashed on ``self.hygiene_drops`` and, when non-zero, appended
        to ``self.warnings`` so they land in the ingest audit record — never silent."""
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
