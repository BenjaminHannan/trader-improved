"""Kalshi resolved-market historical backfill (curated dataset ``event_markets_hist``).

One-time (plus incremental top-up) backfill of SETTLED Kalshi markets for a fixed set
of US macro-release series, feeding the two pre-registered research diagnostics from
the 2026-07 practitioner mechanism scan (longshot-fade calibration; nowcast drift).

Endpoint archaeology (probed live 2026-07-10, no key needed for any of these):

  * archived settled markets (full records: result, settlement_ts, strikes,
    expiration_value = the actual print), cursor-paginated::

        GET https://external-api.kalshi.com/trade-api/v2/historical/markets
            params: series_ticker, limit (<=1000), cursor

    Only the CURRENT (KX-prefixed) series name works — the 2025 ticker migration
    aliased legacy events (e.g. ``CPIYOY-23JUN``) under the KX series, and querying
    the legacy name returns nothing. Depth: KXCPIYOY reaches back to 2022-12.

  * archived transaction history (price, contracts, taker side), per market::

        GET https://external-api.kalshi.com/trade-api/v2/historical/trades
            params: ticker, limit (<=1000), cursor

    Candlesticks do NOT exist for archived legacy markets (404 on every host/path
    combination) — trades are the only price history, so daily bars are built here.
    This mirrors Buergi-Deng-Whelan (MPRA 126350), who used the same trade-level API.

  * the archive lags ~2-3 months; the recent tail comes from the live host::

        GET https://api.elections.kalshi.com/trade-api/v2/markets
            params: series_ticker, status=settled, limit, cursor
        GET https://api.elections.kalshi.com/trade-api/v2/markets/trades
            params: ticker, limit, cursor

PIT: two row types share the schema. ``row_type='bar'`` rows are daily trade
aggregates whose ``knowable_at`` is the last trade timestamp of that UTC day —
public the moment the trade printed. ``row_type='settlement'`` rows carry the
outcome (``yes_price`` = 1.0/0.0) with ``knowable_at = settlement_ts``; the
``result``/``expiration_value`` columns live ONLY on settlement rows so a signal
joining bars can never see an outcome before it was knowable.

Dollar-string parsing: this API generation quotes prices as strings in dollars
(``"0.0300"``) and sizes as float-strings (``"68.00"``); everything is parsed with
``float`` and prices are already probabilities.
"""
from __future__ import annotations

import time

import pandas as pd

from production.data.base import AvailabilityRule, BaseLoader

HIST_BASE = "https://external-api.kalshi.com/trade-api/v2"
LIVE_BASE = "https://api.elections.kalshi.com/trade-api/v2"

# US macro-release series (current KX names; legacy events are aliased underneath).
# Chosen to cover the two diagnostics: CPI/PCE family overlaps the Cleveland Fed
# nowcast series (CLEV_NOWCAST_*), the rest are the liquid economic-data ladders.
DEFAULT_SERIES = [
    "KXCPIYOY", "KXCPI", "KXCPICORE", "KXCPICOREYOY",   # CPI family
    "KXPCECORE",                                          # core PCE
    "KXPAYROLLS", "KXUSNFP", "KXU3", "KXJOBLESS",        # labor
    "KXFED", "KXFEDDECISION",                             # Fed
    "KXGDP",                                              # GDP
]


def _f(value, default=0.0) -> float:
    """Parse the API's stringly-typed numerics (``"0.0300"`` / ``"68.00"``)."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


class KalshiHistoryLoader(BaseLoader):
    dataset = "event_markets_hist"
    vendor = "kalshi"
    source = "kalshi:historical"
    asset_classes = ["events"]
    default_asset_class = "events"
    # Every row carries its own knowable-at moment (trade print / settlement).
    availability_rule = AvailabilityRule("explicit", {"column": "knowable_at"})
    expectations = {
        "columns": ["yes_price", "volume"],
        "ranges": {"yes_price": (0.0, 1.0), "volume": (0.0, None)},
        "min_rows": 1,
        # Two-row-type schema: result/expiration_value are null on every bar row and
        # yes_vwap/taker_yes_frac on every settlement row BY DESIGN, so the generic
        # null-fraction gate would always trip. yes_price/volume/knowable_at are
        # never null by construction; the [0,1] range check stays fatal.
        "max_null_frac": 0.99,
    }
    # Settled markets never revise; a generous overlap only costs a few re-pulls.
    incremental_overlap = pd.Timedelta(days=10)

    def __init__(self, lake=None, instruments=None, series=None, page_limit=1000,
                 max_markets_per_series=5000, max_trade_pages_per_market=30,
                 pause_s=0.15):
        super().__init__(lake, instruments)
        self.series = list(series) if series is not None else list(DEFAULT_SERIES)
        self.page_limit = page_limit
        self.max_markets_per_series = max_markets_per_series
        self.max_trade_pages_per_market = max_trade_pages_per_market
        self.pause_s = pause_s

    # ------------------------------------------------------------------ fetch
    def _paginate(self, session, url: str, params: dict, key: str,
                  max_items: int) -> list[dict]:
        """Cursor-paginate one endpoint, returning up to ``max_items`` records."""
        out: list[dict] = []
        cursor = None
        while len(out) < max_items:
            time.sleep(self.pause_s)
            p = dict(params, limit=self.page_limit)
            if cursor:
                p["cursor"] = cursor
            resp = session.get(url, params=p, timeout=30)
            resp.raise_for_status()
            body = resp.json()
            page = body.get(key) or []
            out.extend(page)
            cursor = body.get("cursor")
            if not cursor or not page:
                break
        return out[:max_items]

    def fetch(self, start, end) -> dict:
        """Return ``{series: {"markets": [...], "trades": {ticker: [...]}}}``.

        Markets come from the archive plus the live settled tail (merged on ticker);
        trades prefer the archive and fall back to the live host when the archive
        has not caught up to a recently settled market. Per-market trade failures
        degrade with a warning; a whole-series listing failure also degrades so one
        renamed series never sinks the backfill.
        """
        import requests

        start = pd.Timestamp(start).tz_localize("UTC") if pd.Timestamp(start).tzinfo is None \
            else pd.Timestamp(start)
        end = pd.Timestamp(end).tz_localize("UTC") if pd.Timestamp(end).tzinfo is None \
            else pd.Timestamp(end)
        session = requests.Session()
        out: dict[str, dict] = {}
        for series in self.series:
            markets: dict[str, dict] = {}
            try:
                archived = self._paginate(
                    session, f"{HIST_BASE}/historical/markets",
                    {"series_ticker": series}, "markets", self.max_markets_per_series)
                live = self._paginate(
                    session, f"{LIVE_BASE}/markets",
                    {"series_ticker": series, "status": "settled"}, "markets",
                    self.max_markets_per_series)
            except Exception as exc:
                self.warnings.append(f"kalshi-hist listing failed for {series}: {exc!r}")
                continue
            for m in archived + live:            # live tail wins on ticker collision
                ticker = m.get("ticker")
                if not ticker:
                    continue
                close = pd.to_datetime(m.get("close_time"), utc=True, errors="coerce")
                if close is pd.NaT or close < start or close > end:
                    continue
                markets[ticker] = m

            trades: dict[str, list] = {}
            for ticker in markets:
                max_trades = self.max_trade_pages_per_market * self.page_limit
                try:
                    rows = self._paginate(
                        session, f"{HIST_BASE}/historical/trades",
                        {"ticker": ticker}, "trades", max_trades)
                    if not rows:                 # archive lag: recent tail lives here
                        rows = self._paginate(
                            session, f"{LIVE_BASE}/markets/trades",
                            {"ticker": ticker}, "trades", max_trades)
                    if len(rows) >= max_trades:
                        self.warnings.append(
                            f"kalshi-hist trade cap hit for {ticker} "
                            f"({max_trades} rows) — history truncated")
                    trades[ticker] = rows
                except Exception as exc:
                    self.warnings.append(
                        f"kalshi-hist trades failed for {ticker}: {exc!r}")
                    continue
            out[series] = {"markets": list(markets.values()), "trades": trades}
        return out

    # -------------------------------------------------------------- transform
    def transform(self, raw) -> pd.DataFrame:
        rows: list[dict] = []
        for series, payload in (raw or {}).items():
            markets = {m.get("ticker"): m for m in payload.get("markets", [])
                       if m.get("ticker")}
            trades = payload.get("trades", {})
            for ticker, meta in markets.items():
                base = {
                    "instrument_id": f"EV:kalshi:{ticker}",
                    "series_ticker": series,
                    "event_key": meta.get("event_ticker") or ticker,
                    "question": meta.get("title") or ticker,
                    "close_time": meta.get("close_time"),
                    "strike_type": meta.get("strike_type"),
                    "floor_strike": _f(meta.get("floor_strike"), default=float("nan")),
                    "cap_strike": _f(meta.get("cap_strike"), default=float("nan")),
                    "venue": "kalshi",
                }
                rows.extend(self._bar_rows(base, trades.get(ticker) or []))
                settle = self._settlement_row(base, meta)
                if settle is not None:
                    rows.append(settle)
                elif meta.get("result") not in (None, ""):
                    self.warnings.append(
                        f"kalshi-hist unmapped result {meta.get('result')!r} "
                        f"for {ticker} — settlement row skipped")
        return _object_cast(pd.DataFrame(rows))

    @staticmethod
    def _bar_rows(base: dict, trade_list: list[dict]) -> list[dict]:
        """Daily UTC bars from trade prints: last, VWAP, volume, taker-side split."""
        parsed = []
        for t in trade_list:
            ts = pd.to_datetime(t.get("created_time"), utc=True, errors="coerce")
            price = _f(t.get("yes_price_dollars"), default=float("nan"))
            if ts is pd.NaT or not (0.0 <= price <= 1.0):
                continue
            parsed.append((ts, price, _f(t.get("count_fp")),
                           t.get("taker_side") == "yes"))
        if not parsed:
            return []
        parsed.sort(key=lambda r: r[0])
        out: list[dict] = []
        frame = pd.DataFrame(parsed, columns=["ts", "price", "count", "taker_yes"])
        for day, g in frame.groupby(frame["ts"].dt.normalize()):
            vol = float(g["count"].sum())
            out.append(dict(
                base,
                row_type="bar",
                obs_date=day.tz_localize(None),
                yes_price=float(g["price"].iloc[-1]),
                yes_vwap=float((g["price"] * g["count"]).sum() / vol) if vol > 0
                else float(g["price"].mean()),
                volume=vol,
                n_trades=int(len(g)),
                taker_yes_frac=float(g.loc[g["taker_yes"], "count"].sum() / vol)
                if vol > 0 else float("nan"),
                knowable_at=g["ts"].iloc[-1].isoformat(),
                result=None,
                expiration_value=None,
            ))
        return out

    @staticmethod
    def _settlement_row(base: dict, meta: dict) -> dict | None:
        """Outcome row: yes_price is the terminal contract value, knowable at
        settlement. Non-binary or voided results yield no row."""
        result = meta.get("result")
        if result not in ("yes", "no"):
            return None
        settled = pd.to_datetime(
            meta.get("settlement_ts") or meta.get("expiration_time")
            or meta.get("close_time"), utc=True, errors="coerce")
        if settled is pd.NaT:
            return None
        return dict(
            base,
            row_type="settlement",
            obs_date=settled.tz_localize(None).normalize(),
            yes_price=1.0 if result == "yes" else 0.0,
            yes_vwap=float("nan"),
            volume=0.0,
            n_trades=0,
            taker_yes_frac=float("nan"),
            knowable_at=settled.isoformat(),
            result=result,
            expiration_value=meta.get("expiration_value"),
        )


def _object_cast(df: pd.DataFrame) -> pd.DataFrame:
    """Force text columns to numpy object dtype (see kalshi.py: the curated audit's
    numeric summary cannot introspect pandas StringDtype)."""
    for col in ("series_ticker", "event_key", "question", "close_time", "strike_type",
                "venue", "row_type", "result", "expiration_value", "knowable_at"):
        if col in df.columns:
            df[col] = df[col].astype(object)
    return df
