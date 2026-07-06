"""Polymarket prediction-market loader (curated dataset ``event_markets``).

Polymarket is a crypto-settled binary event exchange. Two public endpoints, no key:

  * Gamma metadata (paginated by ``offset``/``limit``):
        GET https://gamma-api.polymarket.com/markets
        params: ``limit``, ``offset``, ``closed`` (bool), ``order``, ``ascending``
        -> [ {id, question, conditionId, endDate, closed, volume, liquidity,
              clobTokenIds: "[\"<yesToken>\", \"<noToken>\"]", outcomes, ...}, ... ]
  * CLOB price history for one outcome token:
        GET https://clob.polymarket.com/prices-history
        params: ``market`` (= clobTokenId of the YES outcome), ``interval`` (e.g. "max"),
                ``fidelity`` (minutes; 1440 = daily)
        -> {"history": [ {"t": <epoch_s>, "p": <prob 0..1>}, ... ]}

Polymarket already quotes the YES outcome as a probability in [0,1], so no cent
normalization is needed (the audit range check still guards it). Dollar ``volume`` and
``liquidity`` come from the Gamma market record (USD). Like Kalshi this is snapshot
data, so ``available_from = ingested_at``, and each market degrades independently.
``conditionId`` (the on-chain condition) is the shared-event key that groups the YES
and any sibling markets of one resolution — the correlated-resolution point.
"""
from __future__ import annotations

import json

import pandas as pd

from production.data.base import AvailabilityRule, BaseLoader

GAMMA_BASE = "https://gamma-api.polymarket.com"
CLOB_BASE = "https://clob.polymarket.com"


class PolymarketLoader(BaseLoader):
    dataset = "event_markets"
    vendor = "polymarket"
    source = "polymarket:gamma"
    asset_classes = ["events"]
    default_asset_class = "events"
    availability_rule = AvailabilityRule("ingest_time")
    expectations = {
        "columns": ["yes_price", "volume", "open_interest"],
        "ranges": {"yes_price": (0.0, 1.0), "volume": (0.0, None),
                   "open_interest": (0.0, None)},
        "min_rows": 1,
    }

    def __init__(self, lake=None, instruments=None, closed=False, max_markets=500,
                 page_limit=100):
        super().__init__(lake, instruments)
        self.closed = closed
        self.max_markets = max_markets
        self.page_limit = page_limit

    # ------------------------------------------------------------------ fetch
    def fetch(self, start, end) -> dict:
        """Return ``{"markets": [...], "history": {market_id: [{t,p}, ...]}}``.

        Gamma markets are paged via ``offset``; each market's YES-token price history is
        pulled from the CLOB endpoint. ``start`` bounds the history via ``startTs``.
        Per-market failures degrade with a warning.
        """
        import requests

        start_ts = int(pd.Timestamp(start).timestamp())

        markets: list[dict] = []
        offset = 0
        while len(markets) < self.max_markets:
            resp = requests.get(
                f"{GAMMA_BASE}/markets",
                params={"limit": self.page_limit, "offset": offset,
                        "closed": str(self.closed).lower(), "order": "volume",
                        "ascending": "false"},
                timeout=30)
            resp.raise_for_status()
            page = resp.json()
            if not page:
                break
            markets.extend(page)
            offset += self.page_limit
        markets = markets[: self.max_markets]

        history: dict[str, list] = {}
        for m in markets:
            market_id = _market_id(m)
            yes_token = _yes_token(m)
            if not market_id or not yes_token:
                continue
            try:
                hresp = requests.get(
                    f"{CLOB_BASE}/prices-history",
                    params={"market": yes_token, "startTs": start_ts,
                            "fidelity": 1440},
                    timeout=30)
                hresp.raise_for_status()
                history[market_id] = hresp.json().get("history", [])
            except Exception as exc:
                self.warnings.append(
                    f"polymarket prices-history failed for {market_id}: {exc!r}")
                continue
        return {"markets": markets, "history": history}

    # -------------------------------------------------------------- transform
    def transform(self, raw) -> pd.DataFrame:
        markets = {}
        for m in raw.get("markets", []):
            mid = _market_id(m)
            if mid:
                markets[mid] = m
        history = raw.get("history", {})
        rows: list[dict] = []
        for market_id, points in history.items():
            meta = markets.get(market_id, {})
            # ISO string, not a tz-aware Timestamp (see KalshiLoader.transform note).
            close_time = meta.get("endDate")
            # closed flag -> a coarse status; Polymarket has no per-candle status.
            status = "closed" if meta.get("closed") else "active"
            # conditionId groups sibling markets of one on-chain resolution.
            event_key = meta.get("conditionId") or market_id
            question = meta.get("question") or market_id
            volume = _num(meta.get("volume"))
            liquidity = _num(meta.get("liquidity"))  # oi proxy: open on-book depth (USD)
            instrument_id = f"EV:polymarket:{market_id}"
            for pt in points or []:
                try:
                    ts = pt.get("t")
                    price = pt.get("p")
                    if ts is None or price is None:
                        continue
                    obs_date = pd.Timestamp(int(ts), unit="s", tz="UTC").tz_localize(None).normalize()
                    rows.append({
                        "obs_date": obs_date,
                        "instrument_id": instrument_id,
                        "yes_price": float(price),
                        "volume": volume,
                        "open_interest": liquidity,
                        "close_time": close_time,
                        "status": status,
                        "event_key": event_key,
                        "question": question,
                        "venue": "polymarket",
                    })
                except (TypeError, ValueError) as exc:
                    self.warnings.append(
                        f"polymarket point parse failed for {market_id}: {exc!r}")
                    continue
        df = pd.DataFrame(rows)
        # pandas 3.x infers str columns as StringDtype, which the audit's numeric summary
        # cannot introspect; force numpy object dtype (parquet-round-trippable).
        for col in ("close_time", "status", "event_key", "question", "venue"):
            if col in df.columns:
                df[col] = df[col].astype(object)
        return df


# ---------------------------------------------------------------------- helpers
def _market_id(m: dict) -> str | None:
    mid = m.get("id") or m.get("conditionId")
    return str(mid) if mid is not None else None


def _yes_token(m: dict) -> str | None:
    """First CLOB token id = the YES outcome. Gamma serializes it as a JSON string."""
    raw = m.get("clobTokenIds")
    if raw is None:
        return None
    try:
        tokens = json.loads(raw) if isinstance(raw, str) else list(raw)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    return str(tokens[0]) if tokens else None


def _num(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
