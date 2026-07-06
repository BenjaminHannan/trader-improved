"""Minimal Alpaca *paper*-trading client — raw ``requests``, no ``alpaca-py`` dependency.

Deliberately thin: the only broker surface the daily runner needs is read account /
read positions / submit order. Keeping it to raw HTTP means one dependency fewer and a
transport we can fully monkeypatch in tests (no network ever runs under pytest).

Safety posture, in order of importance:
  * Credentials come from ``APCA_API_KEY_ID`` / ``APCA_API_SECRET_KEY`` env vars (or
    explicit args) — never hardcoded, never logged.
  * ``dry_run`` is the default on order submission and NEVER issues an HTTP request; it
    returns the payloads that *would* have been sent. Going live is an explicit opt-out.
  * Base URL defaults to the paper endpoint. Pointing at the live endpoint is a caller's
    deliberate act.
  * Transient failures (429 rate-limit, 5xx) retry with backoff; a 4xx (bad request /
    auth / not found) is a caller error and raises immediately.
"""
from __future__ import annotations

import os
import time

import pandas as pd
import requests

from production.reference.instruments import vendor_symbol

_PAPER_BASE_URL = "https://paper-api.alpaca.markets"
_RETRY_STATUS = {429, 500, 502, 503, 504}


class ExecutionError(Exception):
    """Raised on a non-retryable broker error (4xx) or exhausted retries."""


class AlpacaPaperClient:
    """Raw-HTTP Alpaca paper client covering account / positions / order submission."""

    def __init__(self, key_id: str | None = None, secret: str | None = None,
                 base_url: str = _PAPER_BASE_URL, *, max_retries: int = 3,
                 backoff: float = 0.5, timeout: float = 15.0) -> None:
        self.key_id = key_id if key_id is not None else os.environ.get("APCA_API_KEY_ID")
        self.secret = secret if secret is not None else os.environ.get("APCA_API_SECRET_KEY")
        self.base_url = base_url.rstrip("/")
        self.max_retries = int(max_retries)
        self.backoff = float(backoff)
        self.timeout = float(timeout)

    # -------------------------------------------------------------- transport
    def _headers(self) -> dict:
        if not self.key_id or not self.secret:
            raise ExecutionError(
                "missing Alpaca credentials — set APCA_API_KEY_ID / APCA_API_SECRET_KEY "
                "or pass key_id/secret (never required for --dry-run)")
        return {
            "APCA-API-KEY-ID": self.key_id,
            "APCA-API-SECRET-KEY": self.secret,
        }

    def _request(self, method: str, path: str, *, params: dict | None = None,
                 json: dict | None = None):
        """Issue a request with retry/backoff on 429 & 5xx; raise ExecutionError on 4xx."""
        url = f"{self.base_url}{path}"
        headers = self._headers()
        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            resp = requests.request(method, url, headers=headers, params=params,
                                    json=json, timeout=self.timeout)
            status = resp.status_code
            if status < 400:
                if resp.content:
                    return resp.json()
                return None
            if status in _RETRY_STATUS and attempt < self.max_retries - 1:
                last_exc = ExecutionError(f"{method} {path} -> {status} (retryable)")
                time.sleep(self.backoff * (2 ** attempt))
                continue
            # Non-retryable 4xx, or retries exhausted.
            text = _safe_text(resp)
            raise ExecutionError(f"{method} {path} -> {status}: {text}")
        raise ExecutionError(f"{method} {path} failed after {self.max_retries} retries "
                             f"({last_exc})")

    # ------------------------------------------------------------------- reads
    def account(self) -> dict:
        """Return the account object (equity, buying power, status, ...)."""
        return self._request("GET", "/v2/account")

    def positions(self) -> pd.DataFrame:
        """Current open positions as a DataFrame [symbol, qty, market_value, side].

        Empty (0-row) frame when the account is flat. Broker symbols, not internal ids —
        the daily runner maps back through the instrument master when diffing.
        """
        raw = self._request("GET", "/v2/positions") or []
        cols = ["symbol", "qty", "market_value", "side"]
        if not raw:
            return pd.DataFrame(columns=cols)
        rows = [{"symbol": p.get("symbol"), "qty": float(p.get("qty", 0.0)),
                 "market_value": float(p.get("market_value", 0.0)),
                 "side": p.get("side")} for p in raw]
        return pd.DataFrame(rows, columns=cols)

    # ------------------------------------------------------------- order write
    def submit_order(self, symbol: str, qty: float, side: str, type_: str = "market",
                     tif: str = "day") -> dict:
        """POST a single order (always issues a request — callers gate with dry_run)."""
        payload = {"symbol": symbol, "qty": str(qty), "side": side,
                   "type": type_, "time_in_force": tif}
        return self._request("POST", "/v2/orders", json=payload)

    def submit_orders(self, orders: pd.DataFrame, master: pd.DataFrame,
                      dry_run: bool = True) -> pd.DataFrame:
        """Map internal-id orders to Alpaca symbols and (unless dry_run) submit them.

        ``orders`` is the frame from :func:`production.execution.orders.target_weights_to_orders`.
        Instrument ids are resolved to broker symbols via the master's ``vendor_symbols``
        JSON (``"alpaca"`` key; crypto is ``BTCUSD`` style). When ``dry_run`` (the default)
        NO HTTP request is issued — the returned frame is the plan that *would* have been
        sent. Unmapped instruments are surfaced with ``status="no_alpaca_symbol"`` and
        never submitted.
        """
        cols = ["instrument_id", "alpaca_symbol", "side", "qty", "order_type",
                "status", "broker_order_id"]
        if orders is None or orders.empty:
            return pd.DataFrame(columns=cols)

        rows: list[dict] = []
        for o in orders.itertuples(index=False):
            iid = o.instrument_id
            symbol = vendor_symbol(master, iid, "alpaca")
            base = {"instrument_id": iid, "alpaca_symbol": symbol, "side": o.side,
                    "qty": float(o.qty), "order_type": o.order_type,
                    "broker_order_id": None}
            if symbol is None:
                rows.append({**base, "status": "no_alpaca_symbol"})
                continue
            if dry_run:
                rows.append({**base, "status": "dry_run"})
                continue
            resp = self.submit_order(symbol, o.qty, o.side, type_=o.order_type)
            rows.append({**base, "status": (resp or {}).get("status", "submitted"),
                         "broker_order_id": (resp or {}).get("id")})
        return pd.DataFrame(rows, columns=cols)


def _safe_text(resp) -> str:
    try:
        return resp.text[:500]
    except Exception:  # noqa: BLE001 - error reporting must never itself raise
        return "<no body>"
