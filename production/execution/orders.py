"""Translate a target book (fraction-of-equity weights) into concrete broker orders.

The engine produces *target weights* — the fraction of book equity each instrument
should carry. To trade them we diff against what we currently hold, size the delta in
dollars, and turn dollars into share/coin quantities the broker will accept:

  delta_notional_i = (w_target_i - w_current_i) * equity_usd
  qty_i            = |delta_notional_i| / price_i   (rounded per sleeve)

Rounding is per sleeve because brokers quantize differently: US equities trade in whole
shares for v1 (no fractional), crypto in fractional coins (6 dp is Alpaca's granularity).
Tiny dollar deltas are dropped (``min_order_usd``) — they cost more in fees/slippage than
the tracking-error they close. Everything here is a pure function of its arguments (no IO,
no clock); the point-in-time reasoning lives upstream in the engine.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from production.signals.base import sleeve_from_id

ORDER_COLUMNS = ["instrument_id", "side", "qty", "notional_usd", "order_type"]

# Per-sleeve quantity granularity. Equities are whole shares (fractional disabled for v1);
# crypto is fractional to 6 decimal places (Alpaca's smallest crypto increment).
_EQUITY_SLEEVES = {"equity", "fx_etf", "commodity_etf"}
_CRYPTO_DP = 6
# Limit-price rounding: equities/ETFs quote in cents (2dp); crypto to 6 significant figures.
_EQUITY_PRICE_DP = 2
_CRYPTO_PRICE_SIGFIGS = 6


def _round_qty(qty: float, sleeve: str) -> float:
    """Round a raw quantity to the sleeve's tradable granularity."""
    if sleeve == "crypto":
        return float(np.round(qty, _CRYPTO_DP))
    # Equities / ETFs: whole shares only.
    return float(np.round(qty))


def _round_limit_price(price: float, sleeve: str) -> float:
    """Round a limit price to the sleeve's tick convention.

    Equities/ETFs quote in cents (2 decimal places); crypto to 6 significant figures
    (Alpaca accepts fine crypto price granularity but rejects excessive precision).
    """
    if sleeve == "crypto":
        if not np.isfinite(price) or price == 0.0:
            return float(price)
        digits = _CRYPTO_PRICE_SIGFIGS - int(math.floor(math.log10(abs(price)))) - 1
        return float(round(price, digits))
    return float(round(price, _EQUITY_PRICE_DP))


def target_weights_to_orders(w_target: pd.Series, w_current: pd.Series,
                             equity_usd: float, prices: pd.Series,
                             min_order_usd: float = 25.0,
                             order_type: str = "market",
                             limit_offset_bps: float = 5.0) -> pd.DataFrame:
    """Diff a target book against current holdings and emit orders.

    Parameters
    ----------
    w_target      : Series instrument_id -> target weight (fraction of ``equity_usd``).
    w_current     : Series instrument_id -> current weight (same units); missing == 0.
    equity_usd    : total account equity the weights are fractions of.
    prices        : Series instrument_id -> latest price (same currency as equity).
    min_order_usd : orders whose rounded notional falls below this are dropped.
    order_type    : ``"market"`` (default) or ``"limit"``.
    limit_offset_bps : half-spread offset for passive limit placement (default 5bp).

    Returns a DataFrame with columns ``[instrument_id, side, qty, notional_usd,
    order_type]``; ``side`` is ``"buy"``/``"sell"``, ``qty`` and ``notional_usd`` are
    positive magnitudes (direction lives in ``side``). Ordered by descending notional for
    readable eyeballing.

    When ``order_type="limit"`` each row gains a ``limit_price`` column: the decision
    price nudged *passively* by ``limit_offset_bps`` — buys priced *below* the mark, sells
    *above* it, so we sit inside/at the touch to CAPTURE spread rather than crossing it.
    Prices round to the sleeve tick (equities/ETFs 2dp; crypto 6 significant figures).
    The ``"market"`` path is bit-identical to before (no ``limit_price`` column).
    """
    w_target = pd.Series(w_target, dtype=float)
    w_current = pd.Series(w_current, dtype=float)
    prices = pd.Series(prices, dtype=float)

    if order_type not in ("market", "limit"):
        raise ValueError(f"order_type must be 'market' or 'limit', got {order_type!r}")
    is_limit = order_type == "limit"
    offset = float(limit_offset_bps) / 1e4

    ids = sorted(set(w_target.index) | set(w_current.index))
    rows: list[dict] = []
    for iid in ids:
        px = float(prices.get(iid, np.nan))
        if not np.isfinite(px) or px <= 0.0:
            # No usable price -> cannot size the order; skip (a live runner should surface
            # this, but the order layer stays a pure sizing function).
            continue
        wt = float(w_target.get(iid, 0.0))
        wc = float(w_current.get(iid, 0.0))
        delta_usd = (wt - wc) * float(equity_usd)
        if delta_usd == 0.0:
            continue
        side = "buy" if delta_usd > 0 else "sell"
        sleeve = sleeve_from_id(iid)
        qty = _round_qty(abs(delta_usd) / px, sleeve)
        if qty <= 0.0:
            continue
        notional = qty * px
        if notional < float(min_order_usd):
            continue
        row = {"instrument_id": iid, "side": side, "qty": qty,
               "notional_usd": notional, "order_type": order_type}
        if is_limit:
            # Passive placement: buy below the mark, sell above it (never cross aggressively).
            raw_limit = px * (1.0 - offset) if side == "buy" else px * (1.0 + offset)
            row["limit_price"] = _round_limit_price(raw_limit, sleeve)
        rows.append(row)

    columns = ORDER_COLUMNS + (["limit_price"] if is_limit else [])
    out = pd.DataFrame(rows, columns=columns)
    if not out.empty:
        out = out.sort_values("notional_usd", ascending=False).reset_index(drop=True)
    return out
