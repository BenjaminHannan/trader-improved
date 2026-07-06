"""Implementation shortfall — the gap between the price we decided at and the price we got.

For each order we compare the *decision* price (the close/mark the target book was built
from) against the realized *fill* price, signed by trade direction so that a cost is always
positive:

  shortfall_bps = side_sign * (fill - decision) / decision * 1e4
    buy  (side_sign = +1): paying *above* the decision price is a positive cost;
    sell (side_sign = -1): selling *below* the decision price is a positive cost.

The book-level number is the notional-weighted mean across orders — big trades dominate
the tracking of execution quality, exactly as they dominate P&L. In paper trading this
measures the paper fill against our decision mark; the same arithmetic carries to live.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SHORTFALL_COLUMNS = ["instrument_id", "side", "decision_price", "fill_price",
                     "notional_usd", "shortfall_bps"]

_SIDE_SIGN = {"buy": 1.0, "sell": -1.0}


def implementation_shortfall(orders: pd.DataFrame, decision_prices: pd.Series,
                             fill_prices: pd.Series) -> tuple[pd.DataFrame, float]:
    """Per-order and aggregate implementation shortfall in basis points.

    Parameters
    ----------
    orders          : frame with at least ``instrument_id``, ``side`` and either
                      ``notional_usd`` (used as the aggregation weight) or ``qty``.
    decision_prices : Series instrument_id -> price the target was decided at.
    fill_prices     : Series instrument_id -> realized fill price.

    Returns ``(per_order_df, aggregate_bps)`` where ``aggregate_bps`` is the
    notional-weighted mean shortfall (``nan`` if no order has a usable price pair).
    """
    decision_prices = pd.Series(decision_prices, dtype=float)
    fill_prices = pd.Series(fill_prices, dtype=float)

    if orders is None or orders.empty:
        return pd.DataFrame(columns=SHORTFALL_COLUMNS), float("nan")

    rows: list[dict] = []
    for o in orders.itertuples(index=False):
        iid = o.instrument_id
        side = o.side
        sign = _SIDE_SIGN.get(side)
        dp = float(decision_prices.get(iid, np.nan))
        fp = float(fill_prices.get(iid, np.nan))
        if sign is None or not np.isfinite(dp) or dp <= 0.0 or not np.isfinite(fp):
            bps = float("nan")
        else:
            bps = sign * (fp - dp) / dp * 1e4
        notional = float(getattr(o, "notional_usd", np.nan))
        if not np.isfinite(notional):
            qty = float(getattr(o, "qty", np.nan))
            notional = qty * fp if np.isfinite(qty) and np.isfinite(fp) else np.nan
        rows.append({"instrument_id": iid, "side": side, "decision_price": dp,
                     "fill_price": fp, "notional_usd": notional, "shortfall_bps": bps})

    df = pd.DataFrame(rows, columns=SHORTFALL_COLUMNS)
    valid = df[np.isfinite(df["shortfall_bps"]) & np.isfinite(df["notional_usd"])
               & (df["notional_usd"] > 0)]
    if valid.empty:
        return df, float("nan")
    w = valid["notional_usd"].to_numpy()
    aggregate = float(np.average(valid["shortfall_bps"].to_numpy(), weights=w))
    return df, aggregate
