"""Momentum-family signals: 12-1 cross-sectional momentum, short-term reversal, tsmom.

All three are pure functions of each instrument's own trailing close series. The
shifts are *positional* on the instrument's sorted trading-day series (via groupby),
so no future bar can influence a past value — the PIT property the corruption harness
verifies.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from production.signals.base import OUTPUT_COLUMNS, Signal, register


def _sorted_prices(sig: Signal, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Sleeve-restricted price rows, sorted per instrument by trading day."""
    prices = sig._restrict_to_sleeves(data["prices"])
    return prices.sort_values(["instrument_id", "obs_date"], kind="stable").reset_index(drop=True)


@register
class Momentum12m1m(Signal):
    """12-1 momentum: return from ~12 months ago to ~1 month ago.

    ``close.shift(21) / close.shift(252) - 1`` on each instrument's own trading-day
    series — the classic momentum window that skips the most recent month to avoid
    short-term reversal contamination.
    """

    name = "mom_12_1"
    sleeves = ["equity", "crypto", "fx_etf", "commodity_etf"]
    required_datasets = ["prices"]
    min_history_days = 273
    horizon_days = 21

    def compute(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        p = _sorted_prices(self, data)
        if p.empty:
            return pd.DataFrame(columns=OUTPUT_COLUMNS)
        close = p.groupby("instrument_id", sort=False)["close"]
        value = close.shift(21) / close.shift(252) - 1.0
        out = pd.DataFrame({"obs_date": p["obs_date"], "instrument_id": p["instrument_id"],
                            "value": value})
        return self._finalize(out, anchor=p)


@register
class ShortTermReversal1m(Signal):
    """1-month short-term reversal: ``-(close / close.shift(21) - 1)``.

    Recent winners tend to revert over the following weeks, so the sign is flipped.
    """

    name = "str_reversal_1m"
    sleeves = ["equity", "crypto"]
    required_datasets = ["prices"]
    min_history_days = 21
    horizon_days = 5

    def compute(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        p = _sorted_prices(self, data)
        if p.empty:
            return pd.DataFrame(columns=OUTPUT_COLUMNS)
        close = p.groupby("instrument_id", sort=False)["close"]
        value = -(p["close"] / close.shift(21) - 1.0)
        out = pd.DataFrame({"obs_date": p["obs_date"], "instrument_id": p["instrument_id"],
                            "value": value})
        return self._finalize(out, anchor=p)


@register
class TimeSeriesMomentum(Signal):
    """Time-series momentum: ``sign(close / close.shift(252) - 1)``.

    A trend-following signal used on the macro sleeves (fx/commodity ETFs): +1 if the
    instrument is above its ~12-month-ago level, -1 if below.
    """

    name = "tsmom"
    sleeves = ["fx_etf", "commodity_etf"]
    required_datasets = ["prices"]
    min_history_days = 252
    horizon_days = 21

    def compute(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        p = _sorted_prices(self, data)
        if p.empty:
            return pd.DataFrame(columns=OUTPUT_COLUMNS)
        close = p.groupby("instrument_id", sort=False)["close"]
        trailing = p["close"] / close.shift(252) - 1.0
        value = np.sign(trailing)
        out = pd.DataFrame({"obs_date": p["obs_date"], "instrument_id": p["instrument_id"],
                            "value": value})
        return self._finalize(out, anchor=p)
