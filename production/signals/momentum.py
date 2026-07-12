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
    """Time-series momentum, canonical vol-scaled form (Moskowitz-Ooi-Pedersen 2012).

    ``r_12m / sigma_ann`` per instrument: the trailing 12-month return scaled by the
    instrument's trailing 252d annualized daily-return vol. The original v1 used
    ``sign(r_12m)`` — a two-valued score that is nearly degenerate under a
    cross-sectional rank at N≈8-16 (first live gate run, 2026-07-07; see
    research-rejected-factor-forensics). The vol-scaled continuous score is the
    canonical construction and carries cross-sectional resolution. The trailing vol
    window is strictly backward-looking (shift(1) before the rolling window feeds a
    same-day score).
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
        ret1d = close.pct_change(fill_method=None)
        sigma = (ret1d.shift(1)
                      .groupby(p["instrument_id"], sort=False)
                      .transform(lambda s: s.rolling(252, min_periods=126).std())
                 * np.sqrt(252.0))
        value = trailing / sigma.replace(0.0, np.nan)
        out = pd.DataFrame({"obs_date": p["obs_date"], "instrument_id": p["instrument_id"],
                            "value": value})
        return self._finalize(out, anchor=p)
