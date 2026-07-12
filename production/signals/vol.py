"""Volatility signal: low-volatility anomaly.

Low-vol names have historically earned higher risk-adjusted returns, so the raw
trailing volatility is negated to make "low vol" a positive score. Computed on each
instrument's own trailing daily returns — strictly backward-looking.
"""
from __future__ import annotations

import pandas as pd

from production.signals.base import OUTPUT_COLUMNS, Signal, register


@register
class LowVolatility(Signal):
    """``-(63d rolling std of daily pct returns)`` with ``min_periods=42``."""

    name = "low_vol"
    sleeves = ["equity", "crypto"]
    required_datasets = ["prices"]
    min_history_days = 63
    horizon_days = 21

    def compute(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        p = self._restrict_to_sleeves(data["prices"])
        p = p.sort_values(["instrument_id", "obs_date"], kind="stable").reset_index(drop=True)
        if p.empty:
            return pd.DataFrame(columns=OUTPUT_COLUMNS)
        rets = p.groupby("instrument_id", sort=False)["close"].pct_change(fill_method=None)
        vol = (rets.groupby(p["instrument_id"], sort=False)
               .transform(lambda s: s.rolling(63, min_periods=42).std()))
        out = pd.DataFrame({"obs_date": p["obs_date"], "instrument_id": p["instrument_id"],
                            "value": -vol})
        return self._finalize(out, anchor=p)
