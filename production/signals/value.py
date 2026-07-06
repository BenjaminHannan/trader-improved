"""Value / long-horizon reversal signal.

`lt_reversal_5y` (De Bondt-Thaler long-term reversal) is registered in code as a
candidate but intentionally NOT listed in ``configs/factors.yaml`` — it needs ~5y of
history and stays a code-level candidate until it has been through the gate. It is
covered by the corruption harness like any registered signal.

The crypto market-cap / TVL "value" variant is deferred to Stage 2 (needs the
reference market-cap and DeFiLlama TVL panels that are not yet ingested).
"""
from __future__ import annotations

import pandas as pd

from production.signals.base import OUTPUT_COLUMNS, Signal, register


@register
class LongHorizonReversal(Signal):
    """Long-term reversal: ``-(1260d return, skipping the most recent 252d)``.

    ``-(close.shift(252) / close.shift(1260) - 1)`` — the 5y-ago-to-1y-ago return,
    negated: multi-year winners tend to underperform subsequently. The recent 12
    months are skipped so this does not collide with 12-1 momentum.

    Equity-only for now; a crypto market-cap/TVL value variant is deferred to Stage 2.
    """

    name = "lt_reversal_5y"
    sleeves = ["equity"]
    required_datasets = ["prices"]
    min_history_days = 1300
    horizon_days = 63

    def compute(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        p = self._restrict_to_sleeves(data["prices"])
        p = p.sort_values(["instrument_id", "obs_date"], kind="stable").reset_index(drop=True)
        if p.empty:
            return pd.DataFrame(columns=OUTPUT_COLUMNS)
        close = p.groupby("instrument_id", sort=False)["close"]
        value = -(close.shift(252) / close.shift(1260) - 1.0)
        out = pd.DataFrame({"obs_date": p["obs_date"], "instrument_id": p["instrument_id"],
                            "value": value})
        return self._finalize(out, anchor=p)
