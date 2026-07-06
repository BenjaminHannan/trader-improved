"""Positioning signal from CFTC Commitments-of-Traders (COT) data.

COT is the archetypal lag-stamped source: the survey is observed on a Tuesday but only
released the following Friday 20:30 UTC. So the value from a Tuesday observation only
becomes usable on that Friday. We compute a rolling z-score of the non-commercial net
positioning ratio on the weekly series, stamp each z at its *availability date*, and
then forward-fill onto the instrument's daily price dates. Crowded longs (high net
positioning) are a contrarian negative, so the z is negated.
"""
from __future__ import annotations

import pandas as pd

from production.signals.base import OUTPUT_COLUMNS, Signal, register


@register
class CotPositioning(Signal):
    """``-(156-week rolling z of noncomm_net/open_interest)`` with ``min_periods=52``.

    The z is computed on the weekly COT series (backward-looking window), stamped at
    the release/availability date, then forward-filled onto daily price dates using an
    as-of join — a value is usable at date D only once its release date has arrived.
    """

    name = "cot_positioning"
    sleeves = ["fx_etf", "commodity_etf"]
    required_datasets = ["prices", "cot"]
    min_history_days = 156
    horizon_days = 21

    def compute(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        cot = self._restrict_to_sleeves(data["cot"])
        prices = self._restrict_to_sleeves(data["prices"])
        if cot.empty or prices.empty:
            return pd.DataFrame(columns=OUTPUT_COLUMNS)

        cot = cot.sort_values(["instrument_id", "obs_date"], kind="stable").reset_index(drop=True)
        ratio = cot["noncomm_net"] / cot["open_interest"]
        grp = ratio.groupby(cot["instrument_id"], sort=False)
        mean = grp.transform(lambda s: s.rolling(156, min_periods=52).mean())
        std = grp.transform(lambda s: s.rolling(156, min_periods=52).std())
        z = (ratio - mean) / std
        # Availability (release) date: normalize the Friday-20:30-UTC timestamp to a
        # date. The value is usable at any decision date D >= this release date.
        avail_date = (pd.to_datetime(cot["available_from"], utc=True)
                      .dt.normalize().dt.tz_localize(None))
        zf = pd.DataFrame({"instrument_id": cot["instrument_id"], "avail_date": avail_date,
                           "value": -z})

        price_dates = prices[["instrument_id", "obs_date"]]
        frames = []
        for iid, pgrp in price_dates.groupby("instrument_id", sort=False):
            right = (zf.loc[zf["instrument_id"] == iid, ["avail_date", "value"]]
                     .sort_values("avail_date", kind="stable").reset_index(drop=True))
            if right.empty:
                continue
            left = pgrp[["obs_date"]].sort_values("obs_date", kind="stable").reset_index(drop=True)
            merged = pd.merge_asof(left, right, left_on="obs_date", right_on="avail_date",
                                   direction="backward")
            frames.append(pd.DataFrame({"obs_date": merged["obs_date"], "instrument_id": iid,
                                        "value": merged["value"]}))
        if not frames:
            return pd.DataFrame(columns=OUTPUT_COLUMNS)
        out = pd.concat(frames, ignore_index=True)
        return self._finalize(out, anchor=prices)
