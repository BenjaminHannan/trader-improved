"""Value signals: long-horizon reversal (equity) and crypto market-cap/TVL value.

`lt_reversal_5y` (De Bondt-Thaler long-term reversal) is registered in code as a
candidate but intentionally NOT listed in ``configs/factors.yaml`` — it needs ~5y of
history and stays a code-level candidate until it has been through the gate. It is
covered by the corruption harness like any registered signal.

`mcap_tvl` (CryptoMcapTvl) is the crypto "value" factor: a cheap protocol earns fee
revenue proportional to the capital locked in it (TVL) relative to its market cap, so
``log(TVL / mcap)`` is a fundamentals-vs-price ratio — high means the market pays little
per unit of locked capital. Market-cap and TVL are lag-stamped snapshot sources
(CoinGecko / DefiLlama), pulled the morning after their observation date, so — exactly
like the macro carry join — the value at date D may use only snapshot rows with
``available_from <= end of day D``. That availability-date ``merge_asof`` is what keeps
it point-in-time. Assets with no TVL (BTC and other non-smart-contract chains) carry no
TVL row and drop out of the factor naturally.
"""
from __future__ import annotations

import numpy as np
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


def _avail_step_by_instrument(df: pd.DataFrame, value_col: str) -> dict[str, pd.DataFrame]:
    """Reduce a per-instrument snapshot frame to per-instrument availability step functions.

    Each snapshot row's ``available_from`` (a UTC timestamp) is normalized to its date:
    the value is usable at any decision date D once ``available_from <= end of day D``,
    i.e. once D reaches that availability date. Ties on the same availability date keep
    the latest-arriving vintage. Returns ``{instrument_id: [avail_date, value_col]}``
    with each frame sorted by ``avail_date`` (merge_asof-ready).
    """
    if df is None or df.empty:
        return {}
    tmp = df[["instrument_id", "available_from", value_col]].copy()
    avail = pd.to_datetime(tmp["available_from"], utc=True)
    tmp["avail_date"] = avail.dt.normalize().dt.tz_localize(None)
    tmp["_af"] = avail
    tmp = (tmp.sort_values(["instrument_id", "avail_date", "_af"], kind="stable")
           .drop_duplicates(["instrument_id", "avail_date"], keep="last"))
    out: dict[str, pd.DataFrame] = {}
    for iid, grp in tmp.groupby("instrument_id", sort=False):
        out[iid] = grp[["avail_date", value_col]].reset_index(drop=True)
    return out


@register
class CryptoMcapTvl(Signal):
    """Crypto value: ``log(TVL / mcap)`` per crypto instrument.

    Total value locked (DefiLlama) over market cap (CoinGecko): a fundamentals-to-price
    ratio. Both are lag-stamped snapshot sources joined onto each instrument's own price
    dates by *availability date* (backward ``merge_asof``), so the value at date D uses
    only snapshot rows knowable by end of day D. Instruments without a TVL series (e.g.
    BTC, a non-smart-contract chain) produce no ratio and drop out of the factor.
    """

    name = "mcap_tvl"
    sleeves = ["crypto"]
    required_datasets = ["prices", "mcap", "tvl"]
    min_history_days = 30
    horizon_days = 21

    def compute(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        prices = self._restrict_to_sleeves(data["prices"])
        if prices.empty:
            return pd.DataFrame(columns=OUTPUT_COLUMNS)
        mcap_avail = _avail_step_by_instrument(data["mcap"], "mcap")
        tvl_avail = _avail_step_by_instrument(data["tvl"], "tvl")

        frames = []
        for iid, grp in prices.groupby("instrument_id", sort=False):
            mc = mcap_avail.get(iid)
            tv = tvl_avail.get(iid)
            if mc is None or tv is None or mc.empty or tv.empty:
                continue  # no TVL (e.g. BTC) -> instrument drops out of the factor
            dates = (grp[["obs_date"]].sort_values("obs_date", kind="stable")
                     .reset_index(drop=True))
            m = pd.merge_asof(dates, mc, left_on="obs_date", right_on="avail_date",
                              direction="backward")
            t = pd.merge_asof(dates, tv, left_on="obs_date", right_on="avail_date",
                              direction="backward")
            ratio = t["tvl"] / m["mcap"]
            value = np.log(ratio.where(ratio > 0))  # non-positive/NaN -> NaN, dropped
            frames.append(pd.DataFrame({"obs_date": dates["obs_date"],
                                        "instrument_id": iid, "value": value}))
        if not frames:
            return pd.DataFrame(columns=OUTPUT_COLUMNS)
        out = pd.concat(frames, ignore_index=True)
        return self._finalize(out, anchor=prices)
