"""Event-market alpha signals — two documented, point-in-time edges.

Both take the curated ``event_markets`` panel and emit a long ``[obs_date,
instrument_id, value]`` frame, exactly like the price-sleeve signals. ``value`` is a
signed directional score: positive = buy YES, negative = buy NO (sell YES).

PIT discipline: every value at date D is a function only of rows knowable by end of day
D. :func:`longshot_bias` is pointwise (uses only that row's own price).
:func:`resolution_convergence` uses each market's *trailing* price history via a
positional ``shift`` on the obs_date-sorted series — a corrupted future price can never
move an earlier value.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

OUTPUT_COLUMNS = ["obs_date", "instrument_id", "value"]


def longshot_bias(panel: pd.DataFrame, low: float = 0.03, high: float = 0.15) -> pd.DataFrame:
    """Fade the favorite-longshot bias.

    The favorite-longshot bias (Griffith 1949; documented across racetrack, sports and
    prediction markets) is the empirical regularity that low-probability ("longshot")
    contracts are *overpriced* and high-probability ("favorite") contracts are
    *underpriced*: bettors overpay for lottery-like payoffs. The trade:

      * YES price in the open interval ``(low, high)`` -> ``value = -1``: the longshot is
        too dear, so **sell YES** (equivalently buy NO).
      * YES price in ``(1 - high, 1 - low)`` -> ``value = +1``: the near-certain favorite
        is too cheap, so **buy YES**.
      * anywhere else (mid-book, or extreme tails outside the bands) -> no position (the
        row is absent from the output).

    Pointwise in ``yes_price`` -> trivially point-in-time.
    """
    if panel is None or panel.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    p = pd.to_numeric(panel["yes_price"], errors="coerce")
    value = pd.Series(np.nan, index=panel.index)
    value[(p > low) & (p < high)] = -1.0          # overpriced longshot -> sell YES
    value[(p > 1 - high) & (p < 1 - low)] = 1.0    # underpriced favorite -> buy YES
    out = pd.DataFrame({
        "obs_date": panel["obs_date"], "instrument_id": panel["instrument_id"],
        "value": value,
    }).dropna(subset=["value"])
    return (out[OUTPUT_COLUMNS]
            .sort_values(["obs_date", "instrument_id"], kind="stable")
            .reset_index(drop=True))


def resolution_convergence(panel: pd.DataFrame, window: int = 5,
                           min_move: float = 0.0) -> pd.DataFrame:
    """Momentum toward resolution, weighted by proximity to the extremes.

    As a market nears its resolution date, price tends to drift *monotonically* toward
    the eventual 0/1 outcome as information accretes — a convergence/momentum effect.
    The signal is the sign of the trailing ``window``-day price drift, scaled by a
    proximity-to-extreme weight ``(1 - 2*|0.5 - price|)`` that is 1 at a coin-flip (most
    room to converge) and 0 at the 0/1 rails (already resolved, no edge left):

        value = sign(price_D - price_{D-window}) * (1 - 2*|0.5 - price_D|)

    Only markets within **30 days of close** are eligible — convergence is a late-life
    effect. Rows with ``|drift| < min_move`` or zero drift emit nothing.

    Trailing ``shift(window)`` per instrument keeps it point-in-time: the value at D reads
    ``price_D`` and ``price_{D-window}`` only, never a future bar.
    """
    if panel is None or panel.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    df = panel.copy()
    df["obs_date"] = pd.to_datetime(df["obs_date"])
    df = df.sort_values(["instrument_id", "obs_date"], kind="stable")
    price = pd.to_numeric(df["yes_price"], errors="coerce")

    # trailing drift over `window` bars, computed within each instrument's own history
    prev = df.groupby("instrument_id", sort=False)["yes_price"].shift(window)
    drift = price - pd.to_numeric(prev, errors="coerce")

    # days-to-close gate (<= 30d); markets with no close_time are ineligible
    obs_utc = df["obs_date"].dt.tz_localize("UTC")
    close = pd.to_datetime(df["close_time"], utc=True, errors="coerce")
    days_to_close = (close - obs_utc).dt.total_seconds() / 86400.0
    near = days_to_close.between(0, 30, inclusive="both")

    proximity = 1.0 - 2.0 * (0.5 - price).abs()   # 1 at p=0.5, 0 at p in {0,1}
    value = np.sign(drift) * proximity

    keep = (drift.abs() >= min_move) & (drift != 0) & near & value.notna()
    out = pd.DataFrame({
        "obs_date": df["obs_date"], "instrument_id": df["instrument_id"], "value": value,
    })[keep]
    return (out[OUTPUT_COLUMNS]
            .sort_values(["obs_date", "instrument_id"], kind="stable")
            .reset_index(drop=True))
