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


# ----------------------------------------------------------------------------------------
# Documented re-specification (backlog #20, 2026-07-10/11): the taker-side longshot fade
# has NO measured net edge on Kalshi US-macro-release "economics" ladders at daily
# granularity. Three independent pre-registered diagnostics on 71k resolved-market rows
# (settlements 2021-07-15 through 2026-07-02) concur:
#
#   1. diagnostics/kalshi_longshot_fade.json, P2 — the tradeable NO-side fade of 5-20c
#      longshots nets -0.91% post-fee (t=-0.40, n=352); the pre-registered bar was
#      t >= +1.645 -> FAIL. (P1's -76.9%, t=-9.7 confirms the raw <=10c tail bias is real
#      but is the *fee-eaten* longshot side itself, not a tradeable NO-side edge.)
#   2. diagnostics/events_calibration.json — an isotonic recalibration of macro entry
#      prices fails to beat the raw price on held-out Brier score (A1: brier_price=0.0781
#      vs brier_g=0.0778, paired bootstrap p=0.41 -> FAIL); prices in this sample are
#      already calibrated outside the <=5c tail, and that tail is exactly what (1) shows is
#      fee-eaten.
#   3. the pooled Mincer-Zarnowitz slope on the macro-only sample is insignificant
#      (psi=0.0176, t=1.30, n=2180) even though Burgi-Deng-Whelan's cross-category slope is
#      significant; Whelan's Table 8 shows the favorite-longshot bias concentrates in OTHER
#      categories (sports, entertainment, ...), not economics.
#
# Evidence-granularity caveat: measured at T-1 daily bars on Kalshi economics ladders only.
# This does NOT bear on :func:`resolution_convergence` (no evidence against it), on
# non-Kalshi venues (Polymarket was not sampled), or on non-macro Kalshi series.
#
# The 12 measured series (current-generation ``KX``-prefixed tickers) plus their legacy
# unprefixed aliases (older Kalshi markets used the bare series name before the ``KX``
# rebrand). Exact-match only — NOT a prefix/substring check, so an unmeasured series that
# merely *starts with* a measured name (e.g. "GDPUSMIN", "CPIDELAY") is not caught by this
# set and defaults to tradeable.
_EXCLUDED_MACRO_SERIES = frozenset({
    "KXCPI", "KXCPIYOY", "KXCPICORE", "KXCPICOREYOY", "KXPCECORE",
    "KXPAYROLLS", "KXUSNFP", "KXU3", "KXJOBLESS", "KXFED", "KXFEDDECISION", "KXGDP",
    "CPI", "CPIYOY", "CPICORE", "CPICOREYOY", "PCECORE",
    "PAYROLLS", "USNFP", "U3", "JOBLESS", "FED", "FEDDECISION", "GDP",
})


def _series_of(event_key, instrument_id: str) -> str:
    """The series prefix of a market: the token before the first ``-``.

    Prefers ``event_key`` (the venue's shared-event ticker, e.g. ``"KXCPIYOY-26MAY"`` ->
    ``"KXCPIYOY"``). Falls back to the ticker embedded in ``instrument_id``
    (``"EV:kalshi:KXCPIYOY-26MAY-T4.2"`` -> ``"KXCPIYOY"``) when ``event_key`` is missing —
    the Kalshi series ticker is always the event ticker's own prefix before its first
    ``-``, so the same split works on either string.
    """
    key = event_key
    if pd.isna(key) or str(key) == "":
        parts = str(instrument_id).split(":", 2)
        key = parts[2] if len(parts) == 3 else str(instrument_id)
    return str(key).split("-", 1)[0]


def _excluded_macro_kalshi_mask(panel: pd.DataFrame) -> pd.Series:
    """Rows on a Kalshi market whose series is one of the 12 excluded macro series.

    Venue-gated: only ``venue == "kalshi"`` rows are eligible, so a Polymarket market
    sharing the same series text is never excluded. Missing ``venue``/``event_key``
    columns default to *not excluded* (tradeable), matching the exact-match-only,
    fail-open design of :data:`_EXCLUDED_MACRO_SERIES`.
    """
    if "venue" in panel.columns:
        is_kalshi = panel["venue"].astype(str).str.lower() == "kalshi"
    else:
        is_kalshi = pd.Series(False, index=panel.index)
    event_key = panel["event_key"] if "event_key" in panel.columns else pd.Series(None, index=panel.index)
    series = pd.Series(
        [_series_of(ek, iid) for ek, iid in zip(event_key, panel["instrument_id"])],
        index=panel.index,
    )
    return is_kalshi & series.isin(_EXCLUDED_MACRO_SERIES)


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

    **Re-specification (backlog #20)**: on Kalshi markets (``venue == "kalshi"``) whose
    series is one of the 12 measured US-macro-release series in
    :data:`_EXCLUDED_MACRO_SERIES` (or a legacy unprefixed alias), this signal never
    fires — see that frozenset's docstring for the three diagnostics that established no
    net edge there. Every other market — other Kalshi series, and all Polymarket markets
    regardless of series text — is unaffected.
    """
    if panel is None or panel.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    p = pd.to_numeric(panel["yes_price"], errors="coerce")
    value = pd.Series(np.nan, index=panel.index)
    value[(p > low) & (p < high)] = -1.0          # overpriced longshot -> sell YES
    value[(p > 1 - high) & (p < 1 - low)] = 1.0    # underpriced favorite -> buy YES
    value[_excluded_macro_kalshi_mask(panel)] = np.nan  # re-spec: no macro-release fade on Kalshi
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
