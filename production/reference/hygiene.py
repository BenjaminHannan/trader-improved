"""Data-hygiene backstops for the two failure modes that silently corrupt equity
backtests: **ticker reuse** and **penny/garbage price rows**.

The synthetic `instrument_id` (see :mod:`production.reference.instruments`) already
defeats most reuse at the schema level. This module is the belt-and-braces layer for
cases where a vendor hands us a *bare symbol* with no id context — chiefly raw
yfinance history, whose adjusted series happily splices unrelated companies onto a
recycled ticker. The blocklist documents the known real-world cases; the price
backstop drops rows that no legitimate S&P 500 / ETF instrument could produce.
"""
from __future__ import annotations

import pandas as pd

# Sentinel far-future date for an open-ended block (rename/delisting is permanent).
_OPEN = "2100-01-01"

# Real, documented cases of yfinance ticker reuse / invalidation. Each symbol maps to a
# list of ``(from, to)`` inclusive-date windows during which the *bare* symbol must NOT
# be trusted, because within that window it either references a different/renamed entity
# or returns stale delisted data.
#
# Extend by appending a window tuple. Keep an inline comment stating the real event so
# the list stays auditable rather than becoming folklore.
REUSED_TICKER_BLOCKLIST: dict[str, list[tuple[str, str]]] = {
    # Facebook -> Meta: the "FB" ticker changed to "META" on 2022-06-09. After that
    # date bare "FB" is invalid; some vendors recycle it, so block it going forward.
    "FB": [("2022-06-09", _OPEN)],
    # Twitter taken private by Musk; "TWTR" delisted 2022-10-27. Any "TWTR" data after
    # is stale/garbage.
    "TWTR": [("2022-10-27", _OPEN)],
    # --- template for future additions -------------------------------------------
    # "XYZ": [("YYYY-MM-DD", "YYYY-MM-DD")],  # <reason: rename/delist/reuse event>
}


def is_blocked(symbol: str, date) -> bool:
    """True if `symbol` falls inside any blocked window on `date` (inclusive)."""
    windows = REUSED_TICKER_BLOCKLIST.get(symbol)
    if not windows:
        return False
    ts = pd.Timestamp(date)
    return any(pd.Timestamp(lo) <= ts <= pd.Timestamp(hi) for lo, hi in windows)


def apply_price_backstop(prices: pd.DataFrame, min_price: float = 0.10) -> pd.DataFrame:
    """Drop rows whose ``close`` is below `min_price`, preserving all other columns.

    Why: a sub-$0.10 close on an S&P 500 / liquid-ETF / major-crypto name is never a
    real quote in this universe. It signals either (a) delisted-symbol reuse where the
    vendor spliced a defunct penny stock onto a recycled ticker, or (b) a corrupted
    vendor row (unadjusted split, decimal error). Either way it explodes downstream
    returns (a $0.05 -> $50 "recovery" is a 1000x fake gain), so it is removed before
    anything computes a return. NaN closes are dropped as well — they cannot clear the
    floor and carry no usable information.
    """
    if prices.empty or "close" not in prices.columns:
        return prices.copy()
    keep = prices["close"] >= min_price
    return prices.loc[keep].reset_index(drop=True)


def apply_flap_screen(prices: pd.DataFrame, max_ratio: float = 20.0,
                      window: int = 5) -> pd.DataFrame:
    """Drop rows whose ``close`` deviates from the instrument's rolling median by
    more than ``max_ratio`` in either direction.

    Why: the first live cross-vendor run (2026-07-07) found dead tickers whose
    vendor series *flap* between two price regimes — EQ:TIE oscillated $2 <-> $15,600
    day-to-day (fake +810,000% returns), CFC 348x, MI 95x, BMC 22x. One such series
    dominates every cross-sectional statistic computed from the panel (an equal-weight
    "market" made from this data had 1,750% annualized vol and correlation ~0.02 with
    Mkt-RF). The signature is REVERSION: the series keeps snapping back, so the
    rolling median stays at the true level while the flap rows sit 20x+ away from it.
    Genuine violent repricings survive because the median follows them within days —
    GME's Jan-2021 squeeze peaks ~5x its trailing 5-day median, far under the 20x
    default. The screen needs >= ``window`` observations per instrument; shorter
    series pass through untouched (nothing to compare against).
    """
    if prices.empty or "close" not in prices.columns:
        return prices.copy()
    if "instrument_id" not in prices.columns or "obs_date" not in prices.columns:
        return prices.copy()

    df = prices.sort_values(["instrument_id", "obs_date"])
    med = (df.groupby("instrument_id")["close"]
             .transform(lambda s: s.rolling(window, center=True,
                                            min_periods=window).median()))
    ratio = df["close"] / med
    # NaN median (series shorter than `window`, or edge rows) -> keep the row.
    keep = ratio.isna() | ((ratio <= max_ratio) & (ratio >= 1.0 / max_ratio))
    return df.loc[keep].sort_index().reset_index(drop=True)


def drop_corrupt_series(prices: pd.DataFrame, max_extreme_moves: int = 5,
                        log_ratio: float = 1.609) -> tuple[pd.DataFrame, list[str]]:
    """Drop ENTIRE instrument series whose close makes catastrophic moves repeatedly.

    Complements :func:`apply_flap_screen` (row-level): a wrong-entity vendor series
    can alternate between price regimes in multi-day RUNS (EQ:TIE sat at ~$2 and
    ~$13,400 in alternating stretches), so run interiors look locally normal and
    survive any row screen. But no genuine large-cap series makes more than a few
    >400% (``log_ratio``=log 5) day-over-day moves EVER — GME's squeeze peaks at
    +135% — while a regime-alternator racks up dozens (every flip is one). A series
    exceeding ``max_extreme_moves`` such moves is corrupt end-to-end and is removed
    wholesale. Returns ``(clean, dropped_instrument_ids)``.
    """
    if prices.empty or "close" not in prices.columns \
            or "instrument_id" not in prices.columns or "obs_date" not in prices.columns:
        return prices.copy(), []
    import numpy as np

    df = prices.sort_values(["instrument_id", "obs_date"])
    logc = np.log(df["close"].where(df["close"] > 0))
    jumps = logc.groupby(df["instrument_id"]).diff().abs() > log_ratio
    n_extreme = jumps.groupby(df["instrument_id"]).sum()
    corrupt = sorted(n_extreme[n_extreme > max_extreme_moves].index)
    if not corrupt:
        return prices.copy(), []
    return prices[~prices["instrument_id"].isin(corrupt)].reset_index(drop=True), corrupt


def apply_hygiene(df: pd.DataFrame, symbol_col: str = "symbol",
                  min_price: float = 0.10) -> tuple[pd.DataFrame, dict]:
    """Apply both hygiene backstops in one pass and report what was dropped.

    Runs the sub-``min_price`` price backstop, then the reused-ticker blocklist
    (dropping rows where ``is_blocked(symbol, obs_date)`` is true for the value in
    ``symbol_col``). Returns ``(clean_df, {"backstop_dropped": n1,
    "blocklist_dropped": n2})``. The counts are how the caller keeps the drops
    auditable rather than silent.

    If ``symbol_col`` is absent (or there is no ``obs_date`` to date the block
    window) the blocklist step is skipped and counted as zero — the price backstop
    still runs. This is the single implementation the price loaders (and any future
    price vendor) share, so hygiene is wired once and enforced everywhere.
    """
    counts = {"backstop_dropped": 0, "blocklist_dropped": 0, "flap_dropped": 0,
              "corrupt_series": []}
    if df is None or len(df) == 0:
        return (df.copy() if df is not None else df), counts

    before = len(df)
    out = apply_price_backstop(df, min_price=min_price)
    counts["backstop_dropped"] = before - len(out)

    before = len(out)
    out, corrupt_ids = drop_corrupt_series(out)
    out = apply_flap_screen(out)
    counts["flap_dropped"] = before - len(out)
    counts["corrupt_series"] = corrupt_ids

    if symbol_col in out.columns and "obs_date" in out.columns and len(out):
        blocked = out.apply(lambda r: is_blocked(r[symbol_col], r["obs_date"]), axis=1)
        counts["blocklist_dropped"] = int(blocked.sum())
        if counts["blocklist_dropped"]:
            out = out.loc[~blocked].reset_index(drop=True)
    return out, counts
