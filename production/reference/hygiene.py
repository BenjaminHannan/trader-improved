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
    counts = {"backstop_dropped": 0, "blocklist_dropped": 0}
    if df is None or len(df) == 0:
        return (df.copy() if df is not None else df), counts

    before = len(df)
    out = apply_price_backstop(df, min_price=min_price)
    counts["backstop_dropped"] = before - len(out)

    if symbol_col in out.columns and "obs_date" in out.columns and len(out):
        blocked = out.apply(lambda r: is_blocked(r[symbol_col], r["obs_date"]), axis=1)
        counts["blocklist_dropped"] = int(blocked.sum())
        if counts["blocklist_dropped"]:
            out = out.loc[~blocked].reset_index(drop=True)
    return out, counts
