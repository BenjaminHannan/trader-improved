"""Carry-family signals: perpetual-funding carry (crypto) and rate-differential
carry (fx ETFs).

FundingCarry is a pure trailing statistic on each instrument's funding series.
RateDifferentialCarry joins two macro rate series onto each fx ETF's price dates —
a lag-stamped source, so only macro rows with ``available_from <= end of day D`` may
inform the value at date D. That availability join (as-of by normalized availability
date) is what keeps it point-in-time.
"""
from __future__ import annotations

import pandas as pd

from production.signals.base import OUTPUT_COLUMNS, Signal, register

# fx ETF symbol -> (foreign 3m rate series, USD 3m rate series). UUP/UDN (dollar-index
# ETFs) have no single foreign leg and are skipped.
FX_RATE_SERIES = {
    "FXE": ("RATE_EU", "DGS3MO_US"),
    "FXY": ("RATE_JP", "DGS3MO_US"),
    "FXB": ("RATE_GB", "DGS3MO_US"),
    "FXA": ("RATE_AU", "DGS3MO_US"),
    "FXC": ("RATE_CA", "DGS3MO_US"),
    "FXF": ("RATE_CH", "DGS3MO_US"),
}


@register
class FundingCarry(Signal):
    """``-(trailing 7d mean funding_rate)``, crypto only.

    Positive perpetual funding means longs pay shorts, i.e. an expensive long — so the
    carry score is the negated trailing mean funding rate.
    """

    name = "carry_funding"
    sleeves = ["crypto"]
    required_datasets = ["funding"]
    min_history_days = 7
    horizon_days = 5

    def compute(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        f = self._restrict_to_sleeves(data["funding"])
        f = f.sort_values(["instrument_id", "obs_date"], kind="stable").reset_index(drop=True)
        if f.empty:
            return pd.DataFrame(columns=OUTPUT_COLUMNS)
        mean7 = (f.groupby("instrument_id", sort=False)["funding_rate"]
                 .transform(lambda s: s.rolling(7).mean()))
        out = pd.DataFrame({"obs_date": f["obs_date"], "instrument_id": f["instrument_id"],
                            "value": -mean7})
        return self._finalize(out, anchor=f)


def _asof_by_avail_date(macro: pd.DataFrame, series_id: str) -> pd.DataFrame:
    """Reduce a macro series to a step function keyed by *availability date*.

    ``available_from`` (a UTC timestamp) is normalized to its date: a value is usable
    at any decision date D with ``available_from <= end of day D``, i.e. once D reaches
    that availability date. Ties on the same availability date keep the latest-arriving
    vintage. Returns ``[avail_date, value]`` sorted by avail_date (merge_asof-ready).

    Vintage semantics (multi-vintage ALFRED series). A macro series may legitimately
    carry several vintages of the same ``obs_date`` — an original release plus later
    revisions — each stamped with its *own* ``available_from``. Every vintage is kept
    and placed at its own availability date, so the ``merge_asof`` consumer sees a step
    function in which a revision becomes effective **only on/after ITS available_from**:
    before that date the prior vintage is still the visible value, and a revision that is
    not yet knowable at decision date D can never move the value at D. Later vintages thus
    override earlier ones from their availability onward, which is exactly PIT-correct.
    The only dedup applied is for *same-availability duplicates of one obs_date* (a
    re-ingestion of the identical (obs_date, available_from)): those collapse to the
    latest-arriving row (by ``ingested_at`` when present) so a single obs_date cannot
    double-count at one availability. On single-vintage data this is a no-op.
    """
    cols = ["obs_date", "available_from", "value"]
    if "ingested_at" in macro.columns:
        cols.append("ingested_at")
    s = macro.loc[macro["series_id"] == series_id, cols].copy()
    if s.empty:
        return pd.DataFrame(columns=["avail_date", "value"])
    avail = pd.to_datetime(s["available_from"], utc=True)
    s["avail_date"] = avail.dt.normalize().dt.tz_localize(None)
    s["_af"] = avail
    # Same-availability duplicates of one obs_date -> keep the latest-arriving vintage.
    dedup_sort = ["_af"] + (["ingested_at"] if "ingested_at" in s.columns else [])
    s = (s.sort_values(dedup_sort, kind="stable")
         .drop_duplicates(["obs_date", "_af"], keep="last"))
    # Step function keyed by availability date; on a shared availability the most recently
    # available vintage is effective.
    s = (s.sort_values(["avail_date", "_af"], kind="stable")
         .drop_duplicates("avail_date", keep="last"))
    return s[["avail_date", "value"]].reset_index(drop=True)


@register
class RateDifferentialCarry(Signal):
    """fx-ETF carry: ``foreign_3m_rate - usd_3m_rate``.

    For each fx ETF the foreign and USD short-rate series are as-of joined (by
    availability date) onto the ETF's own price dates, then differenced. Only macro
    rows knowable by end of day D feed the value at D.
    """

    name = "carry_rate_diff"
    sleeves = ["fx_etf"]
    required_datasets = ["prices", "macro"]
    min_history_days = 21
    horizon_days = 21

    def compute(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        prices = self._restrict_to_sleeves(data["prices"])
        macro = data["macro"]
        if prices.empty:
            return pd.DataFrame(columns=OUTPUT_COLUMNS)

        # Cache each macro series' availability step-function once.
        series_cache: dict[str, pd.DataFrame] = {}

        def get_series(sid: str) -> pd.DataFrame:
            if sid not in series_cache:
                series_cache[sid] = _asof_by_avail_date(macro, sid)
            return series_cache[sid]

        frames = []
        for iid, grp in prices.groupby("instrument_id", sort=False):
            symbol = iid.split(":")[1]
            legs = FX_RATE_SERIES.get(symbol)
            if legs is None:  # e.g. UUP/UDN or an unmapped fx ETF
                continue
            foreign_sid, usd_sid = legs
            foreign = get_series(foreign_sid)
            usd = get_series(usd_sid)
            if foreign.empty or usd.empty:
                continue
            dates = grp[["obs_date"]].sort_values("obs_date", kind="stable").reset_index(drop=True)
            f = pd.merge_asof(dates, foreign, left_on="obs_date", right_on="avail_date",
                              direction="backward")
            u = pd.merge_asof(dates, usd, left_on="obs_date", right_on="avail_date",
                              direction="backward")
            value = f["value"] - u["value"]
            frames.append(pd.DataFrame({"obs_date": dates["obs_date"], "instrument_id": iid,
                                        "value": value}))
        if not frames:
            return pd.DataFrame(columns=OUTPUT_COLUMNS)
        out = pd.concat(frames, ignore_index=True)
        return self._finalize(out, anchor=prices)
