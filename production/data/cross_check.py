"""Cross-vendor price validation — defend against silent corporate-action corruption.

The motivating failure mode (researched in
`research/wiki/questions/research-vendor-cross-check.md`): yfinance is documented to
silently ship prices with a bad adjustment convention — split-adjusted closes spliced
onto unadjusted dividends, a missing split factor, an unadjusted split row. The result
is a *level* series that looks plausible but contains a phantom jump on one date. Such
corruption silently poisons every downstream return, IC, and backtest.

The standard defense named across sources is to cross-check against an independent
second feed (we already ingest Stooq as a fallback). Two design choices matter:

* **Compare RETURNS, not levels.** Vendors legitimately differ in adjustment
  *convention* — one may back-adjust the whole history for a split, another may not —
  so absolute levels diverge for entirely benign reasons and would false-positive
  everywhere. A daily *return* is convention-invariant on all but the event day, so a
  same-day return divergence beyond market noise (>~50bp on liquid names) isolates a
  bad bar / bad adjustment to the exact date it happened.

* **A split shows up as exactly ONE bad return.** A 2:1 split mis-applied from date D
  halves every close from D onward. In level space the two feeds disagree on every date
  >= D; in return space they disagree only on D itself (the ratio close[D]/close[D-1]),
  because from D+1 on both feeds are internally consistent again. So the flag is sharp:
  one date, not a smear — which is why we report per-date flags and a flagged *fraction*
  rather than a level RMSE.

`quarantine_list` turns a report into the set of instruments whose flagged fraction is
too high to trust, for the hygiene layer to exclude from the universe until a human
resolves the discrepancy.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from production.core.lake import Lake


def _daily_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Per-instrument daily pct returns on each frame's OWN sorted dates.

    Each vendor keeps its own trading calendar; returns are computed within a vendor
    (successive close ratio on that vendor's dates). We also carry ``prev_date`` — the
    date the return is measured *from* — so the join can compare only returns that span
    the same pair of dates. That guard is what makes a *missing bar* in one feed drop out
    of the comparison instead of manufacturing a fake divergence: after a gap, a feed's
    return straddles multiple days, its ``prev_date`` no longer matches the other feed's,
    and the observation is excluded rather than flagged.
    """
    out = df[["obs_date", "instrument_id", "close"]].copy()
    out["obs_date"] = pd.to_datetime(out["obs_date"])
    out = out.sort_values(["instrument_id", "obs_date"])
    grp = out.groupby("instrument_id")
    out["ret"] = grp["close"].pct_change()
    out["prev_date"] = grp["obs_date"].shift(1)
    return out.dropna(subset=["ret"])[["obs_date", "instrument_id", "prev_date", "ret"]]


def cross_vendor_report(primary: pd.DataFrame, secondary: pd.DataFrame,
                        return_divergence_bp: float = 50.0,
                        min_overlap: int = 60) -> dict:
    """Compare two curated price frames in return space and report divergences.

    Both frames are in curated price schema ``[obs_date, instrument_id, close, ...]``.
    For each frame we compute per-instrument daily pct returns on that frame's own
    sorted dates, inner-join the returns on ``(obs_date, instrument_id)``, and for every
    instrument with ``>= min_overlap`` joint return observations report:

      * ``n_overlap``            joint return-observation count
      * ``corr``                 Pearson correlation of the two return series
      * ``max_abs_divergence_bp``  largest ``|ret_p - ret_s|`` in basis points
      * ``flagged_dates``        ISO dates where ``|ret_p - ret_s| * 1e4 > threshold``
      * ``flag_frac``            fraction of joint observations that are flagged

    Instruments with fewer than ``min_overlap`` joint observations are listed (with their
    count) under ``insufficient_overlap`` and not scored — there is too little data to
    tell a bad adjustment from noise.

    See the module docstring / research/wiki/questions/research-vendor-cross-check.md for
    WHY this works in return space rather than on levels.
    """
    rp = _daily_returns(primary)
    rs = _daily_returns(secondary)
    joined = rp.merge(rs, on=["obs_date", "instrument_id"],
                      suffixes=("_p", "_s"), how="inner")
    # Compare only returns spanning the same pair of dates — a missing bar in one feed
    # leaves a straddled (multi-day) return whose prev_date no longer matches, so it is
    # excluded here rather than flagged as a bogus divergence.
    joined = joined[joined["prev_date_p"] == joined["prev_date_s"]]

    per_instrument: dict[str, dict] = {}
    insufficient: dict[str, int] = {}

    for iid, g in joined.groupby("instrument_id"):
        n = int(len(g))
        if n < min_overlap:
            insufficient[iid] = n
            continue
        diff_bp = (g["ret_p"] - g["ret_s"]).abs() * 1e4
        flagged_mask = diff_bp > return_divergence_bp
        flagged_dates = sorted(
            pd.to_datetime(g.loc[flagged_mask, "obs_date"]).dt.strftime("%Y-%m-%d")
        )
        # corr is undefined if either series is constant over the window
        if g["ret_p"].std() == 0 or g["ret_s"].std() == 0:
            corr = float("nan")
        else:
            corr = float(np.corrcoef(g["ret_p"], g["ret_s"])[0, 1])
        per_instrument[iid] = {
            "n_overlap": n,
            "corr": corr,
            "max_abs_divergence_bp": float(diff_bp.max()),
            "flagged_dates": flagged_dates,
            "flag_frac": float(flagged_mask.mean()),
        }

    flagged = {iid: r for iid, r in per_instrument.items() if r["flagged_dates"]}
    worst = None
    if per_instrument:
        worst = max(per_instrument.items(),
                    key=lambda kv: kv[1]["max_abs_divergence_bp"])[0]

    return {
        "threshold_bp": float(return_divergence_bp),
        "min_overlap": int(min_overlap),
        "instruments": per_instrument,
        "insufficient_overlap": insufficient,
        "summary": {
            "instruments_checked": len(per_instrument),
            "instruments_flagged": len(flagged),
            "worst_offender": worst,
        },
    }


def quarantine_list(report: dict, max_flag_frac: float = 0.02) -> list[str]:
    """Instrument ids whose flagged fraction exceeds `max_flag_frac`, sorted.

    These are the names too divergent between vendors to trust — the hygiene layer
    should drop them from the universe until the discrepancy is resolved by hand.
    """
    return sorted(
        iid for iid, r in report.get("instruments", {}).items()
        if r["flag_frac"] > max_flag_frac
    )


def write_cross_check_audit(report: dict, lake: Lake,
                            name: str = "cross_check_prices") -> Path:
    """Persist the cross-check report to the lake audit zone as JSON."""
    return lake.write_audit(report, name)
