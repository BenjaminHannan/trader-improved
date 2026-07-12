"""Walk-forward P&L for the event (prediction-market) sleeve.

:func:`event_sleeve_returns` turns the curated ``event_markets`` panel into a *daily return
stream* the multi-sleeve engine can treat as one more sleeve. It is a pure point-in-time
function: the book decided at each rebalance date ``t`` is built from rows knowable at ``t``
(``available_from <= t``) only, and the daily return realized on a date ``d`` reads only
prices/availability dated ``<= d``. So corrupting any panel row dated after ``d`` cannot move
the return at or before ``d`` — the corruption harness in ``tests/test_backtest_engine.py``
exercises exactly this.

The sleeve has no optimizer / risk-model path of its own; sizing is the standalone
fractional-Kelly :func:`production.events.sizing.size_event_book`. Its risk enters the
portfolio purely through the EWMA *sleeve* covariance in
:func:`production.portfolio.allocation.sleeve_allocation` plus the ``events`` risk cap.

Modeling choices (documented approximations):
  * **signal -> edge**: the two documented signals (``longshot_bias`` + ``resolution_convergence``)
    emit a signed directional score in roughly ``[-1, 1]``; those scores are *summed* per
    instrument and passed straight through as the believed ``value`` (edge) that
    ``size_event_book`` interprets per its own convention (edge minus the fee haircut, then
    fractional Kelly capped per group / at the gross ``capital_frac``).
  * **resolution**: when a market runs past its last observation (or its ``close_time``) the
    position settles at the *snapped* value of its last observed price: ``>= 0.97 -> 1``,
    ``<= 0.03 -> 0``, otherwise the final observed price (a mid-book contract that never
    resolved cleanly in-sample is marked at its last print). The settlement P&L is realized on
    the first grid date after the last observation.
  * **P&L**: a position of ``weight`` entered at YES price ``entry`` earns, per day,
    ``weight * side * (p_d - p_prev) / entry`` (side ``+1`` YES, ``-1`` NO). Telescoped over the
    hold this is ``weight * side * (p_exit - entry) / entry`` — e.g. a long from 0.9 settling at
    1.0 makes ``weight * (1/0.9 - 1)``.
  * **cost**: the actual Kalshi taker fee (:func:`production.events.sizing.taker_fee_fraction`
    — an execution re-spec, 2026-07-11, replacing a flat ``FEE_BPS = 200`` (2%) round-trip
    placeholder; see ``production/events/sizing.py``'s module docstring for the fee schedule
    and ``diagnostics/events_tilt_backtest.json`` for the supporting evidence) is charged on
    each instrument's entered weight, at ITS OWN entry price, as a negative return on the
    first effective day of each rebalance — charged ONCE, at entry: Kalshi settlement is
    free, so a position held to settlement pays no exit fee. This changes the realized fee
    for every events signal that runs through this engine (``longshot_bias`` and
    ``resolution_convergence`` today), not just a hypothetical new one: the prior flat 200bp
    was a deliberately conservative placeholder (backlog iteration 25), roughly double the
    actual entry-only taker fee at a typical favorite price; this is a strictly more accurate
    replacement, not a change of sign or intent.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from production.events.markets import dedupe_related, liquid_universe
from production.events.signals import longshot_bias, resolution_convergence
from production.events.sizing import size_event_book, taker_fee_fraction


def _snap_resolution(price: float) -> float:
    """Settle a resolved market: >= 0.97 -> 1, <= 0.03 -> 0, else the final observed price."""
    p = float(price)
    if p >= 0.97:
        return 1.0
    if p <= 0.03:
        return 0.0
    return p


def _combined_signals(avail: pd.DataFrame) -> pd.DataFrame:
    """Sum the two documented signals into a single signed ``value`` (edge) per instrument.

    Both signals emit ``[obs_date, instrument_id, value]``; concatenated and summed on
    (obs_date, instrument_id) they express one blended directional edge. ``size_event_book``
    then takes the latest signal per instrument.
    """
    parts = [longshot_bias(avail), resolution_convergence(avail)]
    parts = [p for p in parts if p is not None and not p.empty]
    if not parts:
        return pd.DataFrame(columns=["obs_date", "instrument_id", "value"])
    sig = pd.concat(parts, ignore_index=True)
    sig = (sig.groupby(["obs_date", "instrument_id"], as_index=False)["value"].sum())
    return sig


def event_sleeve_returns(panel: pd.DataFrame, rebalance_dates, capital_frac: float = 0.10,
                         **sizing_kw) -> pd.Series:
    """Daily return series for the event sleeve over the panel's date range (see module docs)."""
    empty = pd.Series(dtype=float)
    if panel is None or panel.empty:
        return empty

    df = panel.copy()
    df["obs_date"] = pd.to_datetime(df["obs_date"])
    avail_from = pd.to_datetime(df["available_from"], utc=True, errors="coerce")

    # Daily grid = the panel's observed dates. Returns are indexed on this grid.
    grid = pd.DatetimeIndex(sorted(df["obs_date"].unique()))
    if len(grid) == 0:
        return empty

    # Wide last-observed YES price per instrument per date, plus each instrument's final print.
    px_wide = (df.pivot_table(index="obs_date", columns="instrument_id",
                              values="yes_price", aggfunc="last").sort_index())
    last_obs_date = df.groupby("instrument_id")["obs_date"].max()
    last_price = {iid: float(px_wide[iid].dropna().iloc[-1])
                  for iid in px_wide.columns if px_wide[iid].notna().any()}
    resolution = {iid: _snap_resolution(p) for iid, p in last_price.items()}

    def eff_price(iid: str, d: pd.Timestamp) -> float:
        """PIT effective price of ``iid`` at date ``d``: last print <= d while live, else the
        snapped settlement value once ``d`` runs past the last observation."""
        if iid not in px_wide.columns:
            return float("nan")
        if d > last_obs_date.get(iid, grid[-1]):
            return resolution.get(iid, float("nan"))
        col = px_wide[iid]
        sub = col[col.index <= d].dropna()
        return float(sub.iloc[-1]) if len(sub) else float("nan")

    rebals = pd.DatetimeIndex(sorted(pd.to_datetime(pd.Index(rebalance_dates))))
    rebals = rebals[(rebals >= grid[0]) & (rebals <= grid[-1])]

    out = pd.Series(0.0, index=grid, dtype=float)
    if len(rebals) == 0:
        return out

    for ri, t in enumerate(rebals):
        t_next = rebals[ri + 1] if ri + 1 < len(rebals) else grid[-1]
        # --- PIT book at t: only rows knowable by t ---
        mask = avail_from <= pd.Timestamp(t).tz_localize("UTC")
        avail = df[mask.to_numpy()]
        if avail.empty:
            continue
        universe = liquid_universe(avail, **{k: sizing_kw[k] for k in
                                             ("min_volume_usd", "min_days_to_close",
                                              "max_days_to_close") if k in sizing_kw})
        if not universe:
            continue
        groups = dedupe_related(universe)
        universe_ids = {m.id for m in universe}
        sig = _combined_signals(avail)
        if sig.empty:
            continue
        sig = sig[sig["instrument_id"].isin(universe_ids)]
        book_kw = {k: v for k, v in sizing_kw.items()
                   if k in ("per_group_cap",)}
        book = size_event_book(sig, avail, capital_frac=capital_frac, groups=groups,
                               **book_kw)
        if book.empty:
            continue

        # entry price = the latest PIT-knowable YES price at t (exactly the p_market the sizer
        # used) so the realized P&L telescopes cleanly from the entered price.
        entry_px = (avail.sort_values("obs_date", kind="stable")
                    .drop_duplicates("instrument_id", keep="last")
                    .set_index("instrument_id")["yes_price"])
        side = {row["instrument_id"]: (1.0 if row["side"] == "YES" else -1.0)
                for _, row in book.iterrows()}
        weight = {row["instrument_id"]: float(row["weight"])
                  for _, row in book.iterrows()}
        entry = {iid: float(entry_px.get(iid, np.nan)) for iid in weight}

        # holding dates: strictly after t, up to and including t_next (partition per rebalance)
        hold = grid[(grid > t) & (grid <= t_next)]
        if len(hold) == 0:
            continue
        # fee charged on the entered notional on the first effective day, per-instrument at
        # ITS OWN entry price (the taker fee is price-dependent, not a flat rate) — charged
        # ONCE, here, at entry; no exit/settlement fee is ever charged (Kalshi settlement is
        # free), so positions running to their last hold day incur no further cost.
        first = hold[0]
        fee = float(sum(
            weight[iid] * taker_fee_fraction(entry[iid])
            for iid in weight
            if np.isfinite(entry.get(iid, np.nan)) and entry[iid] > 0.0
        ))
        out.loc[first] += -fee

        # per-position running mark, seeded at the entry price so day-1 P&L is measured from
        # the entered level (not from a stale grid print).
        prev_mark = {iid: entry[iid] for iid in weight}
        for d in hold:
            day_ret = 0.0
            for iid in weight:
                e = entry.get(iid, float("nan"))
                if not np.isfinite(e) or e <= 0.0:
                    continue
                p_d = eff_price(iid, d)
                if not np.isfinite(p_d):
                    continue
                pm = prev_mark[iid]
                if not np.isfinite(pm):
                    prev_mark[iid] = p_d
                    continue
                day_ret += weight[iid] * side[iid] * (p_d - pm) / e
                prev_mark[iid] = p_d
            out.loc[d] += day_ret

    return out
