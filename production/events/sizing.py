"""Position sizing for the event book — fractional Kelly with a cost haircut and caps.

Binary event contracts are Bernoulli bets, so the Kelly criterion applies cleanly. This
module turns a directional signal (a believed edge over the market price) into position
weights, subject to three disciplines the research insists on:

  * **costs are never zero** — the actual Kalshi taker-fee haircut (see
    :func:`taker_fee_fraction`) is subtracted from the raw edge before sizing; an edge
    that does not clear the haircut is not traded;
  * **fractional Kelly** — full Kelly is too aggressive under parameter uncertainty, so
    weights are scaled by ``KELLY_FRACTION`` (0.25x);
  * **caps** — one net position per correlated-event group (``per_group_cap``), and total
    gross exposure bounded by ``capital_frac``.

Fee model — execution re-spec (2026-07-11), replacing a flat ``FEE_BPS = 200`` (2%)
round-trip placeholder: :func:`taker_fee` / :func:`taker_fee_fraction` port the Kalshi
taker-fee schedule from ``research/diagnostics/_common.py::taker_fee`` (production
never imports ``research/``, per CLAUDE.md, so the formula is duplicated here, not
imported). Kalshi charges ``ceil-to-cent(0.07 * C * p * (1-p))`` dollars on ``C``
contracts at price ``p``; following Buergi, Deng & Whelan we impute the per-contract
rate from a 100-lot: ``c(p) = ceil(700 * p * (1-p)) / 10000`` (50c price -> 1.75c/
contract; 5c price -> 0.34c/contract). Maker fee is 25% of taker on the 2026-07-07
schedule; the events sleeve is a taker (crosses the spread on entry), so it prices the
taker side. The fee is charged ONCE, at entry — Kalshi settlement is free, so a
position held to settlement pays no exit fee. The prior flat 200bp was a deliberately
conservative placeholder (backlog iteration 25) for ALL events signals
(``longshot_bias``, ``resolution_convergence``, and — once wired —
``political_favorite_tilt``); this is a strictly more accurate replacement, evidence
in ``diagnostics/events_tilt_backtest.json``.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

KELLY_FRACTION = 0.25   # fractional-Kelly multiplier (quarter-Kelly)


def taker_fee(p: float) -> float:
    """Per-contract Kalshi taker fee in dollars at price ``p``, imputed on a 100-lot.

    ``ceil(700 * p * (1-p)) / 10000`` — see the module docstring for the derivation
    (ported from ``research/diagnostics/_common.py::taker_fee``, Whelan Table 8's
    fee schedule). ``p=0.50`` -> ``0.0175`` (1.75c); ``p=0.85`` -> ``0.009`` (0.9c).
    """
    p = float(p)
    return math.ceil(700.0 * p * (1.0 - p)) / 10000.0


def taker_fee_fraction(p: float) -> float:
    """Kalshi taker fee as a fraction of the entry price ``p``.

    A ``$1``-payout YES contract bought at price ``p`` costs ``p + taker_fee(p)``
    dollars; expressed as a fraction of the stake ``p`` that is
    ``taker_fee(p) / p``. Charged ONCE, at entry, per the module docstring. Returns
    ``0.0`` for a non-positive price (guards a degenerate/missing entry price rather
    than dividing by zero).
    """
    p = float(p)
    if p <= 0.0:
        return 0.0
    return taker_fee(p) / p


def bernoulli_variance(p: float) -> float:
    """Variance of a Bernoulli(p) outcome, ``p * (1 - p)``."""
    return float(p) * (1.0 - float(p))


def kelly_fraction(p_model: float, p_market: float, cap: float = 0.05) -> float:
    """Binary-outcome Kelly bet fraction for buying YES at ``p_market``.

    Derivation. A YES contract bought at price ``q = p_market`` pays 1 on a YES resolution
    and 0 otherwise, i.e. net odds ``b = (1 - q) / q`` (stake ``q`` to win ``1 - q``). With
    believed win probability ``p = p_model`` the Kelly fraction ``f* = (p*b - (1-p)) / b``
    simplifies to

        f* = (p_model - p_market) / (1 - p_market).

    When ``p_model < p_market`` the YES bet is negative-Kelly; by symmetry the mirror bet is
    to buy NO at price ``1 - p_market`` with belief ``1 - p_model``, whose Kelly fraction is
    ``(p_market - p_model) / p_market``. We return a *signed* fraction: positive = long YES,
    negative = long NO (its magnitude the NO-side Kelly), clipped to ``[-cap, cap]``.
    """
    p_model = float(p_model)
    p_market = float(p_market)
    edge = p_model - p_market
    if edge >= 0.0:
        denom = 1.0 - p_market
        f = edge / denom if denom > 0 else 0.0
    else:
        f = edge / p_market if p_market > 0 else 0.0   # negative -> NO side
    return float(np.clip(f, -cap, cap))


def size_event_book(signals_df: pd.DataFrame, panel: pd.DataFrame,
                    capital_frac: float = 0.10, per_group_cap: float = 0.02,
                    groups: list | None = None) -> pd.DataFrame:
    """Turn event signals into a book of ``[instrument_id, side, weight]`` positions.

    ``signals_df`` carries a signed ``value`` per instrument = the model's believed edge
    (YES probability minus market price). ``panel`` supplies the market price
    (``yes_price``) at each instrument's latest observation. For each instrument:

      1. subtract the actual Kalshi taker-fee haircut (:func:`taker_fee_fraction` of
         that instrument's market price — see the module docstring) from the edge
         magnitude; an edge that does not clear the haircut is dropped (**costs are
         never zero**);
      2. form ``p_model = p_market + net_edge`` and size with fractional Kelly
         (``KELLY_FRACTION`` x :func:`kelly_fraction`), capped at ``per_group_cap``.

    ``groups`` (from :func:`production.events.markets.dedupe_related`) collapses correlated
    markets: within each group only the **largest ``|value|``** instrument keeps a position
    (one net bet per event). Finally, if total gross weight exceeds ``capital_frac`` the
    whole book is scaled down proportionally so gross == ``capital_frac``.
    """
    if signals_df is None or signals_df.empty:
        return pd.DataFrame(columns=["instrument_id", "side", "weight"])

    # latest signal + latest market price per instrument
    sig = (signals_df.sort_values("obs_date", kind="stable")
           .drop_duplicates("instrument_id", keep="last")
           .set_index("instrument_id"))
    px = (panel.sort_values("obs_date", kind="stable")
          .drop_duplicates("instrument_id", keep="last")
          .set_index("instrument_id")["yes_price"])

    recs: list[dict] = []
    for iid, row in sig.iterrows():
        if iid not in px.index:
            continue
        p_market = float(px.loc[iid])
        edge = float(row["value"])
        haircut = taker_fee_fraction(p_market)
        net_edge = np.sign(edge) * max(abs(edge) - haircut, 0.0)
        if net_edge == 0.0:               # edge did not clear the cost floor -> no trade
            continue
        p_model = float(np.clip(p_market + net_edge, 1e-6, 1.0 - 1e-6))
        f_raw = kelly_fraction(p_model, p_market, cap=1.0)
        f = float(np.clip(KELLY_FRACTION * f_raw, -per_group_cap, per_group_cap))
        if f == 0.0:
            continue
        recs.append({"instrument_id": iid, "side": "YES" if f > 0 else "NO",
                     "weight": abs(f), "abs_signal": abs(edge)})

    if not recs:
        return pd.DataFrame(columns=["instrument_id", "side", "weight"])
    book = pd.DataFrame(recs)

    # one net position per dedupe group: the largest |signal| within the group wins
    if groups:
        member_to_group: dict[str, int] = {}
        for gi, grp in enumerate(groups):
            for m in grp:
                member_to_group[m.id] = gi
        book["group"] = book["instrument_id"].map(member_to_group)
        # ungrouped instruments each form their own singleton group
        ungrouped = book["group"].isna()
        book.loc[ungrouped, "group"] = (
            np.arange(int(ungrouped.sum())) + (len(groups) if groups else 0))
        book = (book.sort_values("abs_signal", kind="stable")
                    .drop_duplicates("group", keep="last"))

    # gross-exposure cap
    gross = book["weight"].sum()
    if gross > capital_frac and gross > 0:
        book["weight"] = book["weight"] * (capital_frac / gross)

    return (book[["instrument_id", "side", "weight"]]
            .sort_values("instrument_id", kind="stable")
            .reset_index(drop=True))
