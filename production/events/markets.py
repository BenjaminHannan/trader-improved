"""Event-market universe construction: liquidity filtering and event de-duplication.

Two jobs, both operating on the curated ``event_markets`` long panel (one row per
market per day, columns ``obs_date, instrument_id, yes_price, volume, open_interest,
close_time, status`` plus, when the loader supplies them, ``event_key, question,
venue``):

  * :func:`liquid_universe` — reduce the panel to the tradable set: markets with enough
    dollar volume and a resolution date that is neither too soon (no edge left, wide
    settlement risk) nor too far out (capital tied up, little convergence). Evaluated at
    each market's *latest observed* row, so the filter is point-in-time.
  * :func:`dedupe_related` — collapse markets that resolve on the *same* underlying
    event into groups. This is the correlated-resolution point from the research: two
    contracts on one event are one bet, not two, and must not each draw a full position.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class EventMarket:
    """A single binary event contract, snapshotted at one decision point.

    ``id`` is the synthetic ``EV:<venue>:<market_id>`` instrument_id. ``yes_price`` is the
    YES probability in [0,1]; ``oi`` is open interest / on-book depth. ``event_key`` is
    the venue's shared-event identifier (Kalshi ``event_ticker`` / Polymarket
    ``conditionId``) used for de-duplication; ``None`` falls back to a question-prefix match.
    """

    id: str
    question: str
    close_time: pd.Timestamp
    yes_price: float
    volume: float
    oi: float
    venue: str
    event_key: str | None = None


def _latest_per_market(df: pd.DataFrame) -> pd.DataFrame:
    """Latest observed row per instrument_id — the point-in-time snapshot of each market."""
    return (df.sort_values("obs_date", kind="stable")
              .drop_duplicates("instrument_id", keep="last"))


def liquid_universe(df: pd.DataFrame, min_volume_usd: float = 1000.0,
                    min_days_to_close: int = 1, max_days_to_close: int = 120) -> list:
    """Filter the curated panel to a tradable list of :class:`EventMarket`.

    Filters (all evaluated at each market's most recent observation):
      * **liquidity** — ``volume >= min_volume_usd``; thin markets have unreliable prices
        and punishing slippage.
      * **nearness** — ``min_days_to_close <= days_to_close <= max_days_to_close`` where
        ``days_to_close = close_time - latest obs_date``. Below the floor there is no time
        for an edge to pay off and settlement mechanics dominate; above the ceiling capital
        is parked with little resolution convergence to harvest.

    Markets with a missing/NaT ``close_time`` are dropped (nearness cannot be assessed).
    """
    if df is None or df.empty:
        return []
    latest = _latest_per_market(df)
    obs = pd.to_datetime(latest["obs_date"])
    obs_utc = obs.dt.tz_localize("UTC") if obs.dt.tz is None else obs.dt.tz_convert("UTC")
    close = pd.to_datetime(latest["close_time"], utc=True, errors="coerce")
    days_to_close = (close - obs_utc).dt.total_seconds() / 86400.0

    out: list[EventMarket] = []
    for (_, row), dtc in zip(latest.iterrows(), days_to_close):
        if pd.isna(dtc):
            continue
        if float(row.get("volume") or 0.0) < min_volume_usd:
            continue
        if not (min_days_to_close <= dtc <= max_days_to_close):
            continue
        out.append(EventMarket(
            id=str(row["instrument_id"]),
            question=str(row.get("question") or row["instrument_id"]),
            close_time=pd.Timestamp(row["close_time"]),
            yes_price=float(row["yes_price"]),
            volume=float(row.get("volume") or 0.0),
            oi=float(row.get("open_interest") or 0.0),
            venue=str(row.get("venue") or str(row["instrument_id"]).split(":")[1]),
            event_key=(None if pd.isna(row.get("event_key")) else row.get("event_key")),
        ))
    return out


def _question_prefix(question: str, prefix_len: int) -> str:
    return str(question or "").strip().lower()[:prefix_len]


def dedupe_related(markets: list, prefix_len: int = 40) -> list:
    """Group markets that resolve on the same underlying event.

    Grouping key, in priority order:
      1. ``event_key`` when present (Kalshi ``event_ticker`` / Polymarket ``conditionId``)
         — the authoritative shared-event identifier from the raw payload;
      2. otherwise an **exact question-string prefix** (first ``prefix_len`` chars,
         normalized) — a heuristic fallback when no venue event key is available.

    Returns a list of groups (each a ``list[EventMarket]``). One group == one independent
    bet: sizing takes a single net position per group so correlated contracts never
    compound into an oversized exposure.
    """
    groups: dict[tuple, list] = {}
    for m in markets:
        if getattr(m, "event_key", None):
            key = ("evt", m.event_key)
        else:
            key = ("q", _question_prefix(m.question, prefix_len))
        groups.setdefault(key, []).append(m)
    return list(groups.values())
