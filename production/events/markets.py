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
    Political markets get one extra rule on top: markets closing on the *same UTC
    calendar date* are unioned into one group even across different ``event_key``s
    (election nights co-resolve — state ladders across many separate events are not
    independent bets; see ``research/wiki/questions/research-political-underconfidence.md``,
    promotion check 4).
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
    ``category`` is the loader-stamped label (currently only Kalshi rows carry one —
    ``"politics"``/``"other"``/``"unknown"``); ``None`` when the source loader doesn't
    stamp it (e.g. Polymarket) or the panel row has no value. It drives the same-UTC-date
    political grouping rule in :func:`dedupe_related`.
    """

    id: str
    question: str
    close_time: pd.Timestamp
    yes_price: float
    volume: float
    oi: float
    venue: str
    event_key: str | None = None
    category: str | None = None


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
            category=(None if pd.isna(row.get("category")) else row.get("category")),
        ))
    return out


def _question_prefix(question: str, prefix_len: int) -> str:
    return str(question or "").strip().lower()[:prefix_len]


def _political_same_date_key(m) -> tuple | None:
    """Synthetic grouping key unioning political markets by same-UTC-close-date.

    Election nights co-resolve: many separately-ticketed state/race markets
    (different ``event_key``s) settle within hours of each other and are, in risk
    terms, one correlated bet — not N independent ones. Returns a key like
    ``("politics_date", "2024-11-05")`` (synthetic group key, e.g.
    ``"politics:2024-11-05"``, in string form) for any market whose ``category`` is
    ``"politics"`` (case-insensitive) and has a resolvable ``close_time``; returns
    ``None`` otherwise so the caller falls through to the ordinary
    event_key/question-prefix grouping. This runs BEFORE the event_key check, so it
    intentionally overrides — unions across — different event_key groups; it never
    fires for non-political markets, leaving their behavior unchanged.
    """
    if str(getattr(m, "category", None) or "").strip().lower() != "politics":
        return None
    close = getattr(m, "close_time", None)
    if close is None:
        return None
    close_ts = pd.Timestamp(close)
    if pd.isna(close_ts):
        return None
    close_utc = close_ts.tz_convert("UTC") if close_ts.tzinfo is not None else close_ts
    return ("politics_date", close_utc.date().isoformat())


def dedupe_related(markets: list, prefix_len: int = 40) -> list:
    """Group markets that resolve on the same underlying event.

    Grouping key, in priority order:
      0. political same-UTC-date union (:func:`_political_same_date_key`) — ANY
         market stamped ``category="politics"`` joins the single group for its
         close date, regardless of ``event_key`` (election-night co-resolution;
         promotion check 4 of the political-favorite-tilt study);
      1. otherwise ``event_key`` when present (Kalshi ``event_ticker`` / Polymarket
         ``conditionId``) — the authoritative shared-event identifier from the raw
         payload;
      2. otherwise an **exact question-string prefix** (first ``prefix_len`` chars,
         normalized) — a heuristic fallback when no venue event key is available.

    Non-political markets are entirely unaffected — rule 0 only ever matches rows
    the loader stamped ``category="politics"``.

    Returns a list of groups (each a ``list[EventMarket]``). One group == one independent
    bet: sizing takes a single net position per group so correlated contracts never
    compound into an oversized exposure.
    """
    groups: dict[tuple, list] = {}
    for m in markets:
        key = _political_same_date_key(m)
        if key is None:
            if getattr(m, "event_key", None):
                key = ("evt", m.event_key)
            else:
                key = ("q", _question_prefix(m.question, prefix_len))
        groups.setdefault(key, []).append(m)
    return list(groups.values())
