#!/usr/bin/env python
"""Stamp ``category`` onto the existing curated ``event_markets_hist`` rows.

The incident (2026-07-11): the events sleeve went live in the headline backtest
(commit d665a3b) but traded ZERO. ``production/events/signals.py::
political_favorite_tilt`` fail-closes without a ``category == "politics"`` column
(deliberately — see that function's docstring on the fail-CLOSED posture). The
live-snapshot loader (``production/data/loaders/kalshi.py``) has always stamped
``category``; the historical backfill loader
(``production/data/loaders/kalshi_history.py``) never did, even though it already
enumerates series BY category for its own category-scoped runs. That loader now
carries a forward fix (stamps ``category`` on every row it produces going forward)
but the 240,078 rows already sitting in the curated lake (2021-07..2026-07,
venue=kalshi) predate the fix and have no ``category`` column at all. This script
is the one-time backfill for those existing rows.

Why a listing-only pass, never a re-fetch of market/trade data: ``category`` is a
property of the SERIES (Kalshi's own taxonomy), not of any individual market or
trade. A ``series_ticker -> category`` map, built by listing every series in each
of Kalshi's Politics/Economics/Financials categories — the exact set
``production.data.loaders.kalshi_history.CATEGORIES`` recognizes, itself mirroring
``production.data.loaders.kalshi.KalshiLoader``'s own default ``categories`` so the
two loaders' ``category`` columns agree — is all that's needed to stamp EVERY
existing row correctly. Each category listing is a single, unauthenticated,
cursor-terminated call (~2k series for Politics, probed live) — re-pulling 240k
rows of market/trade data to populate a column fully determined by
``series_ticker`` alone would be enormously wasteful for zero additional
information.

Unmatched series (found under none of the three listed categories — e.g. a
Kalshi sports/entertainment series that slipped into the historical backfill's
category-scoped Politics pull via a stale/renamed ticker) stamp ``"unknown"``:
fail-closed, the same posture ``political_favorite_tilt`` already assumes for
anything other than ``"politics"``.

Existing rows are re-stamped in place: read via ``lake.read_curated``, ``category``
added, ``ingested_at`` bumped to the remediation run's own timestamp (so
``lake.write_curated``'s vintage dedupe — sort by ``ingested_at``, keep the last
row per ``(obs_date, instrument_id, available_from)`` — deterministically supersedes
the un-stamped rows already on disk; a tie on ``ingested_at`` between the old and
new copy would leave the winner undefined), then written back through the normal
curated-write path. Raw zone and every other curated dataset are untouched.

Usage
-----
    python scripts/remediate_events_hist_category.py --lake-root data          # apply
    python scripts/remediate_events_hist_category.py --lake-root data --dry-run # report only
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone

import pandas as pd

from production.core.lake import Lake, LakeError
from production.data.loaders.kalshi_history import CATEGORIES, KalshiHistoryLoader

_DATASET = "event_markets_hist"
_ASSET_CLASS = "events"


def _build_category_map(pause_s: float = 0.2):
    """``series_ticker -> category`` (lowercase), one Kalshi series-listing call per
    entry in :data:`CATEGORIES`.

    Reuses ``KalshiHistoryLoader._enumerate_category`` — the exact pagination /
    failure-degrade logic the forward-fix loader itself uses against this same
    endpoint — instead of reimplementing it, so the remediation and the loader can
    never silently drift apart on what "a category listing" means. The loader
    instance here is throwaway: no lake is touched and ``fetch``/``transform``/
    ``run`` are never called, only the one private listing helper (which already
    paces itself with ``pause_s`` between pages via the shared ``_paginate``).

    Returns ``(ticker_to_category, per_category_counts, failed_categories, warnings)``.
    """
    import requests

    loader = KalshiHistoryLoader(pause_s=pause_s)
    session = requests.Session()
    ticker_to_category: dict[str, str] = {}
    per_category_counts: dict[str, int] = {}
    failed: list[str] = []
    for category in CATEGORIES:
        tickers = loader._enumerate_category(session, category)
        per_category_counts[category.lower()] = len(tickers)
        if not tickers:
            failed.append(category)
        for t in tickers:
            ticker_to_category.setdefault(t, category.lower())
    return ticker_to_category, per_category_counts, failed, list(loader.warnings)


def run(lake_root: str, dry_run: bool = False, pause_s: float = 0.2) -> dict:
    lake = Lake(lake_root)
    try:
        raw = lake.read_curated(_DATASET, _ASSET_CLASS)
    except LakeError as exc:
        raise SystemExit(f"FATAL: no curated {_DATASET!r} under {lake_root!r}: {exc}")
    if raw.empty:
        raise SystemExit(f"FATAL: curated {_DATASET!r} under {lake_root!r} is empty")
    if "category" in raw.columns and raw["category"].notna().all():
        print(f"{_DATASET} already carries a fully-populated 'category' column — "
              "nothing to do.")
        return {"rows_stamped": 0, "already_stamped": True}

    print(f"read {len(raw)} rows from curated {_DATASET!r} "
          f"({raw['series_ticker'].nunique()} distinct series).")

    ticker_to_category, per_category_counts, failed_categories, warnings = \
        _build_category_map(pause_s)
    print(f"category listing: {per_category_counts} "
          f"({sum(per_category_counts.values())} series total)"
          + (f" — FAILED: {failed_categories}" if failed_categories else ""))
    for w in warnings:
        print(f"    warning: {w}")

    stamped = raw.copy()
    stamped["category"] = (stamped["series_ticker"].map(ticker_to_category)
                            .fillna("unknown").astype(object))

    row_counts = stamped["category"].value_counts().to_dict()
    print("row counts by category:")
    for cat, n in sorted(row_counts.items(), key=lambda kv: -kv[1]):
        print(f"    {cat:<12} {n:>7} rows")

    unmatched_series = sorted(
        set(stamped.loc[stamped["category"] == "unknown", "series_ticker"]))

    written: list = []
    if not dry_run:
        ingest_ts = pd.Timestamp.now(tz="UTC")
        stamped["ingested_at"] = ingest_ts
        written = lake.write_curated(stamped, _DATASET, _ASSET_CLASS)
        print(f"\nrewrote {len(written)} year-partition(s).")

    record = {
        "remediation": "event_markets_hist category backfill",
        "why": ("political_favorite_tilt fail-closes without a category=='politics' "
                "column (production/events/signals.py); kalshi_history.py's backfill "
                "loader never stamped one, unlike the live-snapshot kalshi.py loader, "
                "so the events sleeve traded zero in the headline backtest "
                "(commit d665a3b). One-time backfill of the 240,078 pre-fix rows; "
                "production/data/loaders/kalshi_history.py now stamps category at "
                "ingest time for every new pull going forward."),
        "category_source": ("Kalshi series listing endpoint (GET .../series?category=), "
                            "one call per category in "
                            "production.data.loaders.kalshi_history.CATEGORIES "
                            f"({', '.join(CATEGORIES)}) — the identical set "
                            "kalshi.py's KalshiLoader uses for the live event_markets "
                            "panel's own category column."),
        "series_per_category": per_category_counts,
        "category_listing_failed": failed_categories,
        "category_listing_warnings": warnings,
        "rows_total": int(len(stamped)),
        "row_counts_by_category": {k: int(v) for k, v in row_counts.items()},
        "series_matched": int(len(ticker_to_category)),
        "series_unmatched": len(unmatched_series),
        "unmatched_series_preview": unmatched_series[:50],
        "partitions_rewritten": [str(p) for p in written],
        "dry_run": dry_run,
        "performed_at": datetime.now(timezone.utc).isoformat(),
    }
    if not dry_run:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        out = lake.write_audit(record, f"event_markets_hist_category_remediation_{stamp}")
        print(f"audit written: {out}")
    else:
        print("\n[dry-run] no partitions rewritten, no audit emitted.")
    return record


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="remediate_events_hist_category.py",
        description="Stamp `category` onto existing curated event_markets_hist rows.")
    p.add_argument("--lake-root", default="data", help="lake root dir (default: data)")
    p.add_argument("--dry-run", action="store_true", help="report only; do not rewrite")
    p.add_argument("--pause-s", type=float, default=0.2,
                   help="pacing between category-listing calls (default: 0.2s)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    run(args.lake_root, dry_run=args.dry_run, pause_s=args.pause_s)
    return 0


if __name__ == "__main__":
    sys.exit(main())
