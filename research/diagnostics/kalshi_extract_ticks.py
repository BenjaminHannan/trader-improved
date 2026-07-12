"""One-time extraction: Kalshi raw ``event_markets_hist`` trade tapes -> tidy tick cache.

Part of the maker-fill simulation registered at wiki commit c9d47fd
("PRE-REGISTERED (maker-fill simulation...)"). The raw zone retains full
per-trade tick tapes inside each series payload's ``trades`` map — this script
flattens them into one tidy parquet cache that :mod:`kalshi_maker_fill_sim`
reads (see its ``TICK_CACHE_PATH`` constant).

Streams ONE payload row at a time via ``pyarrow`` (``batch_size=1``) so the
~23MB-per-payload JSON blobs across 499+ politics series plus the earlier
econ partitions are never all resident at once — only the flattened tidy rows
accumulate in memory (~6.1M raw trade rows across both partitions, verified
cheap to hold as a flat list of tuples before one final ``DataFrame`` build).

Output columns: market_ticker, created_time (UTC-naive timestamp), yes_price
(float), count (float), taker_side ("yes"/"no"), is_block_trade (bool),
trade_id. Deduped on trade_id where present (rows lacking a trade_id are kept
as-is — nothing to dedupe against). Both raw partitions are read and merged;
per the task spec, overlapping series across partitions are deduped together
by trade_id, not partition-by-partition.

Run once: ``uv run python research/diagnostics/kalshi_extract_ticks.py``
"""
from __future__ import annotations

import glob
import json
import time
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

# Cache lives outside the repo and outside data/ (gitignored lake) per the
# governing instructions — this is scratch derived from the raw lake, not a
# curated dataset and not a report artifact.
OUT_PATH = Path(
    r"C:\Users\benja\AppData\Local\Temp\claude\C--Users-benja-Downloads-trader-improved"
    r"\4da31de3-3dbd-44d1-a6d2-f10753b484ce\scratchpad\kalshi_ticks.parquet"
)

RAW_GLOB = "data/raw/vendor=kalshi/dataset=event_markets_hist/ingest_date=*/part*.parquet"


def _f(value, default: float = 0.0) -> float:
    """Parse the API's stringly-typed numerics (``"0.0300"``)."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def extract() -> pd.DataFrame:
    files = sorted(glob.glob(RAW_GLOB))
    if not files:
        raise FileNotFoundError(f"no raw partitions matched {RAW_GLOB!r}")

    rows: list[tuple] = []
    partition_counts: dict[str, dict[str, int]] = {}
    t0 = time.time()

    for path in files:
        part_label = Path(path).parent.name  # e.g. "ingest_date=2026-07-11"
        n_payloads = 0
        n_trades = 0
        pf = pq.ParquetFile(path)
        # batch_size=1 -> exactly one (__key__, payload) row materialized at a time.
        for batch in pf.iter_batches(batch_size=1, columns=["__key__", "payload"]):
            d = batch.to_pydict()
            raw_payload = d["payload"][0]
            payload = json.loads(raw_payload) if isinstance(raw_payload, str) else (raw_payload or {})
            trades_by_ticker = (payload or {}).get("trades", {}) or {}
            for market_ticker, trade_list in trades_by_ticker.items():
                for t in trade_list or []:
                    rows.append((
                        t.get("trade_id"),
                        market_ticker,
                        t.get("created_time"),
                        _f(t.get("yes_price_dollars"), default=float("nan")),
                        _f(t.get("count_fp"), default=float("nan")),
                        t.get("taker_side"),
                        bool(t.get("is_block_trade", False)),
                    ))
                    n_trades += 1
            n_payloads += 1
            # payload dict/JSON string go out of scope on the next loop iteration;
            # nothing beyond the tidy tuples above is retained.
        partition_counts[part_label] = {"payloads": n_payloads, "trades": n_trades}
        print(f"{part_label}: {n_payloads} payloads, {n_trades} raw trades "
              f"({time.time() - t0:.1f}s elapsed)")

    df = pd.DataFrame(rows, columns=[
        "trade_id", "market_ticker", "created_time", "yes_price", "count",
        "taker_side", "is_block_trade",
    ])
    n_raw_total = len(df)

    df["created_time"] = (pd.to_datetime(df["created_time"], utc=True, errors="coerce")
                             .dt.tz_localize(None))
    df = df.dropna(subset=["created_time", "yes_price"])

    has_id = df["trade_id"].notna() & (df["trade_id"].astype(str).str.len() > 0)
    with_id = df[has_id].drop_duplicates(subset=["trade_id"], keep="first")
    without_id = df[~has_id]
    df = pd.concat([with_id, without_id], ignore_index=True)
    df = df.sort_values(["market_ticker", "created_time"]).reset_index(drop=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)

    print(f"\nraw trades parsed (all partitions, pre-dedupe): {n_raw_total}")
    print(f"after dropna(created_time, yes_price) + trade_id dedupe: {len(df)}")
    print(f"  dropped by dropna: {n_raw_total - (len(with_id) + len(without_id))}")
    print(f"  dropped by trade_id dedupe: {has_id.sum() - len(with_id)}")
    print(f"  rows with no trade_id (kept, undeduped): {(~has_id).sum()}")
    for label, c in partition_counts.items():
        print(f"  {label}: payloads={c['payloads']} trades={c['trades']}")
    print(f"\nwrote cache: {OUT_PATH}  ({len(df)} rows, "
          f"{OUT_PATH.stat().st_size / 1e6:.1f} MB)")
    return df


if __name__ == "__main__":
    extract()
