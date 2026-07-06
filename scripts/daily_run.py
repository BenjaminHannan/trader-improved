#!/usr/bin/env python
"""Daily execution runner — turn the latest target book into broker orders.

Usage
-----
    # dry run (DEFAULT): print the order list, submit NOTHING, touch no network
    python scripts/daily_run.py --lake-root data
    python scripts/daily_run.py --synthetic

    # go live: diff against real Alpaca positions and actually submit
    python scripts/daily_run.py --lake-root data --live

Flow: load the signal bundle + instrument master, recompute the most recent grid date's
target weights via the SAME engine the backtest validates
(``production.backtest.engine.latest_target_weights``), diff those against current Alpaca
positions, size the deltas into orders, print them, and — only with ``--live`` — submit.

``--dry-run`` is the default and is safe: it never constructs authenticated requests and
never needs a key. ``--live`` is the sole path that reads credentials and POSTs orders.
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd

from production.backtest.engine import latest_target_weights
from production.core.config import CONFIG_DIR, REPO_ROOT, backtest_config

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from production.core.lake import Lake, LakeError
from production.execution.alpaca_paper import AlpacaPaperClient, ExecutionError
from production.execution.orders import target_weights_to_orders
from production.reference.instruments import build_instrument_master, vendor_symbol
from production.signals.base import sleeve_from_id

# Reuse the backtest runner's lake/synthetic loaders verbatim — one bundle-building path,
# tested once. (Both scripts live under the repo root which is on sys.path.)
from scripts.run_backtest import (_instrument_map, _load_lake_bundle, _sector_map,
                                  _synthetic_bundle)

_DEFAULT_EQUITY = 100_000.0


def _latest_prices(prices: pd.DataFrame, ids) -> pd.Series:
    """Most recent close per instrument (the decision mark for order sizing)."""
    sub = prices[prices["instrument_id"].isin(list(ids))]
    if sub.empty:
        return pd.Series(dtype=float)
    last = (sub.sort_values("obs_date").groupby("instrument_id")["close"].last())
    return last.astype(float)


def _current_weights(client: AlpacaPaperClient | None, master: pd.DataFrame,
                     target_ids, equity_usd: float) -> pd.Series:
    """Current holdings as fraction-of-equity, keyed by internal instrument_id.

    Only reachable on the live path (a client + credentials). In dry-run we treat the book
    as flat (all-zero current weights) so the printed orders show the full target build.
    """
    if client is None:
        return pd.Series(0.0, index=list(target_ids))
    positions = client.positions()
    if positions.empty:
        return pd.Series(0.0, index=list(target_ids))
    # Map Alpaca symbols back to internal ids via the master (alpaca vendor symbol).
    sym_to_id = {vendor_symbol(master, iid, "alpaca"): iid
                 for iid in master["instrument_id"]}
    w = pd.Series(0.0, index=list(target_ids))
    for p in positions.itertuples(index=False):
        iid = sym_to_id.get(p.symbol)
        if iid is not None and equity_usd > 0:
            w.loc[iid] = float(p.market_value) / float(equity_usd)
    return w


def _print_orders(orders: pd.DataFrame, target: pd.Series, equity_usd: float,
                  dry_run: bool) -> None:
    mode = "DRY-RUN (no orders submitted)" if dry_run else "LIVE"
    print("\n" + "=" * 60)
    print(f"  daily run — {mode}")
    print(f"  equity_usd {equity_usd:,.0f}   target names {int((target != 0).sum())}")
    print("=" * 60)
    if orders.empty:
        print("  no orders (book already at target within min-notional).")
        return
    print(f"  {'instrument_id':<28}{'side':<6}{'qty':>14}{'notional_usd':>16}")
    for o in orders.itertuples(index=False):
        print(f"  {o.instrument_id:<28}{o.side:<6}{o.qty:>14.6f}{o.notional_usd:>16,.2f}")
    print("-" * 60)
    print(f"  {len(orders)} orders   gross notional "
          f"{orders['notional_usd'].sum():,.2f}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="daily_run.py",
                                description="Daily target-book execution runner.")
    p.add_argument("--config", default=str(CONFIG_DIR / "backtest.yaml"),
                   help="backtest config yaml (default: configs/backtest.yaml)")
    p.add_argument("--lake-root", default="data", help="lake root dir (default: data)")
    p.add_argument("--start", default=None, help="bundle start (obs_date >=)")
    p.add_argument("--end", default=None, help="bundle end (obs_date <=)")
    p.add_argument("--synthetic", action="store_true",
                   help="run on a synthesized GBM bundle (no lake, no network)")
    p.add_argument("--equity", type=float, default=None,
                   help="account equity in USD (default: from Alpaca account when --live, "
                        f"else {_DEFAULT_EQUITY:,.0f})")
    p.add_argument("--min-order-usd", type=float, default=25.0,
                   help="drop orders below this notional (default: 25)")
    # Mutually exclusive dry-run / live; dry-run is the default.
    g = p.add_mutually_exclusive_group()
    g.add_argument("--dry-run", dest="dry_run", action="store_true", default=True,
                   help="print orders, submit nothing (DEFAULT)")
    g.add_argument("--live", dest="dry_run", action="store_false",
                   help="actually submit orders to Alpaca paper")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    cfg = backtest_config()

    if args.synthetic:
        start = args.start or "2019-01-01"
        end = args.end or "2020-06-30"
        print(f"synthetic run: warmup-synthesized bundle, ~{start} .. {end}")
        data, instruments, sectors = _synthetic_bundle(start, end)
        master = build_instrument_master()
    else:
        lake = Lake(args.lake_root)
        data = _load_lake_bundle(lake, args.start, args.end)
        if "prices" not in data:
            print("FATAL: no curated 'prices' — nothing to trade", file=sys.stderr)
            return 2
        instruments = _instrument_map(lake, data["prices"])
        sectors = _sector_map(lake, data["prices"])
        try:
            master = lake.read_reference("instruments")
        except LakeError:
            master = build_instrument_master()

    # --- latest target book via the validated engine path ---
    target = latest_target_weights(data, instruments, cfg=cfg, sectors=sectors)
    if target.empty:
        print("no target weights produced — check data coverage / warmup", file=sys.stderr)
        return 1

    # --- broker client only on the live path (dry-run never authenticates) ---
    client: AlpacaPaperClient | None = None
    equity_usd = args.equity if args.equity is not None else _DEFAULT_EQUITY
    if not args.dry_run:
        try:
            client = AlpacaPaperClient()
            if args.equity is None:
                acct = client.account()
                equity_usd = float(acct.get("equity", equity_usd))
        except ExecutionError as exc:
            print(f"FATAL: cannot go live — {exc}", file=sys.stderr)
            return 3

    w_current = _current_weights(client, master, target.index, equity_usd)
    prices = _latest_prices(data["prices"], target.index)
    orders = target_weights_to_orders(target, w_current, equity_usd, prices,
                                      min_order_usd=args.min_order_usd)
    _print_orders(orders, target, equity_usd, args.dry_run)

    if not args.dry_run and client is not None and not orders.empty:
        result = client.submit_orders(orders, master, dry_run=False)
        submitted = int((result["status"] != "no_alpaca_symbol").sum())
        print(f"\nsubmitted {submitted}/{len(result)} orders to Alpaca paper")
    return 0


if __name__ == "__main__":
    sys.exit(main())
