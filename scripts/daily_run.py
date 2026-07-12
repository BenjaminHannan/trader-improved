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

from production.backtest.engine import latest_target_weights, run_backtest
from production.core.config import CONFIG_DIR, REPO_ROOT, backtest_config

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from production.core.lake import Lake, LakeError
from production.execution.alpaca_paper import AlpacaPaperClient, ExecutionError
from production.execution.orders import target_weights_to_orders
from production.execution.shortfall import implementation_shortfall
from production.execution.tca import (SHORTFALL_LOG_COLUMNS, append_shortfall_log,
                                      calibrate_overrides, write_overrides)
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


def _fetch_fills(client: AlpacaPaperClient, submitted: pd.DataFrame) -> pd.Series:
    """GET each submitted order's realized fill price, keyed by instrument_id.

    ``submitted`` is the ``client.submit_orders`` result filtered to rows that actually
    carry a ``broker_order_id`` (i.e. were really POSTed). Raises :class:`ExecutionError`
    on any broker-read failure — the caller (:func:`_accumulate_shortfall`) treats that as
    a warning, never a run-blocking failure: the orders are already live at the broker,
    so failing to read them back must not fail the run.
    """
    fills: dict[str, float] = {}
    for o in submitted.itertuples(index=False):
        order = client.get_order(o.broker_order_id)
        fp = (order or {}).get("filled_avg_price")
        fills[o.instrument_id] = float(fp) if fp is not None else float("nan")
    return pd.Series(fills, dtype=float)


def _accumulate_shortfall(client: AlpacaPaperClient, lake: Lake, submit_result: pd.DataFrame,
                          decision_prices: pd.Series, run_id: pd.Timestamp) -> None:
    """Backlog #15: append realized --live implementation shortfall to the lake's
    ``shortfall_log`` reference table so ``--calibrate-tca`` has something to learn from.

    Only reachable from the --live path AFTER a real (non-dry-run) submission — --dry-run
    never calls this, so it never writes. A fill-fetch failure (broker read error) degrades
    to a printed warning and returns without writing: the orders already executed at the
    broker, so a shortfall-accounting hiccup must never fail an otherwise-successful run.
    Orders not yet filled (no ``filled_avg_price``) are skipped, not logged as zero/garbage.
    """
    submitted = submit_result[submit_result["broker_order_id"].notna()].reset_index(drop=True)
    if submitted.empty:
        return
    try:
        fill_prices = _fetch_fills(client, submitted)
    except ExecutionError as exc:
        print(f"  WARNING: could not fetch fills for shortfall accounting — {exc} "
              "(shortfall_log not updated this run)", file=sys.stderr)
        return

    shortfall_df, _ = implementation_shortfall(
        submitted[["instrument_id", "side", "qty"]], decision_prices, fill_prices)
    shortfall_df["qty"] = submitted["qty"]
    shortfall_df = shortfall_df[shortfall_df["fill_price"].notna()].copy()
    if shortfall_df.empty:
        print("  shortfall_log: no fills yet this run — nothing appended")
        return
    shortfall_df["filled_at"] = pd.Timestamp.utcnow()
    shortfall_df["run_id"] = run_id
    out = append_shortfall_log(shortfall_df[SHORTFALL_LOG_COLUMNS], lake)
    print(f"  shortfall_log: appended {len(shortfall_df)} fill(s) -> {out}")


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
    has_limit = "limit_price" in orders.columns
    header = f"  {'instrument_id':<28}{'side':<6}{'qty':>14}{'notional_usd':>16}"
    if has_limit:
        header += f"{'limit_price':>16}"
    print(header)
    for o in orders.itertuples(index=False):
        line = (f"  {o.instrument_id:<28}{o.side:<6}{o.qty:>14.6f}"
                f"{o.notional_usd:>16,.2f}")
        if has_limit:
            line += f"{o.limit_price:>16,.6f}"
        print(line)
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
    p.add_argument("--order-type", choices=["market", "limit"], default="limit",
                   help="order type (default: limit — passive placement captures spread "
                        "instead of paying it; unfilled day limits expire)")
    p.add_argument("--limit-offset-bps", type=float, default=5.0,
                   help="passive limit half-spread offset in bps (default: 5); "
                        "buys priced below / sells above the decision mark")
    p.add_argument("--calibrate-tca", action="store_true",
                   help="TCA feedback: read accumulated shortfall (lake reference "
                        "'shortfall_log' or --shortfall-path parquet), calibrate "
                        "per-instrument cost overrides, write the 'cost_overrides' table, "
                        "print a summary, and exit (submits nothing)")
    p.add_argument("--shortfall-path", default=None,
                   help="parquet of accumulated implementation shortfall for "
                        "--calibrate-tca (default: lake reference 'shortfall_log')")
    p.add_argument("--min-fills", type=int, default=20,
                   help="min fills per instrument before TCA calibrates it (default: 20)")
    p.add_argument("--tca-safety", type=float, default=1.25,
                   help="safety multiplier on realized median |shortfall| (default: 1.25)")
    # Mutually exclusive dry-run / live; dry-run is the default.
    g = p.add_mutually_exclusive_group()
    g.add_argument("--dry-run", dest="dry_run", action="store_true", default=True,
                   help="print orders, submit nothing (DEFAULT)")
    g.add_argument("--live", dest="dry_run", action="store_false",
                   help="actually submit orders to Alpaca paper")
    return p


def _calibrate_tca(args) -> int:
    """TCA feedback loop: realized shortfall -> per-instrument cost overrides -> lake table."""
    lake = Lake(args.lake_root)
    if args.shortfall_path is not None:
        shortfall = pd.read_parquet(args.shortfall_path)
        src = args.shortfall_path
    else:
        try:
            shortfall = lake.read_reference("shortfall_log")
            src = "lake reference 'shortfall_log'"
        except LakeError:
            print("FATAL: no shortfall to calibrate from — pass --shortfall-path or "
                  "accumulate a 'shortfall_log' reference table", file=sys.stderr)
            return 2

    overrides = calibrate_overrides(shortfall, min_fills=args.min_fills,
                                    safety=args.tca_safety)
    out = write_overrides(overrides, lake)

    print("\n" + "=" * 60)
    print("  TCA calibration — cost overrides from realized shortfall")
    print(f"  source {src}   fills {len(shortfall)}")
    print("=" * 60)
    if not overrides:
        print("  no instruments cleared the fill gate / exceeded the sleeve default.")
    else:
        print(f"  {'instrument_id':<28}{'half_spread_bps':>16}{'n_fills':>10}"
              f"{'median_|sf|_bps':>18}")
        for iid, o in sorted(overrides.items()):
            print(f"  {iid:<28}{o['half_spread_bps']:>16.2f}{o['n_fills']:>10}"
                  f"{o['median_abs_shortfall_bps']:>18.2f}")
    print("-" * 60)
    print(f"  wrote {len(overrides)} override(s) -> {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    if args.calibrate_tca:
        return _calibrate_tca(args)

    cfg = backtest_config()
    run_id = pd.Timestamp.utcnow()   # stamps every shortfall_log row this invocation appends
    # Always constructed — used for the side-channel reference tables (cost_overrides
    # read, shortfall_log append) regardless of whether the PRICE data below comes from
    # the lake or --synthetic. Cheap: no disk I/O happens until a read/write call.
    lake = Lake(args.lake_root)

    if args.synthetic:
        start = args.start or "2019-01-01"
        end = args.end or "2020-06-30"
        print(f"synthetic run: warmup-synthesized bundle, ~{start} .. {end}")
        data, instruments, sectors = _synthetic_bundle(start, end)
        master = build_instrument_master()
    else:
        events_enabled = bool((cfg.get("events") or {}).get("enabled", True))
        data = _load_lake_bundle(lake, args.start, args.end, events_enabled=events_enabled)
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
    # backlog #14 follow-up: thread the lake's TCA-calibrated cost_overrides table into
    # the SAME CostModel construction run_backtest already performs for
    # scripts/run_backtest.py (production/backtest/engine.py:_load_cost_overrides /
    # run_backtest(lake=...)) — previously only the backtest path consumed it, not the
    # live runner. An absent/never-calibrated table reproduces lake=None behavior
    # bit-identically (read_overrides -> {}); an active table fires the identical loud
    # UserWarning ("cost overrides ACTIVE: ..."), which we also surface on stdout here so
    # a live run never silently changes cost regime (CLAUDE.md). Running run_backtest
    # directly and handing the result to latest_target_weights (its `result=` parameter)
    # avoids a second walk-forward pass — no production/backtest/engine.py changes needed.
    bt_result = run_backtest(data, instruments, cfg=cfg, sectors=sectors, lake=lake)
    for w in bt_result.report["caveats"]["warnings"]:
        print(f"  WARNING: {w}")
    target = latest_target_weights(data, instruments, cfg=cfg, result=bt_result,
                                   sectors=sectors)
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
                                      min_order_usd=args.min_order_usd,
                                      order_type=args.order_type,
                                      limit_offset_bps=args.limit_offset_bps)
    _print_orders(orders, target, equity_usd, args.dry_run)

    if not args.dry_run and client is not None and not orders.empty:
        # backlog #15: --live fills accumulate into the lake's shortfall_log reference
        # table (production/execution/tca.py:append_shortfall_log), feeding
        # --calibrate-tca. --dry-run never reaches this branch, so it writes nothing.
        submit_result = client.submit_orders(orders, master, dry_run=False)
        submitted = int((submit_result["status"] != "no_alpaca_symbol").sum())
        print(f"\nsubmitted {submitted}/{len(submit_result)} orders to Alpaca paper")
        _accumulate_shortfall(client, lake, submit_result, prices, run_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())
