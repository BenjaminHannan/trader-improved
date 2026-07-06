#!/usr/bin/env python
"""Run the walk-forward backtest end-to-end and write a report.

Usage
-----
    python scripts/run_backtest.py --config configs/backtest.yaml --lake-root data \
        --start 2016-01-01 --end 2021-12-31
    python scripts/run_backtest.py --synthetic --start 2019-01-01 --end 2020-06-30

The lake path loads curated datasets into the signal bundle (``prices`` required;
``funding``/``macro``/``cot`` optional with a warning), derives the instrument→sleeve map
from the ``instruments`` reference table (falling back to the id-prefix rule), runs the
engine, writes ``reports/backtest_<UTC>.json`` + ``.txt`` and prints the headline table.

The ``--synthetic`` flag is the no-lake smoke path: it builds the conftest GBM bundle
(with ~3y of warmup history synthesized before ``--start``) and runs the identical engine.
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd

from production.backtest.engine import run_backtest
from production.backtest.report import write_report
from production.core.config import CONFIG_DIR, REPO_ROOT, backtest_config

# Ensure the repo root is importable so `--synthetic` can pull the conftest makers even
# when the script is invoked directly (not under pytest, which adds it automatically).
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from production.core.lake import Lake, LakeError
from production.signals.base import sleeve_from_id

_DATASETS = ("prices", "funding", "macro", "cot")


def _load_lake_bundle(lake: Lake, start, end) -> dict:
    bundle: dict = {}
    for ds in _DATASETS:
        try:
            df = lake.read_curated(ds, start=start, end=end)
        except LakeError:
            df = None
        if df is not None and not df.empty:
            bundle[ds] = df
        elif ds == "prices":
            print(f"FATAL: no curated 'prices' in lake at {lake.root}", file=sys.stderr)
        else:
            print(f"  note: optional dataset '{ds}' absent — continuing without it")
    return bundle


def _instrument_map(lake: Lake, prices: pd.DataFrame) -> pd.Series:
    """instrument_id -> sleeve from the reference master, else the id-prefix fallback."""
    present = set(prices["instrument_id"].unique())
    try:
        master = lake.read_reference("instruments")
        if {"instrument_id", "sleeve"}.issubset(master.columns):
            m = master.set_index("instrument_id")["sleeve"]
            m = m[m.index.isin(present)]
            if not m.empty:
                return m
    except LakeError:
        pass
    print("  note: instrument master unavailable — using id-prefix sleeve mapping")
    return pd.Series({i: sleeve_from_id(i) for i in sorted(present)})


def _sector_map(lake: Lake, prices: pd.DataFrame) -> pd.Series | None:
    """instrument_id -> sector (current-snapshot GICS) from the master, or None.

    Read from the instrument master's ``sector`` column when the reference table exists;
    guarded so a missing table / column / lake simply disables sector threading.
    """
    present = set(prices["instrument_id"].unique())
    try:
        master = lake.read_reference("instruments")
    except LakeError:
        return None
    if not {"instrument_id", "sector"}.issubset(master.columns):
        return None
    m = master.set_index("instrument_id")["sector"]
    m = m[m.index.isin(present)].dropna()
    return m if not m.empty else None


def _synthetic_bundle(start, end):
    """Build a GBM bundle with warmup history before ``start`` (no lake needed)."""
    from tests.conftest import make_cot, make_funding, make_gbm_prices, make_macro

    data_start = (pd.Timestamp(start) - pd.DateOffset(years=3, months=3)).strftime("%Y-%m-%d")
    data_end = pd.Timestamp(end).strftime("%Y-%m-%d")

    # 10 equity ids so the risk model stays over-determined once the 3 sector dummies join
    # the 4 style factors (7 exposure columns < 10 names).
    eq = {f"EQ:SYN{i:02d}:2000-01-03": "equity" for i in range(10)}
    cr = {f"CR:{s}:2017-01-01": "crypto" for s in ["BTC", "ETH", "SOL", "LTC"]}
    fx = {f"FX:{s}:2007-01-03": "fx_etf" for s in ["FXE", "FXY", "FXB"]}
    co = {f"CO:{s}:2006-01-03": "commodity_etf" for s in ["GLD", "USO", "SLV"]}
    instruments = {**eq, **cr, **fx, **co}

    crypto_ids = [i for i, s in instruments.items() if s == "crypto"]
    cot_ids = [i for i, s in instruments.items() if s in ("fx_etf", "commodity_etf")]
    # Deterministic fake sectors (round-robin over 3 labels) for the equity ids, so the
    # sector-threading path is actually exercised on the no-lake smoke run.
    eq_ids = sorted(i for i, s in instruments.items() if s == "equity")
    labels = ("sector_A", "sector_B", "sector_C")
    sectors = pd.Series({iid: labels[k % 3] for k, iid in enumerate(eq_ids)})
    bundle = {
        "prices": make_gbm_prices(instruments, start=data_start, end=data_end),
        "funding": make_funding(crypto_ids, start=data_start, end=data_end),
        "macro": make_macro({"DGS3MO_US": 2.0, "RATE_EU": 0.5, "RATE_JP": -0.1,
                             "RATE_GB": 1.0, "BAMLH0A0HYM2": 4.0, "VIXCLS": 18.0},
                            start=data_start, end=data_end),
        "cot": make_cot(cot_ids, start=data_start, end=data_end),
    }
    return bundle, pd.Series(instruments), sectors


def _print_headline(report: dict) -> None:
    h = report["headline"]
    print("\n" + "=" * 52)
    print(f"  window {h['start']} .. {h['end']}  ({h['n_days']} days)")
    print(f"  net Sharpe   {_f(h['net_sharpe'])}   gross {_f(h['gross_sharpe'])}")
    print(f"  ann return   {_f(h['ann_return_net'])}   ann vol {_f(h['ann_vol_net'])}")
    print(f"  max DD       {_f(h['max_drawdown'])}   hit {_f(h['hit_rate'])}")
    print(f"  PSR(>0)      {_f(h['probabilistic_sharpe'])}   "
          f"deflated {_f(h['deflated_sharpe'])} (n_trials={h['n_trials']})")
    print(f"  Sharpe 95%CI [{_f(h['sharpe_ci95'][0])}, {_f(h['sharpe_ci95'][1])}]")
    if not report["caveats"]["gate_applied"]:
        print("  WARNING: gate NOT applied — trading candidate factors")
    print("=" * 52)


def _f(x) -> str:
    try:
        v = float(x)
        return "n/a" if v != v else f"{v:+.3f}"
    except (TypeError, ValueError):
        return "n/a"


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="run_backtest.py",
                                description="Walk-forward backtest runner.")
    p.add_argument("--config", default=str(CONFIG_DIR / "backtest.yaml"),
                   help="backtest config yaml (default: configs/backtest.yaml)")
    p.add_argument("--lake-root", default="data", help="lake root dir (default: data)")
    p.add_argument("--start", default=None, help="backtest start (obs_date >=)")
    p.add_argument("--end", default=None, help="backtest end (obs_date <=)")
    p.add_argument("--synthetic", action="store_true",
                   help="run on a synthesized GBM bundle without any lake (smoke path)")
    p.add_argument("--report-dir", default="reports", help="output dir (default: reports/)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    cfg = backtest_config()  # validated config; --config path is echoed for provenance

    if args.synthetic:
        start = args.start or "2019-01-01"
        end = args.end or "2020-06-30"
        print(f"synthetic run: warmup-synthesized bundle, trading ~{start} .. {end}")
        data, instruments, sectors = _synthetic_bundle(start, end)
    else:
        lake = Lake(args.lake_root)
        data = _load_lake_bundle(lake, args.start, args.end)
        if "prices" not in data:
            return 2
        instruments = _instrument_map(lake, data["prices"])
        sectors = _sector_map(lake, data["prices"])

    result = run_backtest(data, instruments, cfg=cfg, sectors=sectors)
    path = write_report(result.report, out_dir=args.report_dir)
    _print_headline(result.report)
    print(f"\nreport written: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
