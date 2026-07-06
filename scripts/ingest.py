"""CLI to run data loaders into the parquet lake.

Usage:
    uv run python scripts/ingest.py --dataset prices --sleeve equity \
        --start 2015-01-01 --end 2024-12-31

`--dataset universe` (re)builds the instrument master + membership tables from the
reference package; every other dataset wires the matching loader to the default Lake
and prints the resulting IngestResult summary.

Loaders resolve vendor symbols against the instrument master, so most datasets need
the reference tables to exist first — run `--dataset universe` once before the others.
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd

from production.core.lake import Lake


def _load_instruments(lake: Lake) -> "pd.DataFrame | None":
    """Read the instrument master from the reference zone if it has been built."""
    try:
        return lake.read_reference("instruments")
    except Exception:
        return None


def _build_universe(lake: Lake) -> None:
    """Build instrument master + membership via the reference package (agent B)."""
    try:
        from production.reference import instruments as ref_instruments
        from production.reference import universe as ref_universe
    except ImportError as exc:
        print(f"ERROR: reference package not available yet ({exc}).\n"
              "The universe builder lives in production/reference/ (built separately).",
              file=sys.stderr)
        raise SystemExit(2)

    master = ref_instruments.build_instrument_master(lake=lake)
    lake.write_reference(master, "instruments")
    print(f"instrument master: {len(master)} rows -> reference/instruments.parquet")
    for sleeve in ("equity", "crypto", "fx_etf", "commodity_etf"):
        try:
            mem = ref_universe.static_membership(sleeve)
            lake.write_reference(mem, f"membership_{sleeve}")
            print(f"membership[{sleeve}]: {len(mem)} rows")
        except Exception as exc:  # sleeve without a static builder
            print(f"membership[{sleeve}]: skipped ({exc})")


def _sleeve_symbols(sleeve: str) -> list[str]:
    """Vendor symbols for a sleeve, straight from universe.yaml."""
    from production.core.config import universe_config

    cfg = universe_config()["sleeves"].get(sleeve, {})
    syms = cfg.get("symbols", [])
    if isinstance(syms, dict):        # fx_etf / commodity_etf map symbol -> underlying
        return list(syms.keys())
    return list(syms)                 # crypto is a plain list


def _make_loader(dataset: str, sleeve: str, lake: Lake, instruments):
    """Construct the loader for a dataset (+ sleeve where symbol lists differ)."""
    if dataset == "prices":
        if sleeve == "crypto":
            from production.data.loaders.ccxt_prices import CcxtPricesLoader

            pairs = [f"{s}/USD" for s in _sleeve_symbols("crypto")]
            return CcxtPricesLoader(lake, instruments, symbols=pairs)
        from production.data.loaders.yfinance_prices import YFinancePricesLoader

        return YFinancePricesLoader(lake, instruments, symbols=_sleeve_symbols(sleeve))
    if dataset == "funding":
        from production.data.loaders.ccxt_funding import CcxtFundingLoader

        pairs = [f"{s}/USDT:USDT" for s in _sleeve_symbols("crypto")]
        return CcxtFundingLoader(lake, instruments, symbols=pairs)
    if dataset == "fx":
        from production.data.loaders.frankfurter_fx import FrankfurterFxLoader

        return FrankfurterFxLoader(lake, instruments)
    if dataset == "macro":
        from production.data.loaders.fred_alfred import FredAlfredLoader

        return FredAlfredLoader(lake, instruments)
    if dataset == "french":
        from production.data.loaders.ken_french import KenFrenchLoader

        return KenFrenchLoader(lake, instruments)
    if dataset == "cot":
        from production.data.loaders.cftc_cot import CftcCotLoader

        return CftcCotLoader(lake, instruments)
    raise ValueError(f"no loader for dataset {dataset!r}")


DATASETS = ["prices", "funding", "fx", "macro", "french", "cot", "universe", "all"]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Ingest a dataset into the parquet lake.")
    p.add_argument("--dataset", required=True, choices=DATASETS,
                   help="dataset to ingest (or 'universe' to build reference tables)")
    p.add_argument("--sleeve", default="equity",
                   help="sleeve for symbol-scoped datasets (prices)")
    p.add_argument("--start", default="2010-01-01", help="ISO start date")
    p.add_argument("--end", default=str(pd.Timestamp.now().date()), help="ISO end date")
    p.add_argument("--full", action="store_true",
                   help="full (non-incremental) re-pull, ignoring the watermark")
    p.add_argument("--lake-root", default=None, help="override lake root directory")
    args = p.parse_args(argv)

    lake = Lake(args.lake_root)

    if args.dataset == "universe":
        _build_universe(lake)
        return 0

    instruments = _load_instruments(lake)
    if instruments is None:
        print("WARNING: no instrument master found — run `--dataset universe` first; "
              "symbol resolution will return no rows.", file=sys.stderr)

    todo = (["prices", "funding", "fx", "macro", "french", "cot"]
            if args.dataset == "all" else [args.dataset])
    incremental = not args.full
    rc = 0
    for ds in todo:
        loader = _make_loader(ds, args.sleeve, lake, instruments)
        try:
            res = loader.run(args.start, args.end, incremental=incremental)
            print(f"[{ds}] vendor={res.vendor} rows={res.rows} "
                  f"start={res.start.date()} end={res.end.date()} "
                  f"warnings={res.audit.get('warnings')}")
        except Exception as exc:  # keep going across datasets in an 'all' run
            print(f"[{ds}] FAILED: {exc}", file=sys.stderr)
            rc = 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
