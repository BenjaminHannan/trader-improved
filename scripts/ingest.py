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
    if dataset == "basis":
        from production.data.loaders.ccxt_perp_basis import CcxtPerpBasisLoader

        return CcxtPerpBasisLoader(lake, instruments, symbols=_sleeve_symbols("crypto"))
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
    if dataset == "fundamentals":
        from production.data.loaders.edgar import EdgarFundamentalsLoader

        return EdgarFundamentalsLoader(lake, instruments,
                                       symbols=_sleeve_symbols("equity"))
    raise ValueError(f"no loader for dataset {dataset!r}")


DATASETS = ["prices", "funding", "basis", "fx", "macro", "french", "cot",
            "fundamentals", "universe", "all"]

# Curated `source` values that identify the two independent price feeds we cross-check.
_PRIMARY_SOURCES = ("yfinance",)
_SECONDARY_SOURCES = ("stooq",)


def _source_matches(series: "pd.Series", markers: tuple[str, ...]) -> "pd.Series":
    """Rows whose `source` contains any marker (loaders may stamp 'yfinance:prices')."""
    src = series.astype(str)
    hit = pd.Series(False, index=series.index)
    for m in markers:
        hit |= src.str.contains(m, case=False, na=False)
    return hit


def _run_cross_check(lake: Lake, sleeve: str) -> int:
    """Cross-check the two price feeds in the curated lake and write an audit.

    Splits curated 'prices' rows by their `source` provenance (yfinance vs stooq),
    compares them in return space, prints the summary + quarantine list, and persists
    the report to the audit zone. Returns 0 on success, non-zero when the data needed to
    run is absent.
    """
    from production.data.cross_check import (
        cross_vendor_report, quarantine_list, write_cross_check_audit,
    )

    try:
        prices = lake.read_curated("prices")
    except Exception as exc:
        print(f"[cross-check] no curated prices to check ({exc}).", file=sys.stderr)
        return 1
    if prices.empty:
        print("[cross-check] curated prices are empty — nothing to cross-check.",
              file=sys.stderr)
        return 1

    primary = prices[_source_matches(prices["source"], _PRIMARY_SOURCES)]
    secondary = prices[_source_matches(prices["source"], _SECONDARY_SOURCES)]
    if primary.empty or secondary.empty:
        have = sorted(prices["source"].astype(str).unique())
        print("[cross-check] need both feeds; a vendor's rows are absent "
              f"(sources present: {have}). Ingest both yfinance and stooq first.",
              file=sys.stderr)
        return 1

    report = cross_vendor_report(primary, secondary)
    out = write_cross_check_audit(report, lake)
    s = report["summary"]
    print(f"[cross-check] checked={s['instruments_checked']} "
          f"flagged={s['instruments_flagged']} worst={s['worst_offender']} "
          f"-> {out}")
    if report["insufficient_overlap"]:
        print(f"[cross-check] insufficient overlap (skipped): "
              f"{sorted(report['insufficient_overlap'])}")
    quarantine = quarantine_list(report)
    if quarantine:
        print(f"[cross-check] QUARANTINE ({len(quarantine)}): {quarantine}")
    else:
        print("[cross-check] quarantine list: none")
    return 0


def _make_stage2_loaders(lake: Lake, instruments):
    """Construct every Stage-2 macro/sentiment loader against the default lake.

    These emit macro-style series (ADS, NAAIM, CBOE put/call, FINRA short interest,
    FRED nowcasts, and the optional manual AAII drop-in) and take no per-sleeve symbol
    scoping, so they are wired as a fixed batch behind `--stage 2`.
    """
    from production.data.loaders.stage2 import (
        AaiiManualLoader, AdsLoader, CboePutCallLoader, FinraShortInterestLoader,
        FredStage2Loader, NaaimLoader,
    )

    return [
        FredStage2Loader(lake, instruments),
        AdsLoader(lake, instruments),
        NaaimLoader(lake, instruments),
        CboePutCallLoader(lake, instruments),
        FinraShortInterestLoader(lake, instruments),
        AaiiManualLoader(lake, instruments),
    ]


def _run_stage2(lake: Lake, instruments, start, end, incremental: bool) -> int:
    """Run the Stage-2 loader batch, keeping going across failures (e.g. the AAII
    manual file being absent) and returning a non-zero code if any loader failed."""
    rc = 0
    for loader in _make_stage2_loaders(lake, instruments):
        name = type(loader).__name__
        try:
            res = loader.run(start, end, incremental=incremental)
            print(f"[{name}] vendor={res.vendor} rows={res.rows} "
                  f"start={res.start.date()} end={res.end.date()} "
                  f"warnings={res.audit.get('warnings')}")
        except Exception as exc:
            print(f"[{name}] FAILED: {exc}", file=sys.stderr)
            rc = 1
    return rc


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Ingest a dataset into the parquet lake.")
    p.add_argument("--dataset", default=None, choices=DATASETS,
                   help="dataset to ingest (or 'universe' to build reference tables); "
                        "not required when --stage 2 is given")
    p.add_argument("--stage", type=int, default=1, choices=[1, 2],
                   help="ingest stage: 1 = the --dataset loader (default); "
                        "2 = the Stage-2 macro/sentiment loader batch")
    p.add_argument("--sleeve", default="equity",
                   help="sleeve for symbol-scoped datasets (prices)")
    p.add_argument("--start", default="2010-01-01", help="ISO start date")
    p.add_argument("--end", default=str(pd.Timestamp.now().date()), help="ISO end date")
    p.add_argument("--full", action="store_true",
                   help="full (non-incremental) re-pull, ignoring the watermark")
    p.add_argument("--cross-check", action="store_true",
                   help="cross-check the two curated price feeds (yfinance vs stooq) in "
                        "return space, write an audit, and print any quarantine list")
    p.add_argument("--lake-root", default=None, help="override lake root directory")
    args = p.parse_args(argv)

    lake = Lake(args.lake_root)
    incremental = not args.full

    if args.stage == 2:
        instruments = _load_instruments(lake)
        return _run_stage2(lake, instruments, args.start, args.end, incremental)

    if args.dataset is None:
        if args.cross_check:  # cross-check can run standalone against the curated lake
            return _run_cross_check(lake, args.sleeve)
        p.error("--dataset is required unless --stage 2 or --cross-check is given")

    if args.dataset == "universe":
        _build_universe(lake)
        return 0

    instruments = _load_instruments(lake)
    if instruments is None:
        print("WARNING: no instrument master found — run `--dataset universe` first; "
              "symbol resolution will return no rows.", file=sys.stderr)

    todo = (["prices", "funding", "fx", "macro", "french", "cot"]
            if args.dataset == "all" else [args.dataset])
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

    if args.cross_check and "prices" in todo:  # cross-check the feeds just ingested
        rc |= _run_cross_check(lake, args.sleeve)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
