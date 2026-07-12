#!/usr/bin/env python
"""Run ``production/risk/model.py:validate_against_french`` against the real lake.

This is the TIE-incident detector: it correlates our internally ESTIMATED equity
factor returns (production/risk/factor_returns.py) against Ken French's canonical
academic benchmarks. A corrupt-vendor equity series (wrong-entity ticker reuse,
a bad split adjustment, ...) shows up as a collapsed correlation long before any
backtest number looks obviously wrong -- see research/wiki/log.md 2026-07-07 (the
equal-weight-market-vs-Mkt-RF correlation fell to 0.056 pre-remediation, 0.92
after, on the reference machine). It has never run against THIS machine's lake
(rebuilt fresh 2026-07-10).

Wiring gap this script closes (wiki log open item: "validate_against_french
series-id + monthly-input wiring"):

1. Series-id mismatch. ``validate_against_french``'s own factor->column mapping
   is ``{"market": "Mkt-RF", "momentum": "Mom"}`` -- vendor column names, i.e.
   Ken French's own header spelling. The curated ``french`` lake dataset stores
   the LOADER's internal ``series_id`` (``FF_MKT_RF``, ``FF_SMB``, ``FF_HML``,
   ``FF_RF``, ``FF_MOM`` -- see ``production/data/loaders/ken_french.py``'s
   ``COL_MAP``), not the vendor name. Passed straight through, every join in
   ``validate_against_french`` silently returns empty (no column match) rather
   than raising. ``wire_french_monthly`` below renames series_id -> vendor name
   via ``COL_MAP`` (the loader's own authoritative mapping -- not re-derived by
   guesswork) and raises loudly if an expected series_id is absent.
2. Frequency mismatch. The curated ``french`` dataset is DAILY (one row per
   trading day); ``validate_against_french``'s ``french_monthly`` parameter name
   promises monthly data and its internals do NOT compound the benchmark side
   (only the estimated ``factor_returns`` argument is compounded internally, via
   ``groupby(...).prod()-1``). Handing it raw daily rows keyed by a many-to-one
   Period("M") index silently multiplies rows in the inner join instead of
   raising. ``wire_french_monthly`` compounds the daily french series to monthly
   ((1+r).groupby(month).prod()-1, the exact operation ``validate_against_french``
   already applies to the other side) BEFORE calling the function.

``model.py`` itself needed NO changes: given properly-monthly, correctly-named
inputs (see ``tests/test_risk_model.py::test_validate_against_french_synthetic``,
which already exercises it that way), the function's compounding/correlation
logic is correct. Every fix here is call-site wiring, per the wiki note.

Size vs SMB is a documented artifact, not a defect: our equity universe skews
toward vendor-covered large/liquid names, so it is EXPECTED to be negatively
correlated with Ken French's small-minus-big series (~-0.27 on the reference
machine post-remediation). It is reported for visibility but never gates
PASS/FAIL, and it is computed here at the call site (not via
``validate_against_french``, whose fixed mapping only knows market/momentum --
extending it would be a feature change, not the bug fix this script is for).

Usage
-----
    python scripts/validate_french.py --lake-root data
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from production.core.config import REPO_ROOT, risk_config
from production.core.lake import Lake, LakeError
from production.data.loaders.ken_french import COL_MAP
from production.risk.factor_returns import estimate_factor_returns
from production.risk.model import validate_against_french
from production.signals.base import sleeve_from_id

# factor (exposures.py column) -> vendor column name (Ken French's own header,
# and validate_against_french's own internal mapping for market/momentum).
FACTOR_TO_VENDOR = {"market": "Mkt-RF", "momentum": "Mom", "size": "SMB"}

# Only these factors are checked against configs/risk.yaml's
# validation.french_min_abs_corr gate. Size is informational only (see module
# docstring) -- never included here.
_GATED_FACTORS = ("market", "momentum")

DEFAULT_OUT = REPO_ROOT / "diagnostics" / "french_validation.json"


# --------------------------------------------------------------------- wiring
def _monthly_compound(daily: pd.DataFrame) -> pd.DataFrame:
    """Compound daily decimal returns to calendar-month returns.

    Identical to the compounding ``validate_against_french`` already applies to
    its ``factor_returns`` argument internally -- factored out so the call site
    can apply the SAME operation to the french benchmark side (which the
    function does not compound for you) and to factor pairs (size) the
    function's fixed mapping does not compute at all.
    """
    df = daily.copy()
    df.index = pd.to_datetime(df.index)
    return (1.0 + df).groupby(df.index.to_period("M")).prod() - 1.0


def extra_factor_correlation(factor_returns: pd.DataFrame, french_monthly: pd.DataFrame,
                             factor: str, vendor_name: str) -> float | None:
    """Correlate one daily estimated factor against one already-monthly wired
    french column, for factor/vendor pairs ``validate_against_french``'s own
    fixed ``{"market": "Mkt-RF", "momentum": "Mom"}`` mapping does not cover
    (currently: size vs SMB). Same monthly-compounding + inner-join-on-month
    + Pearson corr as ``validate_against_french`` itself, just parameterized
    to an arbitrary pair instead of the hardcoded two.

    Returns ``None`` if either column is absent or fewer than 3 overlapping
    months remain after the join (mirrors ``validate_against_french``'s own
    minimum-overlap guard).
    """
    if factor not in factor_returns.columns or vendor_name not in french_monthly.columns:
        return None
    fr_monthly = _monthly_compound(factor_returns[[factor]])
    joined = pd.concat([fr_monthly[factor], french_monthly[vendor_name]],
                       axis=1, join="inner").dropna()
    if len(joined) < 3:
        return None
    return float(joined.iloc[:, 0].corr(joined.iloc[:, 1]))


def wire_french_monthly(french_long: pd.DataFrame,
                        factors: dict[str, str] = FACTOR_TO_VENDOR) -> pd.DataFrame:
    """Pivot the curated ``french`` long panel into a monthly wide frame keyed
    by VENDOR column name (``Mkt-RF``, ``Mom``, ``SMB``, ...).

    ``french_long`` is the raw ``lake.read_curated("french")`` frame: long,
    daily, keyed by the loader's internal ``series_id`` (``FF_MKT_RF`` etc, see
    ``production/data/loaders/ken_french.py``). This is the call-site wiring the
    wiki log flagged as an open item -- see module docstring for the two gaps it
    closes. Raises ``ValueError`` loudly if a requested factor's series_id is
    missing from the curated frame, rather than silently producing an
    empty/misaligned join that would look like an ordinary low correlation
    instead of a wiring bug.
    """
    if french_long.empty:
        raise ValueError("wire_french_monthly: french curated frame is empty")
    wide_daily = french_long.pivot_table(index="obs_date", columns="series_id",
                                         values="value", aggfunc="last").sort_index()
    cols = {}
    for factor, vendor_name in factors.items():
        series_id = COL_MAP.get(vendor_name)
        if series_id is None:
            raise ValueError(
                f"wire_french_monthly: no series_id known for vendor column "
                f"{vendor_name!r} (factor {factor!r}) in ken_french.COL_MAP "
                f"({COL_MAP}) -- the mapping is stale or the factor name is wrong.")
        if series_id not in wide_daily.columns:
            raise ValueError(
                f"wire_french_monthly: series_id {series_id!r} (factor "
                f"{factor!r} -> vendor {vendor_name!r}) not present in the "
                f"curated french data; available series_ids="
                f"{sorted(wide_daily.columns)}. Refusing to silently correlate "
                f"against a missing series -- check COL_MAP against the lake.")
        cols[vendor_name] = wide_daily[series_id]
    return _monthly_compound(pd.DataFrame(cols))


# --------------------------------------------------------------------- loading
def _load_equity(lake: Lake, start, end) -> tuple[pd.DataFrame, list[str]]:
    prices = lake.read_curated("prices", asset_class="equity", start=start, end=end)
    if prices.empty:
        raise SystemExit("FATAL: no curated equity prices in the lake -- run ingestion first.")
    # Vendor-dedupe with the lake's latest-visible-vintage rule (asof_panel's
    # tie-break at as_of=now): on reused tickers the vendors can serve DIFFERENT
    # entities (2026-07-11 KG/MI/SBNY finding), and an undeduped mix manufactures
    # run-alternator returns that alone dragged market<->Mkt-RF to 0.538.
    if "available_from" in prices.columns:
        prices = (prices.sort_values(["available_from", "ingested_at"])
                        .drop_duplicates(subset=["obs_date", "instrument_id"],
                                         keep="last"))
    present = set(prices["instrument_id"].unique())
    try:
        master = lake.read_reference("instruments")
        ids = sorted(set(master.loc[master["sleeve"] == "equity", "instrument_id"]) & present)
    except LakeError:
        ids = []
    if not ids:  # fall back to the id-prefix rule (mirrors scripts/score_risk_model.py)
        ids = sorted({i for i in present if sleeve_from_id(i) == "equity"})
    if not ids:
        raise SystemExit("FATAL: no equity instruments present in the price panel.")
    return prices, ids


# -------------------------------------------------------------------- scoring
def run(lake_root: str, start=None, end=None) -> dict:
    lake = Lake(lake_root)
    cfg = risk_config()
    min_abs_corr = float(cfg["validation"]["french_min_abs_corr"])

    prices, ids = _load_equity(lake, start, end)
    sleeve_prices = prices[prices["instrument_id"].isin(ids)]
    fr_start = pd.to_datetime(sleeve_prices["obs_date"]).min()
    fr_end = pd.to_datetime(sleeve_prices["obs_date"]).max()
    factor_returns, _residuals = estimate_factor_returns(
        sleeve_prices, "equity", fr_start, fr_end, cfg)
    if factor_returns.empty:
        raise SystemExit("FATAL: estimate_factor_returns produced no rows -- too little history?")

    french_long = lake.read_curated("french")
    french_monthly = wire_french_monthly(french_long)

    # market / momentum: run through the actual production detector function.
    core = validate_against_french(factor_returns, french_monthly[["Mkt-RF", "Mom"]])

    verdict: dict[str, dict] = {}
    for factor in ("market", "momentum"):
        corr = core.get(factor)
        passed = (corr is not None and abs(corr) >= min_abs_corr)
        verdict[factor] = {"vendor_series": FACTOR_TO_VENDOR[factor], "corr": corr,
                           "gated": True, "passed": passed}

    # size: call-site-only (validate_against_french's fixed mapping has no size
    # pair) -- negative corr vs SMB is EXPECTED, never gates pass/fail.
    size_corr = extra_factor_correlation(factor_returns, french_monthly, "size", "SMB")
    verdict["size"] = {"vendor_series": FACTOR_TO_VENDOR["size"], "corr": size_corr,
                       "gated": False, "passed": None}

    n_months = int(pd.Index(factor_returns.index).to_period("M").nunique())
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "lake_root": str(lake_root),
        "min_abs_corr": min_abs_corr,
        "n_equity_ids": len(ids),
        "n_months": n_months,
        "factor_return_span": [str(pd.Timestamp(factor_returns.index.min()).date()),
                               str(pd.Timestamp(factor_returns.index.max()).date())],
        "series_id_map": dict(COL_MAP),
        "verdict": verdict,
    }


# ----------------------------------------------------------------------- cli
def _print_verdict(result: dict) -> None:
    span = result["factor_return_span"]
    print(f"validate_against_french: {result['n_equity_ids']} equity ids, "
         f"{span[0]}..{span[1]} ({result['n_months']} months), "
         f"gate |corr| >= {result['min_abs_corr']}")
    print(f"{'factor':<10}{'vs':<9}{'corr':>8}  {'verdict'}")
    for factor in ("market", "momentum", "size"):
        cell = result["verdict"][factor]
        corr = cell["corr"]
        corr_s = f"{corr:+.3f}" if corr is not None else "n/a"
        if not cell["gated"]:
            v = "INFO (not gated, negative expected)"
        else:
            v = "PASS" if cell["passed"] else "FAIL"
        print(f"{factor:<10}{cell['vendor_series']:<9}{corr_s:>8}  {v}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="validate_french.py",
        description="Validate estimated equity factor returns against Ken French benchmarks.")
    p.add_argument("--lake-root", default="data", help="lake root dir (default: data)")
    p.add_argument("--start", default=None, help="equity prices start (obs_date >=)")
    p.add_argument("--end", default=None, help="equity prices end (obs_date <=)")
    p.add_argument("--out", default=str(DEFAULT_OUT),
                   help="diagnostics JSON path (default: diagnostics/french_validation.json)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = run(args.lake_root, args.start, args.end)

    _print_verdict(result)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\ndiagnostics written: {out_path}")

    gated_pass = all(result["verdict"][f]["passed"] for f in _GATED_FACTORS)
    return 0 if gated_pass else 1


if __name__ == "__main__":
    sys.exit(main())
