#!/usr/bin/env python
"""Run the risk-model scoring harness against the lake and write a ledger.

Usage
-----
    python scripts/score_risk_model.py --lake-root data --start 2019-01-01 --end 2021-12-31

Builds the incumbent Sigma_t per sleeve from ``production/risk/model.py``'s
existing ``RiskModel.build`` (the same constructor the live backtest/allocation
path uses), runs :func:`production.risk.validation.bias_stats` for families 1-4
(the "available extras" -- 5/6 need pre-built pure-factor / cross-sleeve books
this script does not assemble; see the note below), prints the per-(sleeve x
family) table, runs the baseline health check (>= 90% of family-1/2 cells
in-band -- research/wiki/questions/research-risk-model-validation.md Q1), and
writes the ledger JSON to ``diagnostics/risk_harness/``.

This is a BASELINE run (no candidate model): it establishes/monitors the
CURRENT model's calibration. Adoption calls (comparing an incumbent to a
candidate via :func:`production.risk.validation.adoption_verdict`) are a
separate, later invocation once an actual candidate change exists -- Q1's
harness adjudicates Q2-Q5 in the wiki, none of which are implemented here.

Why families 5/6 are unavailable from this script alone: family 5 (pure-factor
books) needs UNIT-EXPOSURE portfolios solved from the equity structural
sleeve's ``B`` matrix (``production/risk/exposures.py``) -- that requires a
small QP/pseudo-inverse step this script does not build, and the crypto
structural sleeve's ``min_obs=252`` factor-covariance gate
(``production/risk/model.py:_build_structural``) means crypto often falls back
to the small-sleeve full-covariance mode with no ``B`` at all, so "the" factor
exposures are sleeve- and history-dependent, not a fixed input. Family 6
(cross-sleeve hedged books, e.g. long equity names / short the matching sector
ETF) needs a name->sector-ETF mapping that does not exist as a lake table --
the instrument reference master carries a GICS ``sector`` label, not a
tradable-hedge ``instrument_id``. Both interfaces exist in
:func:`production.risk.validation.build_test_portfolios` (pass ``external``);
this script simply has nothing to pass yet.

The ``--synthetic`` no-lake smoke path exercises the identical scoring
mechanics on the conftest GBM bundle, mirroring ``scripts/run_backtest.py``.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from production.core.config import REPO_ROOT, risk_config
from production.risk.model import RiskModel
from production.risk.validation import (baseline_health_check, bias_stats,
                                        build_ledger_record, build_test_portfolios,
                                        write_ledger)

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from production.core.lake import Lake, LakeError
from production.signals.base import sleeve_from_id

_FAMILIES = (1, 2, 3, 4)


# ------------------------------------------------------------------- loading
def _load_prices(lake: Lake, start, end) -> pd.DataFrame:
    try:
        return lake.read_curated("prices", start=start, end=end)
    except LakeError:
        return pd.DataFrame()


def _instrument_map(lake: Lake, prices: pd.DataFrame) -> pd.Series:
    """instrument_id -> sleeve from the reference master, else the id-prefix
    fallback (same rule as scripts/run_backtest.py's ``_instrument_map``)."""
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
    return pd.Series({i: sleeve_from_id(i) for i in sorted(present)})


def _sector_map(lake: Lake, prices: pd.DataFrame) -> pd.Series | None:
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


def _synthetic_prices(start, end):
    """The conftest GBM bundle, no lake needed (mirrors run_backtest.py)."""
    from tests.conftest import make_gbm_prices

    data_start = (pd.Timestamp(start) - pd.DateOffset(years=2)).strftime("%Y-%m-%d")
    eq = {f"EQ:SYN{i:02d}:2000-01-03": "equity" for i in range(10)}
    fx = {f"FX:{s}:2007-01-03": "fx_etf" for s in ["FXE", "FXY", "FXB"]}
    instruments = {**eq, **fx}
    prices = make_gbm_prices(instruments, start=data_start, end=str(pd.Timestamp(end).date()))
    return prices, pd.Series(instruments), None


# -------------------------------------------------------------------- scoring
_MIN_COVERAGE = 0.98


def _wide_returns(prices: pd.DataFrame, ids: list[str],
                  min_coverage: float = _MIN_COVERAGE) -> pd.DataFrame:
    """Complete-panel daily returns for the full-span coverage core of a sleeve.

    The bias-statistic z needs a REALIZED h-day portfolio return; one NaN daily
    return in any held name poisons the whole window (this is exactly what
    zeroed the churned equity/crypto sleeves on the first baseline run —
    RiskModel.build itself was fine). Calibration wants a fixed complete panel:
    keep only ids with >= 98% price coverage over the scored span (the
    full-span core, so the cross-name date intersection stays dense),
    forward-fill residual stale-quote holes (limit 10 days), and drop any
    still-incomplete dates. The dropped-count is printed so a survivor-core
    caveat is never silent; a time-varying-universe scoring mode is the
    documented follow-up for measuring the model on the exact PIT book.
    """
    df = prices[prices["instrument_id"].isin(ids)]
    close = df.pivot_table(index="obs_date", columns="instrument_id",
                           values="close", aggfunc="last").sort_index()
    coverage = close.notna().mean()
    kept = coverage[coverage >= min_coverage].index.tolist()
    dropped = int((coverage < min_coverage).sum())
    if dropped:
        print(f"  coverage filter: scoring {len(kept)}/{len(coverage)} ids "
              f"(dropped {dropped} below {min_coverage:.0%} full-span coverage)")
    if len(kept) < 3:
        return pd.DataFrame()
    filled = close[kept].ffill(limit=10)
    rets = filled.pct_change().dropna(how="all").dropna(axis=0, how="any")
    return rets


def _sigma_fn(prices: pd.DataFrame, sleeve: str, ids: list[str], cfg: dict,
             sectors: pd.Series | None):
    """Wrap RiskModel.build as a validation.SigmaFn: as_of = the window's last
    date, i.e. strictly before the evaluation date (the caller sliced the
    window to `< t` already) -- RiskModel.build's own `obs_date <= as_of`
    filter then reproduces the identical PIT cutoff."""
    def fn(window: pd.DataFrame) -> pd.DataFrame:
        if window.empty:
            raise ValueError("empty trailing window")
        as_of = window.index.max()
        return RiskModel.build(prices, sleeve, as_of, ids, cfg, sectors).covariance()
    return fn


def _memoize_sigma_fn(sigma_fn):
    """Wrap a validation.SigmaFn with an as_of-keyed memo.

    score_sleeve calls bias_stats once per family (4x) against the SAME
    sigma_fn, and bias_stats itself walks the SAME ~113 non-overlapping
    evaluation dates every time (evaluation_dates is a pure function of the
    returns panel + h/min_obs, which score_sleeve holds fixed across all 4
    calls). Each of those 4 walks independently rebuilds Sigma_t via
    RiskModel.build for every date -- 4x the expensive covariance build for
    zero new information.

    Cache key: `as_of = window.index.max()`, exactly what the wrapped fn
    (see `_sigma_fn.fn` above) already derives from `window` to pass into
    RiskModel.build. This key is sufficient -- not just convenient -- because:
      1. score_sleeve builds ONE `returns_panel` and passes that same object,
         unmutated, into every bias_stats call for this sleeve;
      2. bias_stats always calls sigma_fn(returns_panel.iloc[:pos]) -- the
         window is a PREFIX of that one fixed panel, never an arbitrary slice;
      3. a prefix of a fixed sequence is uniquely determined by its length,
         and its length is uniquely determined by its last index value
         (returns_panel.index is monotonic and has no duplicate dates), so
         `as_of` alone pins down the window's content -- no need to hash the
         window itself.

    Memory: the cache holds one covariance DataFrame per evaluation date, for
    the life of one score_sleeve call (it is a local dict, not module-level
    state, so it is freed when score_sleeve returns). Worst case is the
    ~400-name equity sleeve: a 400x400 float64 frame is 400*400*8 bytes ~=
    1.3MB; ~113 evaluation dates (the T~=120 the validation module's
    docstring cites) -> ~150MB peak resident. That is acceptable for a CLI
    run, so no eviction policy is implemented.

    Exceptions from the underlying build (e.g. too few trailing observations,
    a singular covariance) are NOT cached: only a successful Sigma is stored.
    bias_stats treats a sigma_fn failure as "skip this date" per call, not a
    crash -- caching a failure would silently convert a per-call retry into a
    permanently poisoned date the moment any one family hit it first.
    """
    cache: dict = {}

    def wrapped(window: pd.DataFrame) -> pd.DataFrame:
        as_of = window.index.max()
        if as_of in cache:
            return cache[as_of]
        sigma = sigma_fn(window)   # let exceptions propagate uncached
        cache[as_of] = sigma
        return sigma
    return wrapped


def score_sleeve(prices: pd.DataFrame, sleeve: str, ids: list[str], cfg: dict,
                 sectors: pd.Series | None, n: int, seed: int,
                 min_obs: int) -> dict:
    """Bias-stats cells for families 1-4 on one sleeve. Returns {family: dict}."""
    returns_panel = _wide_returns(prices, ids)
    rng = np.random.default_rng(seed)
    # Memoized across all 4 families below: same returns_panel, same
    # evaluation-date grid, so each date's expensive RiskModel.build runs
    # once instead of 4x -- see _memoize_sigma_fn's docstring for why the
    # as_of key is sufficient.
    sigma_fn = _memoize_sigma_fn(_sigma_fn(prices, sleeve, ids, cfg, sectors))
    cells = {}
    for family in _FAMILIES:
        if family == 4:
            weights = lambda Sig, _s=sleeve, _r=returns_panel, _rng=rng: \
                build_test_portfolios(_r, _s, 4, _rng, sigma=Sig)
        else:
            weights = build_test_portfolios(returns_panel, sleeve, family, rng, n=n)
        cells[family] = bias_stats(returns_panel, sigma_fn, weights, min_obs=min_obs)
    return cells


def _print_table(all_cells: dict) -> None:
    print("\n" + "=" * 78)
    print(f"{'sleeve':<14}{'family':<8}{'B':>8}{'band':>18}{'in_band':>10}"
         f"{'T':>7}{'MRAD':>8}")
    print("-" * 78)
    for (sleeve, family), cell in sorted(all_cells.items()):
        band = cell.get("band", (float("nan"), float("nan")))
        band_s = f"[{band[0]:.3f},{band[1]:.3f}]" if np.isfinite(band[0]) else "n/a"
        print(f"{sleeve:<14}{family:<8}{cell.get('B', float('nan')):>8.3f}"
             f"{band_s:>18}{str(cell.get('in_band')):>10}"
             f"{cell.get('T', 0):>7}{cell.get('MRAD', float('nan')):>8.3f}")
    print("=" * 78)


# ----------------------------------------------------------------------- main
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="score_risk_model.py",
                                description="Risk-model bias-statistic baseline harness.")
    p.add_argument("--lake-root", default="data", help="lake root dir (default: data)")
    p.add_argument("--start", default=None, help="prices start (obs_date >=)")
    p.add_argument("--end", default=None, help="prices end (obs_date <=)")
    p.add_argument("--synthetic", action="store_true",
                   help="run on a synthesized GBM bundle without any lake (smoke path)")
    p.add_argument("--n", type=int, default=100, help="random draws per family 1/2 (default 100)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--min-obs", type=int, default=252,
                   help="trailing obs required before the first evaluation date")
    p.add_argument("--ledger-dir", default="diagnostics/risk_harness")
    p.add_argument("--instrument-cov", default=None,
                   help="CANDIDATE override for instrument_covariance (small ETF "
                        "sleeves): 'lw_cc' | 'lw' | 'fixed:<shrink>' (e.g. fixed:0.0 "
                        "= raw EWMA). Structural sleeves are untouched. The config "
                        "file is never modified — this exists so candidate-vs-"
                        "incumbent adjudication runs are one command each.")
    p.add_argument("--sleeves", default=None,
                   help="comma-separated sleeve subset (default: all present); a "
                        "candidate that only changes instrument_covariance only "
                        "needs the covariance sleeves re-scored")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    cfg = risk_config()
    if args.instrument_cov:
        spec = str(args.instrument_cov)
        icov = dict(cfg.get("instrument_covariance") or {})
        if spec.startswith("fixed"):
            icov["method"] = "fixed"
            if ":" in spec:
                icov["shrinkage_to_diagonal"] = float(spec.split(":", 1)[1])
        else:
            icov["method"] = spec
        cfg = {**cfg, "instrument_covariance": icov}
        print(f"CANDIDATE instrument_covariance override: {icov} "
              "(config file untouched)")

    if args.synthetic:
        print("synthetic run: GBM bundle, no lake")
        prices, instruments, sectors = _synthetic_prices(
            args.start or "2019-01-01", args.end or "2020-06-30")
    else:
        lake = Lake(args.lake_root)
        prices = _load_prices(lake, args.start, args.end)
        if prices.empty:
            print(f"FATAL: no curated 'prices' in lake at {lake.root} -- "
                 f"run ingestion first, or pass --synthetic for a no-lake smoke run.",
                 file=sys.stderr)
            return 2
        instruments = _instrument_map(lake, prices)
        sectors = _sector_map(lake, prices)

    all_sleeves = list(cfg["structural_sleeves"]) + list(cfg["covariance_sleeves"])
    present_sleeves = [s for s in all_sleeves if (instruments == s).any()]
    if args.sleeves:
        wanted = {s.strip() for s in args.sleeves.split(",") if s.strip()}
        present_sleeves = [s for s in present_sleeves if s in wanted]
    if not present_sleeves:
        print("FATAL: no sleeve in configs/risk.yaml has any instrument in the price panel.",
             file=sys.stderr)
        return 2

    all_cells: dict = {}
    for sleeve in present_sleeves:
        ids = sorted(instruments[instruments == sleeve].index)
        sleeve_sectors = sectors.reindex(ids).dropna() if sectors is not None else None
        try:
            cells = score_sleeve(prices, sleeve, ids, cfg, sleeve_sectors,
                                 args.n, args.seed, args.min_obs)
        except Exception as exc:  # noqa: BLE001 - one bad sleeve must not sink the run
            print(f"  note: sleeve {sleeve!r} failed to score ({exc!r}) -- skipping")
            continue
        for family, cell in cells.items():
            all_cells[(sleeve, family)] = cell

    if not all_cells:
        print("FATAL: no sleeve produced a scoreable cell (insufficient history everywhere?).",
             file=sys.stderr)
        return 2

    _print_table(all_cells)
    health = baseline_health_check(all_cells)
    print(f"\nbaseline health: {'PASS' if health['passed'] else 'FAIL'} "
         f"({health['fraction_in_band']:.0%} of {health['n_cells']} random-book cells "
         f"in-band; need >= 90%)")
    if health["failing_cells"]:
        print(f"  failing cells: {health['failing_cells']}")

    record = build_ledger_record(
        incumbent_cells=all_cells, baseline_health=health,
        notes=[f"n={args.n}", f"seed={args.seed}", f"min_obs={args.min_obs}",
              f"sleeves={present_sleeves}",
              "families 5/6 not scored: no pre-built external books (see module docstring)"])
    path = write_ledger(record, out_dir=args.ledger_dir)
    print(f"\nledger written: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
