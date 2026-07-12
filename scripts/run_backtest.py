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
It also loads the curated ``event_markets`` panel (Kalshi/Polymarket) for the ``events``
sleeve, config-gated by ``events.enabled`` in configs/backtest.yaml (default on); an
absent/empty curated dataset degrades to a printed note and a run without the sleeve,
exactly like the other optional lake datasets (never a crash). The report's per-sleeve
table then carries ``events`` alongside crypto/fx_etf/... whenever it fired.

The ``--synthetic`` flag is the no-lake smoke path: it builds the conftest GBM bundle
(with ~3y of warmup history synthesized before ``--start``) and runs the identical engine
without any lake access, so it never sees an events panel either.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from production.backtest.engine import run_backtest
from production.backtest.report import write_report
from production.core.config import CONFIG_DIR, REPO_ROOT, backtest_config

# Ensure the repo root is importable so `--synthetic` can pull the conftest makers even
# when the script is invoked directly (not under pytest, which adds it automatically).
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from production.core.lake import Lake, LakeError, SIGNAL_BUNDLE_DATASETS, read_signal_bundle
from production.data.cross_check import quarantine_list
from production.signals.base import sleeve_from_id

_DATASETS = SIGNAL_BUNDLE_DATASETS
_QUARANTINE_PREVIEW_N = 5
_EVENTS_DATASET = "event_markets"
_EVENTS_ASSET_CLASS = "events"


def _load_event_markets(lake: Lake, start, end, events_enabled: bool = True):
    """Load the curated Kalshi/Polymarket ``event_markets`` panel for the events sleeve.

    Config-gated (``events.enabled`` in configs/backtest.yaml, default True) and, once
    enabled, degrades exactly like the optional ``SIGNAL_BUNDLE_DATASETS`` entries (e.g.
    ``tvl``): an absent or empty curated dataset prints a note and returns ``None`` rather
    than raising — the lake path never crashes for missing kalshi/events data. Disabling
    the flag skips the lake read entirely (its own note), so ``events.enabled=false``
    behaves identically to "no events data ever ingested".
    """
    if not events_enabled:
        print("  note: events sleeve disabled via configs/backtest.yaml "
              "(events.enabled=false) — skipping")
        return None
    try:
        df = lake.read_curated(_EVENTS_DATASET, _EVENTS_ASSET_CLASS, start=start, end=end)
    except LakeError:
        df = None
    if df is None or df.empty:
        print(f"  note: optional dataset '{_EVENTS_DATASET}' absent — continuing "
              "without the events sleeve")
        return None
    return df


def _load_lake_bundle(lake: Lake, start, end, events_enabled: bool = True) -> dict:
    """Load the signal bundle from the lake, echoing which optional datasets are absent.

    ``prices`` is required (a FATAL note if missing); the rest are optional. Bundle-key
    normalization (mcap<-crypto_meta, tvl<-defi_tvl) is handled by
    ``production.core.lake.read_signal_bundle``. The ``events`` sleeve's ``event_markets``
    panel is loaded separately (see ``_load_event_markets``) since, unlike the signal
    bundle datasets, it is config-gated rather than always-attempted.
    """
    bundle = read_signal_bundle(lake, start=start, end=end)
    for ds in _DATASETS:
        if ds in bundle:
            continue
        if ds == "prices":
            print(f"FATAL: no curated 'prices' in lake at {lake.root}", file=sys.stderr)
        else:
            print(f"  note: optional dataset '{ds}' absent — continuing without it")
    event_panel = _load_event_markets(lake, start, end, events_enabled=events_enabled)
    if event_panel is not None:
        bundle["event_markets"] = event_panel
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


def quarantine_audit_path(lake_root) -> Path:
    """Default on-disk location of the price cross-check audit for a lake root."""
    return Path(lake_root) / "audit" / "cross_check_prices.json"


def _preview(ids: list[str]) -> str:
    shown = ", ".join(ids[:_QUARANTINE_PREVIEW_N])
    more = f", ... ({len(ids) - _QUARANTINE_PREVIEW_N} more)" if len(ids) > _QUARANTINE_PREVIEW_N else ""
    return f"[{shown}{more}]"


def apply_quarantine(instruments: pd.Series, audit_path, no_quarantine: bool = False) -> pd.Series:
    """Drop cross-check-quarantined EQUITY instruments from the instrument->sleeve map.

    ``scripts/ingest.py --cross-check`` writes a return-space vendor-divergence report to
    ``<lake_root>/audit/cross_check_prices.json``; ``production.data.cross_check.quarantine_list``
    turns it into the set of instrument_ids whose flagged fraction is too high to trust.

    Class distinction (wiki backlog #18 / research/wiki/sources/etf-adjustment-methodology.md):
    the FX-ETF / commodity-ETF slice of that list is explained by cross-vendor adjusted-close
    *methodology* divergence (anchor price, rounding, retrieval-date drift on distribution
    events) — the instruments are healthy, the comparison series was the bug. Dropping all
    quarantined FX ETFs would gut the fx sleeve an ACCEPTED factor trades, so that class is
    kept with a loud warning instead. Quarantined EQUITY names (ticker-reuse-class corruption)
    have no such exoneration and are dropped from the traded universe until resolved by hand.

    Parameters
    ----------
    instruments   : instrument_id -> sleeve Series (as built by ``_instrument_map``).
    audit_path    : path to the cross-check audit JSON (see ``quarantine_audit_path``).
    no_quarantine : bypass the filter entirely (still prints a loud note that it was bypassed).

    An absent audit file reproduces prior behavior bit-identically: no filtering, no message.
    A malformed audit file is treated the same way (parsed defensively; never sinks the run).
    """
    audit_path = Path(audit_path)
    if no_quarantine:
        print(f"quarantine: --no-quarantine set — price cross-check quarantine filter "
              f"BYPASSED ({audit_path} not consulted)")
        return instruments
    if not audit_path.exists():
        return instruments
    try:
        with open(audit_path) as f:
            report = json.load(f)
        quarantined = quarantine_list(report)
    except Exception as exc:  # noqa: BLE001 - a malformed audit file must not sink the run
        print(f"quarantine: could not parse {audit_path} ({exc}) — skipping quarantine filter")
        return instruments

    present = [iid for iid in quarantined if iid in instruments.index]
    if not present:
        return instruments

    def _class(iid: str) -> str:
        try:
            return sleeve_from_id(iid)
        except ValueError:
            return "unknown"

    equity_drop = [iid for iid in present if _class(iid) == "equity"]
    other_warn = [iid for iid in present if iid not in equity_drop]

    if equity_drop:
        print(f"quarantine: dropped {len(equity_drop)} equity instrument(s) flagged by the "
              f"price cross-check: {_preview(equity_drop)}")
        instruments = instruments.drop(index=equity_drop)

    if other_warn:
        print(f"quarantine: WARNING — {len(other_warn)} non-equity instrument(s) flagged by "
              "the price cross-check are KEPT (FX/commodity-ETF vendor adjusted-close "
              "divergence is a cross-check-methodology artifact, not corrupt primary data — "
              f"see research/wiki/sources/etf-adjustment-methodology.md): {_preview(other_warn)}")

    return instruments


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
    p.add_argument("--no-quarantine", action="store_true",
                   help="bypass price cross-check quarantine filtering (loud note when used)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    cfg = backtest_config()  # validated config; --config path is echoed for provenance

    if args.synthetic:
        start = args.start or "2019-01-01"
        end = args.end or "2020-06-30"
        print(f"synthetic run: warmup-synthesized bundle, trading ~{start} .. {end}")
        data, instruments, sectors = _synthetic_bundle(start, end)
        lake = None
    else:
        lake = Lake(args.lake_root)
        events_enabled = bool((cfg.get("events") or {}).get("enabled", True))
        data = _load_lake_bundle(lake, args.start, args.end, events_enabled=events_enabled)
        if "prices" not in data:
            return 2
        instruments = _instrument_map(lake, data["prices"])
        instruments = apply_quarantine(instruments, quarantine_audit_path(args.lake_root),
                                       no_quarantine=args.no_quarantine)
        sectors = _sector_map(lake, data["prices"])

    factors_cfg = None
    if args.synthetic:
        # The synthetic smoke exercises ENGINE MECHANICS. Once the real gate has run,
        # the live accepted set may require datasets (basis, full macro) the GBM
        # bundle deliberately lacks — so pin the smoke to an all-candidate registry
        # snapshot; the real (lake) path stays gated.
        import tempfile

        import yaml as _yaml

        from production.core.config import CONFIG_DIR as _CD
        _fc = _yaml.safe_load(open(_CD / "factors.yaml"))
        for _spec in _fc["factors"].values():
            _spec["status"] = "candidate"
        _tmp = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
        _yaml.safe_dump(_fc, _tmp, sort_keys=False)
        _tmp.close()
        factors_cfg = _tmp.name

    result = run_backtest(data, instruments, cfg=cfg, sectors=sectors,
                          factors_cfg=factors_cfg, lake=lake)
    path = write_report(result.report, out_dir=args.report_dir)
    _print_headline(result.report)
    print(f"\nreport written: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
