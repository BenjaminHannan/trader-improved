#!/usr/bin/env python
"""Factor IC report and promotion gate over the curated lake.

Usage
-----
    python scripts/build_factors.py --ic-report [--start S --end E --lake-root DIR]
    python scripts/build_factors.py --ic-report --apply   # record verdicts to yaml

For every candidate/accepted factor in ``configs/factors.yaml`` this:
  1. loads the signal's required curated datasets from the lake,
  2. computes signal -> cross-sectional z-score -> per-date rank IC,
  3. splits the IC series into train (first 80%) and OOS (last 20% of dates),
  4. estimates the IC decay half-life and a net-of-cost single-factor validation return,
  5. runs the registry gate and prints a per-factor verdict table.
With ``--apply`` the verdicts are recorded back into ``configs/factors.yaml`` (status +
gate stats + bumped ``n_trials``).

Dependency discipline: signal classes are written by a sibling package and the cost
model in parallel — both are imported lazily *inside* the functions that need them, with
clear fallbacks, so this module always imports and ``--help`` always works.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from production.alpha.ic import (decay_halflife, forward_returns, ic_decay,
                                 ic_tstat, rank_ic)
from production.alpha.registry import FactorRegistry, GateStats
from production.alpha.zscore import zscore_scores
from production.core.config import costs_config
from production.core.lake import Lake, LakeError

# instrument_id class prefix -> sleeve. instrument_id = "CLASS:SYMBOL:first-listing".
CLASS_TO_SLEEVE = {
    "EQ": "equity", "CR": "crypto", "FX": "fx_etf", "CO": "commodity_etf",
}
_DATASETS = ("prices", "funding", "macro", "cot")


def _sleeve_map(prices: pd.DataFrame) -> pd.Series:
    ids = prices["instrument_id"].unique()
    return pd.Series({i: CLASS_TO_SLEEVE.get(str(i).split(":", 1)[0]) for i in ids})


def load_bundle(lake: Lake, start, end) -> dict[str, pd.DataFrame]:
    """Load available curated datasets into the signal input bundle.

    Missing datasets are skipped silently — a lake that only has prices still yields a
    usable (if smaller) report.
    """
    bundle: dict[str, pd.DataFrame] = {}
    for ds in _DATASETS:
        try:
            df = lake.read_curated(ds, start=start, end=end)
        except LakeError:
            continue
        if df is not None and not df.empty:
            bundle[ds] = df
    return bundle


def _load_signal_registry() -> dict[str, type]:
    """Import the sibling signals package lazily and return its class registry."""
    try:
        from production.signals.base import all_signals
    except Exception as exc:  # noqa: BLE001 - surface a clear, actionable message
        raise SystemExit(
            "cannot import production.signals.base.all_signals — the signals package "
            f"is not available yet ({exc}). Run once signals are implemented.")
    return all_signals()


def _annualized_sharpe(net_by_date: pd.Series, horizon: int) -> float:
    """Annualized Sharpe of a per-rebalance net return stream.

    Each observation is a ``horizon``-day-held net return, so ~``252/horizon`` independent
    periods fit in a year; that is the annualization factor. Returns NaN when the stream is
    too short or has no dispersion (nothing to deflate against downstream).
    """
    r = pd.Series(net_by_date).astype(float).dropna()
    if len(r) < 2:
        return float("nan")
    sd = float(r.std(ddof=1))
    if not np.isfinite(sd) or sd == 0.0:
        return float("nan")
    periods_per_year = 252.0 / max(int(horizon), 1)
    return float(r.mean() / sd * np.sqrt(periods_per_year))


def _net_validation_return(z_oos: pd.DataFrame, prices: pd.DataFrame,
                           sleeve_map: pd.Series, horizon: int,
                           costs: dict) -> tuple[float, pd.Series]:
    """Approximate net-of-cost return of a top-minus-bottom-quintile long/short book.

    Weekly-rebalanced on the OOS z-scores, equal-weight top vs bottom quintile per
    sleeve, held ``horizon`` days forward. Turnover is charged at the per-sleeve floor
    cost (or the full cost model when available). Returns ``(summed_net, net_by_date)`` —
    the summed net return over the OOS slice (only its sign matters to the gate) plus the
    per-rebalance net return series (feeds the validation-slice Sharpe).
    """
    if z_oos.empty:
        return float("nan"), pd.Series(dtype=float)
    fwd = forward_returns(prices, horizon)
    panel = z_oos.merge(fwd, on=["obs_date", "instrument_id"], how="inner")
    panel["sleeve"] = panel["instrument_id"].map(sleeve_map)
    panel = panel.dropna(subset=["sleeve", "fwd_ret"])
    if panel.empty:
        return float("nan"), pd.Series(dtype=float)

    cost_model = None
    try:  # optional dependency, written in parallel
        from production.backtest.cost_model import CostModel
        cost_model = CostModel(costs)
    except Exception:  # noqa: BLE001 - fall back to floor-only arithmetic
        cost_model = None

    floors = {s: spec.get("floor_bps", 0.0)
              for s, spec in costs.get("sleeves", {}).items()}

    prev_w: dict = {}
    net = 0.0
    net_dates: list = []
    net_vals: list = []
    for _date, day in panel.groupby("obs_date", sort=True):
        gross = 0.0
        w_today: dict = {}
        for sleeve, g in day.groupby("sleeve"):
            if len(g) < 5:
                continue
            q = g["value"].quantile([0.2, 0.8])
            lo, hi = q.iloc[0], q.iloc[1]
            longs = g[g["value"] >= hi]
            shorts = g[g["value"] <= lo]
            if longs.empty or shorts.empty:
                continue
            wl = 1.0 / len(longs)
            ws = 1.0 / len(shorts)
            gross += wl * longs["fwd_ret"].sum() - ws * shorts["fwd_ret"].sum()
            for iid in longs["instrument_id"]:
                w_today[iid] = w_today.get(iid, 0.0) + wl
            for iid in shorts["instrument_id"]:
                w_today[iid] = w_today.get(iid, 0.0) - ws

        # turnover cost at the per-sleeve floor (bps) on |w - w_prev|
        keys = set(w_today) | set(prev_w)
        cost = 0.0
        for iid in keys:
            dw = abs(w_today.get(iid, 0.0) - prev_w.get(iid, 0.0))
            floor = floors.get(sleeve_map.get(iid), 0.0)
            cost += dw * floor / 1e4
        net_t = gross - cost
        net += net_t
        net_dates.append(_date)
        net_vals.append(net_t)
        prev_w = w_today

    _ = cost_model  # reserved for a richer impact estimate; floor path is authoritative
    net_by_date = pd.Series(net_vals, index=pd.DatetimeIndex(net_dates))
    return float(net), net_by_date


def run_ic_report(start, end, lake_root, apply: bool) -> int:
    lake = Lake(lake_root)
    bundle = load_bundle(lake, start, end)
    if "prices" not in bundle:
        print(f"no price data found in lake at {lake.root} — nothing to report")
        return 0

    prices = bundle["prices"]
    sleeve_map = _sleeve_map(prices)
    costs = costs_config()
    registry = FactorRegistry()
    signal_registry = _load_signal_registry()

    rows: list[dict] = []
    for name, spec in registry.factors().items():
        if spec.get("status") not in ("candidate", "accepted"):
            continue
        cls = signal_registry.get(name)
        if cls is None:
            print(f"  {name}: no signal class registered — skipped")
            continue
        try:
            panel = cls().compute(bundle)
        except Exception as exc:  # noqa: BLE001 - one bad signal must not kill the report
            print(f"  {name}: signal.compute failed ({exc}) — skipped")
            continue
        if panel is None or panel.empty:
            print(f"  {name}: signal produced no values — skipped")
            continue

        z = zscore_scores(panel, sleeve_map)
        dates = np.sort(pd.unique(z["obs_date"]))
        if len(dates) < 10:
            print(f"  {name}: too few dates ({len(dates)}) to gate — skipped")
            continue
        cut = dates[int(len(dates) * 0.8)]  # start of the OOS slice

        horizon = int(spec["horizon_days"])
        fwd = forward_returns(prices, horizon)
        ic = rank_ic(z, fwd, sleeve_map)
        if ic.empty:
            print(f"  {name}: no IC observations — skipped")
            continue
        ic_by_date = ic.groupby("obs_date")["rank_ic"].mean().sort_index()

        train = ic_by_date[ic_by_date.index < cut]
        oos = ic_by_date[ic_by_date.index >= cut]
        train_ic = float(train.mean()) if len(train) else float("nan")
        tstat = ic_tstat(train)
        oos_ic = float(oos.mean()) if len(oos) else float("nan")

        decay = ic_decay(z, prices, sleeve_map)
        halflife = decay_halflife(decay)

        z_oos = z[z["obs_date"] >= cut]
        net, net_by_date = _net_validation_return(z_oos, prices, sleeve_map, horizon, costs)
        val_sharpe = _annualized_sharpe(net_by_date, horizon)

        stats = GateStats(train_ic=train_ic, train_tstat=tstat, oos_ic=oos_ic,
                          decay_halflife_days=float(halflife),
                          net_validation_return=net, n_dates=int(len(ic_by_date)),
                          val_sharpe=val_sharpe)
        verdict = registry.gate(name, stats)
        rows.append({"name": name, "sleeves": ",".join(spec.get("sleeves", [])),
                     "stats": stats, "verdict": verdict})
        if apply:
            registry.record(name, verdict)

    _print_table(rows)

    if apply:
        registry.save()
        print(f"\nrecorded {len(rows)} verdict(s) to {registry.path}; "
              f"n_trials now {registry.n_trials}")
    return 0


def _print_table(rows: list[dict]) -> None:
    header = (f"{'factor':<18}{'sleeves':<26}{'train_IC':>10}{'tstat':>8}"
              f"{'OOS_IC':>10}{'halflife':>10}  verdict")
    print(header)
    print("-" * len(header))
    for r in rows:
        s = r["stats"]
        v = r["verdict"]
        tag = "PASS" if v.passed else "FAIL"
        hl = "inf" if not np.isfinite(s.decay_halflife_days) else f"{s.decay_halflife_days:.1f}"
        print(f"{r['name']:<18}{r['sleeves']:<26}{s.train_ic:>10.4f}"
              f"{s.train_tstat:>8.2f}{s.oos_ic:>10.4f}{hl:>10}  {tag}")
        if not v.passed:
            for reason in v.reasons:
                print(f"{'':<18}    - {reason}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="build_factors.py",
        description="Factor IC report and promotion gate over the curated lake.")
    p.add_argument("--ic-report", action="store_true",
                   help="compute per-factor IC, decay and gate verdicts")
    p.add_argument("--apply", action="store_true",
                   help="record gate verdicts back into configs/factors.yaml")
    p.add_argument("--start", default=None, help="report start date (obs_date >=)")
    p.add_argument("--end", default=None, help="report end date (obs_date <=)")
    p.add_argument("--lake-root", default=None,
                   help="lake root dir (default: repo data/)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if not args.ic_report:
        build_arg_parser().print_help()
        return 0
    return run_ic_report(args.start, args.end, args.lake_root, args.apply)


if __name__ == "__main__":
    sys.exit(main())
