#!/usr/bin/env python
"""Trailing-24-month health check for every ACCEPTED factor (measurement only).

Implements the committed rule (``research/wiki/log.md``, 2026-07-12 "Factor-health
rule + basis_carry demotion round" — see the commit before this script existed,
620f1ad): a live factor is demoted when its trailing-24-month rank-IC, sign-adjusted
to the accepted direction, has month-clustered t <= 0. Symmetric: every factor with
``status == "accepted"`` in ``configs/factors.yaml`` is measured every round, none
singled out.

Measurement spec
-----------------
For each accepted factor:
  1. build the same signal panel the engine trades (``production/signals/*`` via the
     dotted path in ``configs/factors.yaml``, loaded the same way
     ``scripts/build_factors.py`` does — ``read_signal_bundle`` over the full lake so
     every signal gets its real warm-up history, not a truncated slice);
  2. cross-sectional z-score (``production.alpha.zscore.zscore_scores``);
  3. daily rank-IC vs forward returns at the factor's OWN ``horizon_days``
     (``production.alpha.ic.rank_ic`` / ``forward_returns``), sleeves aggregated with
     the same precision (n_names-1) weighting ``build_factors.py`` uses, so a thin
     sleeve cannot swamp a deep one;
  4. sign-adjust every daily IC to the accepted direction (the sign of the factor's
     recorded ``gate_stats.train_ic`` — the gate requires OOS IC to share that sign
     before a factor can be accepted at all, so train_ic and oos_ic always agree);
  5. restrict to the trailing 24 months ending at the latest available ``obs_date``;
  6. month-clustered (calendar-month) intercept-only cluster-robust t-stat on the
     sign-adjusted daily IC series.

``t <= 0`` -> DEMOTE. The prices panel is vendor-deduped BEFORE any of the above
touches it, mirroring ``scripts/score_risk_model.py::_wide_returns`` (a reused ticker
served by two different vendor entities otherwise lets an arbitrary vendor win per
cell and can manufacture fake return spikes — the 2026-07-11 KG/MI/SBNY finding).

This module never mutates ``configs/factors.yaml`` on its own. ``--apply`` calls
``FactorRegistry.demote`` for every factor the rule says to demote and prints what
changed; without it, this is read-only measurement + a diagnostics JSON.

Dependency discipline: mirrors ``build_factors.py`` — signal classes come from a
sibling package, imported lazily inside the functions that need them.

Usage
-----
    python scripts/factor_health.py                       # measure + print + write json
    python scripts/factor_health.py --apply                # measure, then demote + save
    python scripts/factor_health.py --lake-root data --start 2015-01-01
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from production.alpha.ic import forward_returns, rank_ic
from production.alpha.registry import FactorRegistry
from production.alpha.zscore import zscore_scores
from production.core.config import REPO_ROOT
from production.core.lake import Lake

# ``scripts`` is loose files, not an installed package (pyproject only packages
# ``production*``) — ``python scripts/factor_health.py`` puts ``scripts/`` itself on
# sys.path[0], not the repo root, so ``scripts.build_factors`` is not importable
# without this. Same fix as scripts/score_risk_model.py's REPO_ROOT insertion.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from scripts.build_factors import _load_signal_registry, _sleeve_map, load_bundle  # noqa: E402

_TRAILING_MONTHS = 24
_DIAGNOSTICS_DIR = REPO_ROOT / "diagnostics"


# --------------------------------------------------------------------- dedupe
def _dedupe_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Vendor-dedupe rule mirrored from ``scripts/score_risk_model.py::_wide_returns``.

    Sort by ``(available_from, ingested_at)`` and keep the latest-visible vintage per
    ``(obs_date, instrument_id)`` — applied BEFORE the deduped frame feeds either the
    signal's own computation or the forward-return/IC step, since a reused-ticker
    vendor collision corrupts both identically.
    """
    if "available_from" not in prices.columns:
        return prices
    sort_cols = ["available_from"]
    if "ingested_at" in prices.columns:
        sort_cols.append("ingested_at")
    return (prices.sort_values(sort_cols)
                  .drop_duplicates(subset=["obs_date", "instrument_id"], keep="last")
                  .reset_index(drop=True))


# ---------------------------------------------------------- cluster-robust t
def _cluster_mean_t(values: np.ndarray, clusters: np.ndarray) -> dict:
    """Intercept-only cluster-robust (CR1) t-stat of a series' mean.

    Inlined rather than imported: ``scripts/`` is production-adjacent and does not
    import ``research/`` helpers (no other script under ``scripts/`` does either).
    This is the same CR1 intercept-only cluster OLS as
    ``research/diagnostics/_common.py::cluster_ols`` / ``cluster_mean_test`` — mirrored
    here as a small standalone helper rather than imported, per that boundary.
    """
    y = np.asarray(values, dtype=float)
    n = len(y)
    labels = pd.factorize(pd.Series(clusters))[0]
    g = int(labels.max()) + 1 if n else 0
    if n < 2 or g < 2:
        return {"mean": float(y.mean()) if n else float("nan"),
               "se": float("nan"), "t": float("nan"), "n": n, "n_clusters": g}

    X = np.ones((n, 1))
    k = 1
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    resid = y - X @ beta
    meat = np.zeros((k, k))
    for lab in range(g):
        idx = labels == lab
        s = X[idx].T @ resid[idx]
        meat += np.outer(s, s)
    corr = (g / (g - 1)) * ((n - 1) / (n - k)) if n > k else 1.0
    V = corr * XtX_inv @ meat @ XtX_inv
    se = float(np.sqrt(V[0, 0])) if V[0, 0] > 0 else float("nan")
    t = float(beta[0] / se) if se == se and se > 0 else float("nan")
    return {"mean": float(beta[0]), "se": se, "t": t, "n": n, "n_clusters": g}


# --------------------------------------------------------------- measurement
def _load_signal_registry_safe() -> dict[str, type]:
    return _load_signal_registry()


def measure_factor(name: str, spec: dict, bundle: dict[str, pd.DataFrame],
                   prices: pd.DataFrame, sleeve_map: pd.Series,
                   signal_registry: dict[str, type]) -> dict:
    """One factor's trailing-24m sign-adjusted rank-IC health, or an ``error`` entry."""
    cls = signal_registry.get(name)
    if cls is None:
        return {"name": name, "error": "no signal class registered"}
    try:
        panel = cls().compute(bundle)
    except Exception as exc:  # noqa: BLE001 - one bad signal must not kill the report
        return {"name": name, "error": f"signal.compute failed ({exc})"}
    if panel is None or panel.empty:
        return {"name": name, "error": "signal produced no values"}

    z = zscore_scores(panel, sleeve_map)
    if z.empty:
        return {"name": name, "error": "no z-scores produced"}

    horizon = int(spec["horizon_days"])
    fwd = forward_returns(prices, horizon)
    ic = rank_ic(z, fwd, sleeve_map)
    if ic.empty:
        return {"name": name, "error": "no IC observations"}

    # Precision-weighted (n_names - 1) per-date aggregation across sleeves — same
    # weighting build_factors.py::run_ic_report uses, so a thin sleeve cannot swamp a
    # deep one in the pooled daily IC (2026-07-07 mom_12_1 aggregation-artifact note).
    icw = ic.assign(_w=(ic["n_names"].clip(lower=2) - 1).astype(float))
    ic_by_date = (icw.assign(_wx=icw["rank_ic"] * icw["_w"])
                     .groupby("obs_date")[["_wx", "_w"]].sum()
                     .pipe(lambda g: g["_wx"] / g["_w"]).sort_index())
    if ic_by_date.empty:
        return {"name": name, "error": "no aggregated IC observations"}

    latest_obs = pd.Timestamp(prices["obs_date"].max())
    window_start = latest_obs - pd.DateOffset(months=_TRAILING_MONTHS)
    trailing = ic_by_date[ic_by_date.index >= window_start]
    if trailing.empty:
        return {"name": name, "error": "no IC observations in the trailing 24m window"}

    gate_stats = spec.get("gate_stats") or {}
    train_ic = gate_stats.get("train_ic")
    if train_ic is None or train_ic == 0:
        return {"name": name, "error": "no gate_stats.train_ic to sign-adjust against"}
    accepted_sign = 1.0 if train_ic > 0 else -1.0

    adj = trailing * accepted_sign
    clusters = trailing.index.to_period("M").astype(str).to_numpy()
    stat = _cluster_mean_t(adj.to_numpy(), clusters)

    tstat = stat["t"]
    has_verdict = tstat == tstat  # not NaN
    demote = bool(has_verdict and tstat <= 0)

    evidence = {
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "rule": "trailing_24m_sign_adjusted_rank_ic_month_clustered_t_le_0",
        "window_start": str(window_start.date()),
        "window_end": str(latest_obs.date()),
        "horizon_days": horizon,
        "n_days": int(stat["n"]),
        "n_month_clusters": int(stat["n_clusters"]),
        "accepted_sign": int(accepted_sign),
        "mean_sign_adjusted_ic": float(adj.mean()),
        "month_clustered_t": float(tstat) if has_verdict else None,
        "demote": demote,
    }
    return {
        "name": name,
        "sleeves": list(spec.get("sleeves", [])),
        "horizon_days": horizon,
        "n_days": evidence["n_days"],
        "mean_ic": evidence["mean_sign_adjusted_ic"],
        "tstat": tstat,
        "has_verdict": has_verdict,
        "demote": demote,
        "evidence": evidence,
    }


def run(lake_root, start, end, apply: bool) -> int:
    lake = Lake(lake_root)
    bundle = load_bundle(lake, start, end)
    if "prices" not in bundle:
        print(f"no price data found in lake at {lake.root} — nothing to measure")
        return 0

    bundle = dict(bundle)
    bundle["prices"] = _dedupe_prices(bundle["prices"])
    prices = bundle["prices"]
    sleeve_map = _sleeve_map(prices)
    signal_registry = _load_signal_registry_safe()

    registry = FactorRegistry()
    accepted = registry.factors(status="accepted")
    if not accepted:
        print("no accepted factors in configs/factors.yaml — nothing to measure")
        return 0

    results = [measure_factor(name, spec, bundle, prices, sleeve_map, signal_registry)
              for name, spec in sorted(accepted.items())]

    _print_table(results)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    record = {
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "rule": "trailing_24m_sign_adjusted_rank_ic_month_clustered_t_le_0",
        "trailing_months": _TRAILING_MONTHS,
        "factors": {r["name"]: r.get("evidence", {"error": r.get("error")}) for r in results},
    }
    _DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _DIAGNOSTICS_DIR / f"factor_health_{run_id}.json"
    with open(out_path, "w") as f:
        json.dump(record, f, indent=2, allow_nan=False, default=str)
    print(f"\ndiagnostics written: {out_path}")

    if apply:
        changed = []
        for r in results:
            if r.get("demote"):
                registry.demote(r["name"], r["evidence"])
                changed.append(r["name"])
        if changed:
            registry.save()
            print(f"\nAPPLIED: demoted {changed} in {registry.path}")
        else:
            print("\nAPPLIED: no factor tripped the demotion rule — nothing changed")
    return 0


def _print_table(results: list[dict]) -> None:
    header = (f"{'factor':<18}{'n_days':>8}{'mean_IC':>11}{'month_t':>10}  DEMOTE")
    print(header)
    print("-" * len(header))
    for r in results:
        if r.get("error"):
            print(f"{r['name']:<18}  ERROR: {r['error']}")
            continue
        t_s = f"{r['tstat']:.3f}" if r["has_verdict"] else "n/a"
        verdict = "YES" if r["demote"] else ("n/a" if not r["has_verdict"] else "no")
        print(f"{r['name']:<18}{r['n_days']:>8}{r['mean_ic']:>11.5f}{t_s:>10}  {verdict}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="factor_health.py",
        description="Trailing-24m factor-health measurement (committed demotion rule).")
    p.add_argument("--apply", action="store_true",
                   help="demote every factor the rule flags and save configs/factors.yaml")
    p.add_argument("--start", default=None, help="lake load start (obs_date >=)")
    p.add_argument("--end", default=None, help="lake load end (obs_date <=)")
    p.add_argument("--lake-root", default=None, help="lake root dir (default: repo data/)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    return run(args.lake_root, args.start, args.end, args.apply)


if __name__ == "__main__":
    sys.exit(main())
