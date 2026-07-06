"""Assemble and persist the backtest report.

``build_report`` distills a ``BacktestResult`` into a JSON-serializable dict of headline
statistics, per-sleeve breakdowns, factor/alpha attribution, and the honesty caveats that
must travel with every number (multiple-testing count behind the deflated Sharpe, the
borrow-cost-free assumption, and whether the promotion gate had actually been applied).
``write_report`` drops both a machine-readable ``.json`` and a human ``.txt`` into
``reports/`` (never ``data/``).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

from production.backtest.attribution import factor_attribution, per_alpha_contribution
from production.backtest.bootstrap import sharpe_ci
from production.backtest.deflated_sharpe import deflated_sharpe, probabilistic_sharpe
from production.backtest.metrics import (ann_return, ann_vol, hit_rate,
                                         information_ratio, max_drawdown, sharpe,
                                         turnover)
from production.core.config import REPO_ROOT


def _sleeve_table(result) -> dict:
    out: dict = {}
    for s in result.sleeve_returns.columns:
        r = result.sleeve_returns[s]
        wh = result.weights_history.get(s)
        out[s] = {
            "sharpe": sharpe(r),
            "ann_return": ann_return(r),
            "ann_vol": ann_vol(r),
            "max_drawdown": max_drawdown(r),
            "hit_rate": hit_rate(r),
            "turnover": turnover(wh) if wh is not None else 0.0,
        }
    return out


def build_report(result, cfg: dict, registry) -> dict:
    """Build the headline report dict from a completed ``BacktestResult``."""
    net = pd.Series(result.total_returns).astype(float).dropna()
    gross = pd.Series(result.total_gross_returns).astype(float).dropna()
    zero = pd.Series(0.0, index=net.index)

    n = len(net)
    sr_ann_net = sharpe(net)
    sr_ann_gross = sharpe(gross)
    # per-observation Sharpe + higher moments for the PSR/DSR algebra
    sr_pp = sharpe(net, freq=1)
    sk = float(skew(net)) if n > 2 else 0.0
    ku = float(kurtosis(net, fisher=False)) if n > 3 else 3.0

    # variance of the trial Sharpes: absent the actual trial cloud, use the (non-normal)
    # Sharpe-estimator variance as the standard deflation proxy — documented approximation.
    radicand = max(1.0 - sk * sr_pp + ((ku - 1.0) / 4.0) * sr_pp * sr_pp, 1e-12)
    var_trials_sr = radicand / max(n - 1, 1)

    n_trials = int(registry.n_trials)
    psr = probabilistic_sharpe(sr_pp, 0.0, n, sk, ku)          # prob true Sharpe > 0
    dsr = deflated_sharpe(sr_pp, n_trials, var_trials_sr, n, sk, ku)
    ci_lo, ci_hi = sharpe_ci(net)

    attribution = factor_attribution(
        net, result._factor_returns if result._factor_returns is not None else pd.DataFrame())
    per_alpha = per_alpha_contribution(
        result._weights_by_factor or {},
        result._inst_returns if result._inst_returns is not None else pd.DataFrame(net))

    headline = {
        "n_days": n,
        "start": str(net.index[0].date()) if n else None,
        "end": str(net.index[-1].date()) if n else None,
        "net_sharpe": sr_ann_net,
        "gross_sharpe": sr_ann_gross,
        "net_information_ratio": information_ratio(net, zero),
        "gross_information_ratio": information_ratio(gross, zero),
        "ann_return_net": ann_return(net),
        "ann_return_gross": ann_return(gross),
        "ann_vol_net": ann_vol(net),
        "max_drawdown": max_drawdown(net),
        "hit_rate": hit_rate(net),
        "avg_cost_drag_bps_per_day": float(pd.Series(result.costs).mean() * 1e4),
        "probabilistic_sharpe": psr,
        "deflated_sharpe": dsr,
        "n_trials": n_trials,
        "sharpe_ci95": [ci_lo, ci_hi],
    }

    report = {
        "headline": headline,
        "per_sleeve": _sleeve_table(result),
        "factor_attribution": attribution,
        "per_alpha_contribution": per_alpha,
        "ic_realized_vs_training": result._ic_realized_vs_training or {},
        "factors_used": result._factors_used or [],
        "caveats": {
            "gate_applied": bool(result._gate_applied),
            "borrow_cost_free": True,  # short financing / borrow costs are NOT modeled
            "warnings": list(result._warnings or []),
        },
        "config_echo": _echo_config(cfg),
    }
    return report


def _echo_config(cfg: dict) -> dict:
    """A shallow echo of the knobs that shaped the run (for reproducibility)."""
    keys = ("walk_forward", "optimizer", "sleeve_vol_target", "total_vol_target",
            "constraints", "allocation", "overlays")
    return {k: cfg[k] for k in keys if k in cfg}


def _coerce(o):
    """Make numpy / pandas scalars JSON-friendly (and turn NaN/inf into None)."""
    if isinstance(o, dict):
        return {str(k): _coerce(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_coerce(v) for v in o]
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return None if not np.isfinite(f) else f
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def _human_text(report: dict) -> str:
    h = report["headline"]
    lines = ["trader-improved — backtest report",
             "=" * 42,
             f"window        {h['start']} .. {h['end']}  ({h['n_days']} days)",
             f"net Sharpe    {_fmt(h['net_sharpe'])}   (gross {_fmt(h['gross_sharpe'])})",
             f"net IR vs 0   {_fmt(h['net_information_ratio'])}",
             f"ann return    {_fmt(h['ann_return_net'])}   ann vol {_fmt(h['ann_vol_net'])}",
             f"max drawdown  {_fmt(h['max_drawdown'])}   hit rate {_fmt(h['hit_rate'])}",
             f"cost drag     {_fmt(h['avg_cost_drag_bps_per_day'])} bps/day",
             f"PSR (>0)      {_fmt(h['probabilistic_sharpe'])}",
             f"deflated SR   {_fmt(h['deflated_sharpe'])}   (n_trials={h['n_trials']})",
             f"Sharpe 95% CI [{_fmt(h['sharpe_ci95'][0])}, {_fmt(h['sharpe_ci95'][1])}]",
             "",
             "per sleeve:"]
    for s, m in report["per_sleeve"].items():
        lines.append(f"  {s:<14} Sharpe {_fmt(m['sharpe'])}  ann {_fmt(m['ann_return'])}"
                     f"  vol {_fmt(m['ann_vol'])}  DD {_fmt(m['max_drawdown'])}"
                     f"  turnover {_fmt(m['turnover'])}")
    lines.append("")
    lines.append("factor attribution (annualized contribution):")
    for f, c in report["factor_attribution"].get("contributions", {}).items():
        lines.append(f"  {f:<14} {_fmt(c)}")
    lines.append(f"  specific      {_fmt(report['factor_attribution'].get('specific'))}")
    lines.append("")
    cav = report["caveats"]
    if not cav["gate_applied"]:
        lines.append("WARNING: gate NOT applied — trading candidate factors.")
    lines.append("caveat: borrow / short-financing costs are NOT modeled.")
    for w in cav["warnings"]:
        lines.append(f"  - {w}")
    return "\n".join(lines) + "\n"


def _fmt(x) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "  n/a"
    return f"{float(x):+.3f}"


def write_report(report: dict, out_dir: str | Path = "reports") -> Path:
    """Write ``backtest_<UTCtimestamp>.json`` and a companion ``.txt``; return the JSON path."""
    out = Path(out_dir)
    if not out.is_absolute():
        out = REPO_ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = out / f"backtest_{ts}.json"
    txt_path = out / f"backtest_{ts}.txt"
    with open(json_path, "w") as f:
        json.dump(_coerce(report), f, indent=2, allow_nan=False)
    with open(txt_path, "w") as f:
        f.write(_human_text(report))
    return json_path
