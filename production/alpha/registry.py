"""The factor registry and the promotion gate.

Every factor ever tested lives forever in ``configs/factors.yaml`` — that file is the
multiple-testing ledger, and ``n_trials`` on it feeds the deflated Sharpe downstream.
This module reads and mutates that file (never through the cached ``load_yaml`` — the
registry owns a private, mutable copy).

A candidate is promoted to *accepted* only if it clears every gate on data strictly
before the out-of-sample slice:
1. train-window rank-IC t-stat ``|t| >= train_ic_tstat_min``;
2. OOS-slice IC has the same sign as train IC and ``|IC_oos| >= oos_ic_abs_min``;
3. IC decay half-life ``>= min_decay_halflife_days`` (must survive to the next rebalance);
4. net-of-cost single-factor validation return ``> 0`` (when required).
Failing any gate rejects the factor. Thresholds come from the yaml ``gate`` block, not
from code — the discipline is configuration, not a magic number buried here.
"""
from __future__ import annotations

import importlib
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import yaml

from production.core.config import CONFIG_DIR


@dataclass
class GateStats:
    train_ic: float
    train_tstat: float
    oos_ic: float
    decay_halflife_days: float
    net_validation_return: float
    n_dates: int
    # Annualized net single-factor Sharpe on the validation slice — the per-trial Sharpe the
    # deflated-Sharpe var_trials is estimated from. Defaulted so old factors.yaml files (which
    # never recorded it) still load and reconstruct cleanly.
    val_sharpe: float = float("nan")


@dataclass
class GateVerdict:
    passed: bool
    reasons: list[str]
    stats: GateStats


class FactorRegistry:
    """Read/mutate ``configs/factors.yaml`` and apply the promotion gate."""

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path is not None else CONFIG_DIR / "factors.yaml"
        with open(self.path) as f:
            self.cfg = yaml.safe_load(f)

    # ------------------------------------------------------------------ views
    def factors(self, status: str | None = None) -> dict[str, dict]:
        """All factor specs, optionally filtered to a single ``status``."""
        facs = self.cfg["factors"]
        if status is None:
            return dict(facs)
        return {k: v for k, v in facs.items() if v.get("status") == status}

    @property
    def n_trials(self) -> int:
        return int(self.cfg.get("n_trials", 0))

    def signal_class(self, name: str) -> type:
        """Import-resolve the dotted ``signal:`` path for a factor to its class object."""
        dotted = self.cfg["factors"][name]["signal"]
        module_path, _, cls_name = dotted.rpartition(".")
        module = importlib.import_module(module_path)
        return getattr(module, cls_name)

    # ------------------------------------------------------------------- gate
    def gate(self, name: str, stats: GateStats) -> GateVerdict:
        g = self.cfg["gate"]
        reasons: list[str] = []
        passed = True

        tstat_min = float(g["train_ic_tstat_min"])
        if not (abs(stats.train_tstat) >= tstat_min):
            passed = False
            reasons.append(
                f"train IC t-stat |{stats.train_tstat:.3f}| < {tstat_min}")

        oos_min = float(g["oos_ic_abs_min"])
        same_sign = (stats.train_ic != 0 and stats.oos_ic != 0
                     and np.sign(stats.oos_ic) == np.sign(stats.train_ic))
        if not same_sign:
            passed = False
            reasons.append(
                f"OOS IC {stats.oos_ic:.4f} sign disagrees with train IC "
                f"{stats.train_ic:.4f}")
        if not (abs(stats.oos_ic) >= oos_min):
            passed = False
            reasons.append(f"|OOS IC| {abs(stats.oos_ic):.4f} < {oos_min}")

        hl_min = float(g["min_decay_halflife_days"])
        if not (stats.decay_halflife_days >= hl_min):
            passed = False
            reasons.append(
                f"decay half-life {stats.decay_halflife_days:.2f}d < {hl_min}d")

        if bool(g.get("require_net_positive_validation", False)):
            if not (stats.net_validation_return > 0):
                passed = False
                reasons.append(
                    f"net validation return {stats.net_validation_return:.5f} <= 0")

        if passed:
            reasons.append("all gates passed")
        return GateVerdict(passed=passed, reasons=reasons, stats=stats)

    # ----------------------------------------------------------------- record
    def record(self, name: str, verdict: GateVerdict) -> None:
        """Set status accepted/rejected, store gate stats, bump the trial ledger."""
        if name not in self.cfg["factors"]:
            raise KeyError(f"unknown factor {name!r}")
        spec = self.cfg["factors"][name]
        spec["status"] = "accepted" if verdict.passed else "rejected"
        spec["gate_stats"] = asdict(verdict.stats)
        self.cfg["n_trials"] = int(self.cfg.get("n_trials", 0)) + 1

    def save(self, path: str | Path | None = None) -> None:
        """Write the (possibly mutated) config back to yaml, preserving key order."""
        out = Path(path) if path is not None else self.path
        with open(out, "w") as f:
            yaml.safe_dump(self.cfg, f, sort_keys=False)
