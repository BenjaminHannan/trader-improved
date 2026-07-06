"""Typed YAML config loading + validation.

All configs live in configs/. Loading is centralized here so every module gets the
same validated view and missing keys fail loudly at load time, not deep in a backtest.
"""
from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"

SLEEVES = ("equity", "crypto", "fx_etf", "commodity_etf")


class ConfigError(Exception):
    pass


def _require(cfg: dict, keys: list[str], name: str) -> None:
    missing = [k for k in keys if k not in cfg]
    if missing:
        raise ConfigError(f"{name}: missing required keys {missing}")


@functools.lru_cache(maxsize=None)
def load_yaml(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.is_absolute():
        p = CONFIG_DIR / p
    if not p.exists():
        raise ConfigError(f"config file not found: {p}")
    with open(p) as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ConfigError(f"{p}: top level must be a mapping")
    return cfg


def universe_config() -> dict[str, Any]:
    cfg = load_yaml("universe.yaml")
    _require(cfg, ["sleeves"], "universe.yaml")
    unknown = set(cfg["sleeves"]) - set(SLEEVES)
    if unknown:
        raise ConfigError(f"universe.yaml: unknown sleeves {unknown}")
    return cfg


def factors_config() -> dict[str, Any]:
    cfg = load_yaml("factors.yaml")
    _require(cfg, ["gate", "factors", "n_trials"], "factors.yaml")
    _require(
        cfg["gate"],
        ["train_ic_tstat_min", "oos_ic_abs_min", "oos_fraction",
         "min_decay_halflife_days", "require_net_positive_validation"],
        "factors.yaml:gate",
    )
    for name, spec in cfg["factors"].items():
        _require(spec, ["signal", "sleeves", "horizon_days", "min_history_days", "status"],
                 f"factors.yaml:factors.{name}")
        if spec["status"] not in ("candidate", "accepted", "rejected"):
            raise ConfigError(f"factors.yaml: {name} has invalid status {spec['status']!r}")
    return cfg


def costs_config() -> dict[str, Any]:
    cfg = load_yaml("costs.yaml")
    _require(cfg, ["impact_alpha", "adv_window_days", "cap_bps", "sleeves"], "costs.yaml")
    for sleeve, spec in cfg["sleeves"].items():
        _require(spec, ["floor_bps", "half_spread_bps"], f"costs.yaml:sleeves.{sleeve}")
        if spec["floor_bps"] <= 0:
            raise ConfigError(f"costs.yaml: {sleeve} floor must be > 0 — costs are never zero")
    return cfg


def risk_config() -> dict[str, Any]:
    cfg = load_yaml("risk.yaml")
    _require(cfg, ["structural_sleeves", "covariance_sleeves", "exposures",
                   "factor_covariance", "specific_risk", "instrument_covariance"], "risk.yaml")
    return cfg


def backtest_config() -> dict[str, Any]:
    cfg = load_yaml("backtest.yaml")
    _require(cfg, ["walk_forward", "optimizer", "sleeve_vol_target", "total_vol_target",
                   "constraints", "allocation", "overlays"], "backtest.yaml")
    return cfg
