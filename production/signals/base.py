"""Signal abstract base, registry, and shared PIT-safe helpers.

Every signal is a *pure* point-in-time function of the input bundle: its value at
date D may depend only on input rows knowable by end of day D. For price-driven
signals that reduces to using each instrument's own trailing history (positional
`shift`/`rolling` on the sorted trading-day series). For lag-stamped sources (macro,
COT) it additionally means honouring `available_from`: a row is usable at D only if
`available_from <= end of day D (UTC)`. The corruption harness in
`tests/test_no_lookahead.py` is parametrized over the entire registry and mutates the
future tail of every input frame; any signal that leaks the future fails it.

Signals emit a long panel ``[obs_date, instrument_id, value]`` and only for
instruments whose sleeve (inferred from the ``instrument_id`` prefix) is in the
signal's declared ``sleeves`` — the caller may hand over a wider universe.
"""
from __future__ import annotations

import importlib
from abc import ABC, abstractmethod

import pandas as pd

OUTPUT_COLUMNS = ["obs_date", "instrument_id", "value"]

# instrument_id is `CLASS:SYMBOL:first-listing-date`; the class prefix fixes the sleeve.
_PREFIX_TO_SLEEVE = {
    "EQ": "equity",
    "CR": "crypto",
    "FX": "fx_etf",
    "CO": "commodity_etf",
}


def sleeve_from_id(instrument_id: str) -> str:
    """Map an ``instrument_id`` to its sleeve via the class prefix.

    ``EQ`` -> equity, ``CR`` -> crypto, ``FX`` -> fx_etf, ``CO`` -> commodity_etf.
    """
    prefix = instrument_id.split(":", 1)[0]
    try:
        return _PREFIX_TO_SLEEVE[prefix]
    except KeyError as exc:  # pragma: no cover - guards malformed ids
        raise ValueError(f"unknown instrument_id prefix in {instrument_id!r}") from exc


class Signal(ABC):
    """A PIT-pure alpha signal.

    Attributes mirror the ``factors.yaml`` entry for registered signals:
        name              registry key; must equal the factors.yaml key
        sleeves           sleeves this signal is defined on
        required_datasets subset of {"prices","funding","macro","cot"}
        min_history_days  no value emitted before instrument_start + this many days
        horizon_days      intended holding/scoring horizon
    """

    name: str
    sleeves: list[str]
    required_datasets: list[str]
    min_history_days: int
    horizon_days: int

    @abstractmethod
    def compute(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Return a long panel ``[obs_date, instrument_id, value]``.

        PURE point-in-time: the value at date D uses only inputs knowable by end of
        day D. Emits values only for instruments in ``self.sleeves`` (the caller may
        pass a wider universe). NaNs are dropped from the output.
        """
        raise NotImplementedError

    # ---------------------------------------------------------------- helpers
    def _restrict_to_sleeves(self, df: pd.DataFrame) -> pd.DataFrame:
        """Keep only rows whose instrument_id maps to one of this signal's sleeves."""
        if df.empty:
            return df
        wanted = set(self.sleeves)
        mask = df["instrument_id"].map(sleeve_from_id).isin(wanted)
        return df.loc[mask]

    def _finalize(self, out: pd.DataFrame, anchor: pd.DataFrame) -> pd.DataFrame:
        """Drop NaN values, enforce min_history, return canonical column order.

        ``anchor`` supplies each instrument's first observation date; no value is
        emitted before ``first_obs + min_history_days`` (calendar days). This makes
        the earliest emitted date deterministic regardless of whether an instrument
        trades on a 5-day (ETF/equity) or 7-day (crypto) calendar.
        """
        if out.empty:
            return pd.DataFrame(columns=OUTPUT_COLUMNS)
        out = out.dropna(subset=["value"])
        if out.empty:
            return pd.DataFrame(columns=OUTPUT_COLUMNS)
        first_obs = anchor.groupby("instrument_id")["obs_date"].min()
        floor = out["instrument_id"].map(first_obs) + pd.Timedelta(days=self.min_history_days)
        out = out.loc[out["obs_date"] >= floor]
        return (out[OUTPUT_COLUMNS]
                .sort_values(["obs_date", "instrument_id"], kind="stable")
                .reset_index(drop=True))


# ------------------------------------------------------------------- registry
SIGNAL_REGISTRY: dict[str, type[Signal]] = {}

# Modules imported (for @register side effects) by all_signals().
_SIGNAL_MODULES = ("momentum", "vol", "carry", "value", "positioning")


def register(cls: type[Signal]) -> type[Signal]:
    """Class decorator: register a Signal subclass under ``cls.name``."""
    if not isinstance(cls, type) or not issubclass(cls, Signal):
        raise TypeError(f"register expects a Signal subclass, got {cls!r}")
    name = getattr(cls, "name", None)
    if not name:
        raise ValueError(f"{cls.__name__} must define a non-empty `name`")
    existing = SIGNAL_REGISTRY.get(name)
    if existing is not None and existing is not cls:
        raise ValueError(f"duplicate signal name {name!r}: {existing} vs {cls}")
    SIGNAL_REGISTRY[name] = cls
    return cls


def all_signals() -> dict[str, type[Signal]]:
    """Import every signal module for its @register side effects; return the registry."""
    for module in _SIGNAL_MODULES:
        importlib.import_module(f"production.signals.{module}")
    return SIGNAL_REGISTRY
