"""Prediction-market (event) subsystem.

A self-contained sleeve for binary event contracts (Kalshi, Polymarket). Two curated
loaders land a single ``event_markets`` dataset; :mod:`production.events.markets`
builds a liquid, de-correlated universe from it; :mod:`production.events.signals`
carries two documented edges (favorite-longshot bias, resolution convergence); and
:mod:`production.events.sizing` turns signals into fractional-Kelly weights with a
cost haircut and per-group / gross caps.

Integration into the multi-sleeve engine is a later wave — nothing here imports the
engine or allocation, and the loaders write through the same PIT lake pipeline every
other dataset uses (``available_from`` = ingest time for these snapshot pulls).
"""
from __future__ import annotations

from production.events.markets import EventMarket, dedupe_related, liquid_universe
from production.events.signals import longshot_bias, resolution_convergence
from production.events.sizing import (
    bernoulli_variance,
    kelly_fraction,
    size_event_book,
)

__all__ = [
    "EventMarket",
    "liquid_universe",
    "dedupe_related",
    "longshot_bias",
    "resolution_convergence",
    "bernoulli_variance",
    "kelly_fraction",
    "size_event_book",
]
