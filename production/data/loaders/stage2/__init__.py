"""Stage-2 macro / sentiment loaders.

Each loader emits macro-style SERIES rows [obs_date, series_id, value, +mandatory]
under asset_class "macro". The load-bearing decision in every one of these is the
availability lag — sentiment and nowcast series are notorious look-ahead traps
because their headline number is revised, or is published a day or more after the
observation it is dated to. The availability rule on each class pins that lag
explicitly so a backtest can never see a value before it was knowable.
"""
from __future__ import annotations

from production.data.loaders.stage2.ads import AdsLoader
from production.data.loaders.stage2.aaii_manual import AaiiManualLoader
from production.data.loaders.stage2.cboe_putcall import CboePutCallLoader
from production.data.loaders.stage2.finra_short import FinraShortInterestLoader
from production.data.loaders.stage2.fred_stage2 import FredStage2Loader
from production.data.loaders.stage2.naaim import NaaimLoader

__all__ = [
    "AdsLoader",
    "AaiiManualLoader",
    "CboePutCallLoader",
    "FinraShortInterestLoader",
    "FredStage2Loader",
    "NaaimLoader",
]
