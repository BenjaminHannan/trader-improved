"""Stage-2 FRED/ALFRED nowcast series (WEI, GDPNOW, CFNAI).

These are the three weekly/monthly activity nowcasts we add in Stage 2. They live in
ALFRED just like the Stage-1 rates and spreads, so the point-in-time story is identical
and exact: `realtime_start` on each observation is the moment that vintage first became
public, and we stamp `available_from` from it. A revised GDPNow print therefore never
leaks backwards — the prior vintage is what a same-day decision saw.

We reuse `FredAlfredLoader` wholesale; the only thing that changes is the default series
list. The alias map is identity (WEI<-WEI, GDPNOW<-GDPNOW, CFNAI<-CFNAI) because these
mnemonics are already human-readable, so the inherited transform (which falls back to
the raw id when a mnemonic is absent from the Stage-1 ALIAS table) emits them unchanged.
"""
from __future__ import annotations

from production.data.loaders.fred_alfred import FredAlfredLoader

# Identity aliases: these FRED mnemonics are already the internal series_ids we want.
STAGE2_ALIAS = {"WEI": "WEI", "GDPNOW": "GDPNOW", "CFNAI": "CFNAI"}


class FredStage2Loader(FredAlfredLoader):
    """ALFRED nowcasts: Weekly Economic Index, GDPNow, Chicago Fed NAI."""

    source = "fred:alfred:stage2"

    def __init__(self, lake=None, instruments=None, series=None, api_key=None):
        super().__init__(lake, instruments,
                         series=series or list(STAGE2_ALIAS.keys()),
                         api_key=api_key)
