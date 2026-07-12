"""Stage-3 alternative-data loaders — documented placeholders.

Each class is a real `BaseLoader` subclass so the ingest wiring, availability
vocabulary, and audit expectations are pinned now, but `fetch`/`transform` raise
`NotImplementedError` until Stage 3. The docstring on each states the availability
rule it WILL use, because the publication lag is the load-bearing design decision and
belongs in the code the moment the source is chosen — not deferred to implementation.
"""
from __future__ import annotations

import pandas as pd

from production.data.base import AvailabilityRule, BaseLoader


class _Stub(BaseLoader):
    """Common stub behaviour: refuse to fetch until Stage 3."""

    def fetch(self, start, end):
        raise NotImplementedError(f"{type(self).__name__} is a Stage-3 stub")

    def transform(self, raw) -> pd.DataFrame:
        raise NotImplementedError(f"{type(self).__name__} is a Stage-3 stub")


class ENTSOELoader(_Stub):
    """European power generation/load (ENTSO-E transparency platform).

    Availability rule: obs_offset ~ 1 hour after the settlement period, since ENTSO-E
    publishes near-real-time with a short lag. Emitted as macro series.
    """
    dataset = "power"
    vendor = "entsoe"
    availability_rule = AvailabilityRule("obs_offset", {"offset": pd.Timedelta(hours=1)})


class JODILoader(_Stub):
    """JODI global oil/gas supply-demand.

    Availability rule: obs_offset of ~60 days — JODI monthly data is released with a
    roughly two-month reporting lag; stamp conservatively at obs_date + 60 days.
    """
    dataset = "energy_balance"
    vendor = "jodi"
    availability_rule = AvailabilityRule("obs_offset", {"offset": pd.Timedelta(days=60)})


class VIIRSLoader(_Stub):
    """VIIRS nighttime-lights / flaring radiance (NASA/NOAA).

    Availability rule: obs_offset of ~1 day — the daily granule is processed and
    published roughly a day after the overpass.
    """
    dataset = "nightlights"
    vendor = "viirs"
    availability_rule = AvailabilityRule("obs_offset", {"offset": pd.Timedelta(days=1)})


class OpenSkyLoader(_Stub):
    """OpenSky flight-traffic aggregates.

    Availability rule: ingest_time — state vectors are queried live; a daily
    aggregate is knowable only when the snapshot is taken.
    """
    dataset = "flights"
    vendor = "opensky"
    availability_rule = AvailabilityRule("ingest_time")


class ZillowLoader(_Stub):
    """Zillow housing indices (ZHVI/ZORI).

    Availability rule: obs_offset of ~16 days — monthly indices publish mid-month for
    the prior month; stamp obs_date + ~16 days.
    """
    dataset = "housing"
    vendor = "zillow"
    availability_rule = AvailabilityRule("obs_offset", {"offset": pd.Timedelta(days=16)})


class RedfinLoader(_Stub):
    """Redfin housing-market weekly metrics.

    Availability rule: next_weekday_time — Redfin's weekly data drops on a fixed
    weekday; stamp the first such weekday on/after obs_date.
    """
    dataset = "housing"
    vendor = "redfin"
    availability_rule = AvailabilityRule("next_weekday_time",
                                         {"weekday": 2, "hour": 12, "minute": 0})


class GoogleTrendsLoader(_Stub):
    """Google Trends search interest.

    Availability rule: obs_offset of ~2 days — Trends finalizes daily values with a
    short revision lag; stamp conservatively at obs_date + 2 days.
    """
    dataset = "search_trends"
    vendor = "google_trends"
    availability_rule = AvailabilityRule("obs_offset", {"offset": pd.Timedelta(days=2)})


class WikipediaPageviewsLoader(_Stub):
    """Wikipedia pageview counts (Wikimedia REST API).

    Availability rule: obs_offset of ~1 day — daily pageview aggregates are published
    the following day.
    """
    dataset = "pageviews"
    vendor = "wikipedia"
    availability_rule = AvailabilityRule("obs_offset", {"offset": pd.Timedelta(days=1)})
