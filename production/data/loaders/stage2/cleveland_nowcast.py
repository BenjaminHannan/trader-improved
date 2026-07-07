"""Cleveland Fed daily inflation nowcasts — with the AS-PUBLISHED vintage history.

The nowcasting page's FusionCharts data file (served at an ``.xlsx`` URL but actually
JSON) carries one chart node per quarter back to 2013:Q3, each holding the DAILY
nowcast path for that quarter (CPI / Core CPI / PCE / Core PCE) — i.e. the
as-published vintage history that the 2026-07-07 research sweep concluded had no
public archive. Each label is the publication date of that day's nowcast.

Vintage semantics: ``obs_date`` = the publication date; ``available_from`` =
publication day 15:00 UTC (the Fed posts ~10:00 ET). A nowcast is a forecast OF the
quarter, knowable the day it is published — never joined by the target quarter.

CAVEAT (documented, not yet verified): we treat the charted per-day values as
as-published vintages. The Fed's EC 2023-06 evaluates as-published accuracy from an
internal archive, and the chart has always displayed the daily path, so re-running
history would be pointless for them — but until a captured live value is compared
against the same date's value in a later file pull, treat gated uses as
``vintage_quality=believed-as-published``. The daily scheduled re-pull creates
exactly that check over time.

Series ids: ``CLEV_NOWCAST_CPI``, ``CLEV_NOWCAST_CORECPI``, ``CLEV_NOWCAST_PCE``,
``CLEV_NOWCAST_COREPCE`` (quarterly-annualized percent change for the then-current
target quarter).
"""
from __future__ import annotations

import json

import pandas as pd

from production.data.base import AvailabilityRule, BaseLoader, IngestError

NOWCAST_URL = ("https://www.clevelandfed.org/-/media/files/webcharts/"
               "inflationnowcasting/nowcast_quarter.xlsx")
_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")

SERIES_MAP = {
    "CPI Inflation": "CLEV_NOWCAST_CPI",
    "Core CPI Inflation": "CLEV_NOWCAST_CORECPI",
    "PCE Inflation": "CLEV_NOWCAST_PCE",
    "Core PCE Inflation": "CLEV_NOWCAST_COREPCE",
}

_QUARTER_START_MONTH = {"Q1": 1, "Q2": 4, "Q3": 7, "Q4": 10}


def _label_to_date(label: str, q_year: int, q_num: str) -> pd.Timestamp | None:
    """Resolve an ``MM/DD`` chart label to a full date near its quarter.

    Labels run from ~6 weeks before the target quarter starts through its end, so
    a December label on a Q1 chart belongs to the PRIOR calendar year. Choose the
    candidate year that lands inside [quarter_start - 60d, quarter_end + 10d].
    """
    try:
        mm, dd = (int(x) for x in str(label).strip().split("/"))
    except (ValueError, AttributeError):
        return None
    q_start = pd.Timestamp(q_year, _QUARTER_START_MONTH[q_num], 1)
    q_end = q_start + pd.DateOffset(months=3) - pd.Timedelta(days=1)
    for year in (q_year, q_year - 1):
        try:
            cand = pd.Timestamp(year, mm, dd)
        except ValueError:  # e.g. 02/29 on a non-leap candidate year
            continue
        if q_start - pd.Timedelta(days=60) <= cand <= q_end + pd.Timedelta(days=10):
            return cand
    return None


class ClevelandNowcastLoader(BaseLoader):
    dataset = "macro"
    vendor = "cleveland_fed"
    source = "cleveland_fed:nowcast"
    asset_classes = ["macro"]
    default_asset_class = "macro"
    # Published ~10:00 ET; stamp same-day 15:00 UTC (conservative post-publication).
    availability_rule = AvailabilityRule("obs_offset", {"offset": pd.Timedelta(hours=15)})
    expectations = {
        "columns": ["value"],
        "ranges": {"value": (-25.0, 50.0)},   # annualized % change; sane macro band
        "max_null_frac": 0.02,
        "min_rows": 1,
    }

    def __init__(self, lake=None, instruments=None, url=NOWCAST_URL):
        super().__init__(lake, instruments)
        self.url = url

    def fetch(self, start, end) -> list:
        import requests

        resp = requests.get(self.url, headers={"User-Agent": _UA}, timeout=60)
        resp.raise_for_status()
        try:
            return resp.json()
        except ValueError as exc:
            raise IngestError(
                f"cleveland nowcast: expected FusionCharts JSON, got "
                f"{resp.text[:80]!r}") from exc

    def transform(self, raw) -> pd.DataFrame:
        cols = ["obs_date", "series_id", "value", "asset_class"]
        rows: list[dict] = []
        for node in raw or []:
            chart = node.get("chart", {})
            sub = str(chart.get("subcaption", ""))       # e.g. "2013:Q3"
            if ":" not in sub:
                continue
            q_year_s, q_num = sub.split(":", 1)
            try:
                q_year = int(q_year_s)
            except ValueError:
                continue
            if q_num not in _QUARTER_START_MONTH:
                continue
            cats = (node.get("categories") or [{}])[0].get("category", [])
            labels = [c.get("label") for c in cats]
            for series in node.get("dataset", []):
                sid = SERIES_MAP.get(str(series.get("seriesname", "")).strip())
                if sid is None:      # "Actual ..." markers etc. — actuals live in ALFRED
                    continue
                for label, point in zip(labels, series.get("data", [])):
                    val = point.get("value") if isinstance(point, dict) else None
                    if val in (None, ""):
                        continue
                    d = _label_to_date(label, q_year, q_num)
                    if d is None:
                        continue
                    try:
                        rows.append({"obs_date": d, "series_id": sid,
                                     "value": float(val), "asset_class": "macro",
                                     "_q_start": pd.Timestamp(
                                         q_year, _QUARTER_START_MONTH[q_num], 1)})
                    except (TypeError, ValueError):
                        continue
        if not rows:
            return pd.DataFrame(columns=cols)
        df = pd.DataFrame(rows)
        # Quarter charts overlap at transitions (the old quarter's tail and the new
        # quarter's ramp are published side by side); on those days keep the NEWER
        # target quarter — that is "the current-quarter nowcast" a reader saw.
        # Sort on the explicit quarter start so resolution never depends on the
        # file's node order.
        df = (df.sort_values(["series_id", "obs_date", "_q_start"], kind="stable")
                .drop_duplicates(subset=["obs_date", "series_id"], keep="last"))
        return (df[cols].sort_values(["series_id", "obs_date"])
                  .reset_index(drop=True))
