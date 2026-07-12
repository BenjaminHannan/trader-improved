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

MONTHLY vintages (probed live 2026-07-10): the sibling ``nowcast_month.xlsx`` file
has the same FusionCharts structure with one node per TARGET MONTH (subcaption
``YYYY-M``, 2013-7 onward) holding that month's daily as-published MoM nowcast path.
Because two target months are nowcast simultaneously on any publication day (the
prior month until its release + the current month), the target month is encoded in
the series id — ``CLEV_NOWCAST_CPI_MOM:2025-07`` — so vintages for different target
months never collide on the lake's (obs_date, series_id) key. These are the join
partners for the Kalshi MoM inflation ladders (KXCPI/KXCPICORE/KXPCECORE).
"""
from __future__ import annotations

import json

import pandas as pd

from production.data.base import AvailabilityRule, BaseLoader, IngestError

NOWCAST_URL = ("https://www.clevelandfed.org/-/media/files/webcharts/"
               "inflationnowcasting/nowcast_quarter.xlsx")
NOWCAST_MONTH_URL = ("https://www.clevelandfed.org/-/media/files/webcharts/"
                     "inflationnowcasting/nowcast_month.xlsx")
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


def _label_to_date_monthly(label: str, m_start: pd.Timestamp,
                           m_end: pd.Timestamp) -> pd.Timestamp | None:
    """Resolve an ``MM/DD`` label near its target month (see _transform_month)."""
    try:
        mm, dd = (int(x) for x in str(label).strip().split("/"))
    except (ValueError, AttributeError):
        return None
    for year in (m_start.year - 1, m_start.year, m_start.year + 1):
        try:
            cand = pd.Timestamp(year, mm, dd)
        except ValueError:
            continue
        if m_start - pd.Timedelta(days=75) <= cand <= m_end + pd.Timedelta(days=60):
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

    def __init__(self, lake=None, instruments=None, url=NOWCAST_URL,
                 month_url=NOWCAST_MONTH_URL):
        super().__init__(lake, instruments)
        self.url = url
        self.month_url = month_url

    def fetch(self, start, end) -> dict:
        import requests

        resp = requests.get(self.url, headers={"User-Agent": _UA}, timeout=60)
        resp.raise_for_status()
        try:
            quarter = resp.json()
        except ValueError as exc:
            raise IngestError(
                f"cleveland nowcast: expected FusionCharts JSON, got "
                f"{resp.text[:80]!r}") from exc
        # The monthly file degrades instead of raising: the scheduled quarterly
        # archiver must never be sunk by the research-grade monthly panel.
        month: list = []
        try:
            mresp = requests.get(self.month_url, headers={"User-Agent": _UA},
                                 timeout=60)
            mresp.raise_for_status()
            month = mresp.json()
        except Exception as exc:
            self.warnings.append(f"cleveland monthly nowcast fetch failed: {exc!r}")
        return {"quarter": quarter, "month": month}

    def transform(self, raw) -> pd.DataFrame:
        # Backward compatible: a bare list is the historical quarterly payload.
        if isinstance(raw, list):
            return self._transform_quarter(raw)
        frames = [self._transform_quarter((raw or {}).get("quarter") or []),
                  self._transform_month((raw or {}).get("month") or [])]
        frames = [f for f in frames if not f.empty]
        if not frames:
            return self._transform_quarter([])
        return pd.concat(frames, ignore_index=True)

    def _transform_quarter(self, raw) -> pd.DataFrame:
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

    def _transform_month(self, raw) -> pd.DataFrame:
        """Monthly MoM vintages, target month encoded in the series id.

        Subcaption is ``YYYY-M``; labels run from ~2 months before the target month
        through its mid-next-month release, so a January label on a December node
        belongs to the FOLLOWING year — candidates y-1/y/y+1 are disambiguated by a
        [-75d, +60d] window around the target month. No cross-node dedup: each
        (target month, series) pair is its own vintage series by construction.
        """
        cols = ["obs_date", "series_id", "value", "asset_class"]
        rows: list[dict] = []
        for node in raw or []:
            chart = node.get("chart", {})
            sub = str(chart.get("subcaption", ""))       # e.g. "2025-7"
            if "-" not in sub:
                continue
            y_s, m_s = sub.split("-", 1)
            try:
                t_year, t_month = int(y_s), int(m_s)
            except ValueError:
                continue
            if not 1 <= t_month <= 12:
                continue
            m_start = pd.Timestamp(t_year, t_month, 1)
            m_end = m_start + pd.DateOffset(months=1) - pd.Timedelta(days=1)
            tag = f"{t_year}-{t_month:02d}"
            cats = (node.get("categories") or [{}])[0].get("category", [])
            labels = [c.get("label") for c in cats]
            for series in node.get("dataset", []):
                base_sid = SERIES_MAP.get(str(series.get("seriesname", "")).strip())
                if base_sid is None:                     # "Actual ..." markers
                    continue
                sid = f"{base_sid}_MOM:{tag}"
                for label, point in zip(labels, series.get("data", [])):
                    val = point.get("value") if isinstance(point, dict) else None
                    if val in (None, ""):
                        continue
                    d = _label_to_date_monthly(label, m_start, m_end)
                    if d is None:
                        continue
                    try:
                        rows.append({"obs_date": d, "series_id": sid,
                                     "value": float(val), "asset_class": "macro"})
                    except (TypeError, ValueError):
                        continue
        if not rows:
            return pd.DataFrame(columns=cols)
        return (pd.DataFrame(rows)[cols]
                .drop_duplicates(subset=["obs_date", "series_id"], keep="last")
                .sort_values(["series_id", "obs_date"])
                .reset_index(drop=True))
