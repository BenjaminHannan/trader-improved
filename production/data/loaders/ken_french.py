"""Fama-French daily factor returns downloaded directly from Ken French's data library.

VALIDATION-ONLY dataset. These well-known factor returns are never traded and never
fed to a live signal — they exist so `risk/validate_against_french` can sanity-check
that our internally estimated equity factor returns correlate with the canonical
academic series. Vendor values are in percent; we store decimal returns (divide by
100). series_id: FF_MKT_RF, FF_SMB, FF_HML, FF_RF, FF_MOM.

We fetch the two published CSV zips ourselves (requests + zipfile + pandas) rather than
going through pandas_datareader's `famafrench` reader, which raises a brittle
``'>' not supported between instances of 'str' and 'int'`` when Dartmouth tweaks the
file preamble. Parsing the raw CSV — skip the preamble until the daily date rows, stop
at the first blank/footer line, divide the percent values by 100 — is both simpler and
robust to those layout wobbles.

Availability: the library refreshes with a lag of a few business days; we stamp a
conservative `obs_date + 7 calendar days` (>= 5 business days) so no validation date
could ever leak forward.
"""
from __future__ import annotations

import re

import pandas as pd

from production.data.base import AvailabilityRule, BaseLoader

# Vendor column -> internal series_id.
COL_MAP = {
    "Mkt-RF": "FF_MKT_RF",
    "SMB": "FF_SMB",
    "HML": "FF_HML",
    "RF": "FF_RF",
    "Mom": "FF_MOM",
}

# Published daily CSV zips (keyless, direct from Dartmouth).
FACTORS_URL = ("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
               "F-F_Research_Data_Factors_daily_CSV.zip")
MOMENTUM_URL = ("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
                "F-F_Momentum_Factor_daily_CSV.zip")

# A daily observation row starts with an 8-digit YYYYMMDD date in the first column.
_DATE_RE = re.compile(r"^\s*\d{8}\s*$")


def parse_french_csv(text: str) -> pd.DataFrame:
    """Parse a Ken French daily factor CSV into a DataFrame indexed by date.

    The files carry a multi-line preamble, then a header line naming the factor
    columns (its first column — the date — is blank), then the daily rows, then a
    blank line and a copyright footer. We keep the header nearest the data block, read
    the contiguous run of date rows, and stop at the first non-date line after them.
    Values are left in percent; the caller divides by 100.
    """
    header: list[str] | None = None
    data: list[list[str]] = []
    started = False
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if _DATE_RE.match(parts[0]):
            started = True
            data.append(parts)
        elif started:
            break  # first blank/footer line after the daily block ends it
        elif len(parts) > 1 and any(parts[1:]):
            header = parts  # remember the closest header-like line before the data
    if not data:
        return pd.DataFrame()

    ncols = len(data[0]) - 1
    if header and len(header) >= ncols + 1:
        fields = header[1:ncols + 1]
    else:  # header lost in a layout change: fall back to positional names
        fields = [f"c{i}" for i in range(ncols)]
    dates = pd.DatetimeIndex([pd.Timestamp(row[0]) for row in data])
    frame = {}
    for i, name in enumerate(fields):
        frame[name] = pd.to_numeric(
            pd.Series([row[i + 1] for row in data]), errors="coerce").to_numpy()
    return pd.DataFrame(frame, index=dates)


class KenFrenchLoader(BaseLoader):
    dataset = "french"
    vendor = "ken_french"
    source = "ken_french:factors"
    asset_classes = ["macro"]
    default_asset_class = "macro"
    availability_rule = AvailabilityRule("obs_offset", {"offset": pd.Timedelta(days=7)})
    expectations = {
        "columns": ["value"],
        "ranges": {"value": (-1.0, 1.0)},  # daily decimal returns
        "max_null_frac": 0.02,
        "min_rows": 1,
    }

    def fetch(self, start, end) -> dict[str, pd.DataFrame]:
        import io
        import zipfile

        import requests

        s, e = pd.Timestamp(start), pd.Timestamp(end)
        out: dict[str, pd.DataFrame] = {}
        for key, url in (("factors", FACTORS_URL), ("momentum", MOMENTUM_URL)):
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                name = zf.namelist()[0]           # single CSV per zip
                text = zf.read(name).decode("latin-1")
            table = parse_french_csv(text)
            if not table.empty:  # clip to the requested window
                table = table[(table.index >= s) & (table.index <= e)]
            out[key] = table
        return out

    def transform(self, raw) -> pd.DataFrame:
        frames = []
        for _, table in raw.items():
            idx = pd.to_datetime(table.index)
            for col in table.columns:
                sid = COL_MAP.get(str(col).strip())
                if sid is None:
                    continue
                frames.append(pd.DataFrame({
                    "obs_date": idx,
                    "series_id": sid,
                    "value": pd.to_numeric(table[col], errors="coerce").to_numpy() / 100.0,
                }))
        if not frames:
            return pd.DataFrame(columns=["obs_date", "series_id", "value", "asset_class"])
        out = pd.concat(frames, ignore_index=True).dropna(subset=["value"])
        out["asset_class"] = "macro"
        return out
