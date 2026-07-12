"""Macro series from FRED/ALFRED with point-in-time vintages.

The whole reason to use ALFRED (Archival FRED) rather than plain FRED is vintages:
`realtime_start` tells us exactly when each observation *first became public*, so a
revised GDP print doesn't leak backwards into a backtest. When an API key is present
we pull the ALFRED observations endpoint and stamp `available_from` explicitly from
`realtime_start`. Without a key we degrade to the keyless fredgraph CSV, which carries
NO vintage information — so we stamp a deliberately conservative `obs_date + 1 day`
and raise an audit warning so the loss of PIT fidelity is never silent.

series_id carries the INTERNAL alias (RATE_EU, DGS3MO_US, ...), decoupling signal code
from FRED's opaque mnemonics.
"""
from __future__ import annotations

import os

import pandas as pd

from production.data.base import AvailabilityRule, BaseLoader

# FRED mnemonic -> internal alias. RATE_* are 3-month interbank rates per country;
# DGS3MO -> DGS3MO_US; the credit-spread and vol series keep their FRED names.
ALIAS = {
    "IR3TIB01EZM156N": "RATE_EU",
    "IR3TIB01JPM156N": "RATE_JP",
    "IR3TIB01GBM156N": "RATE_GB",
    "IR3TIB01AUM156N": "RATE_AU",
    "IR3TIB01CAM156N": "RATE_CA",
    "IR3TIB01CHM156N": "RATE_CH",
    "DGS3MO": "DGS3MO_US",
    "DGS2": "DGS2",
    "DGS10": "DGS10",
    "DGS20": "DGS20",
    "BAMLH0A0HYM2": "BAMLH0A0HYM2",
    "VIXCLS": "VIXCLS",
    "VXVCLS": "VXVCLS",
}
FALLBACK_WARNING = "no vintages — fredgraph fallback"

# ALFRED rejects realtime windows spanning > 2000 vintage dates; 4 calendar years of a
# daily-vintage series is ~1040, comfortably under the cap.
_REALTIME_CHUNK_YEARS = 4


def _realtime_chunks(start, end) -> list[tuple]:
    """Split [start, today/end] into consecutive <=4-year realtime windows.

    The final chunk is right-open at 9999-12-31 so brand-new vintages are included.
    Windows are [cs, ce] inclusive with ce = next_cs - 1 day, so no vintage date is
    double-counted at a boundary.
    """
    s = pd.Timestamp(start)
    today = pd.Timestamp.now().normalize()
    hard_end = max(pd.Timestamp(end), today)
    chunks: list[tuple] = []
    cs = s
    while True:
        ce = cs + pd.DateOffset(years=_REALTIME_CHUNK_YEARS) - pd.Timedelta(days=1)
        if ce >= hard_end:
            chunks.append((cs.date(), "9999-12-31"))
            return chunks
        chunks.append((cs.date(), ce.date()))
        cs = ce + pd.Timedelta(days=1)


class FredAlfredLoader(BaseLoader):
    dataset = "macro"
    vendor = "fred"
    source = "fred:alfred"
    asset_classes = ["macro"]
    default_asset_class = "macro"
    # Rule is finalized in transform: explicit(realtime_start) when vintaged, else
    # a conservative obs_offset. Default here covers the vintaged (preferred) path.
    availability_rule = AvailabilityRule("explicit", {"column": "realtime_start"})
    expectations = {
        "columns": ["value"],
        "max_null_frac": 0.05,
        "min_rows": 1,
    }

    def __init__(self, lake=None, instruments=None, series=None, api_key=None):
        super().__init__(lake, instruments)
        self.series = series or list(ALIAS.keys())
        self.api_key = api_key if api_key is not None else os.environ.get("FRED_API_KEY")

    def fetch(self, start, end) -> dict[str, pd.DataFrame]:
        out: dict[str, pd.DataFrame] = {}
        s, e = pd.Timestamp(start).date(), pd.Timestamp(end).date()
        for fid in self.series:
            if self.api_key:
                out[fid] = self._fetch_vintaged(fid, s, e)
            else:  # keyless fredgraph CSV — no vintages
                import requests

                resp = requests.get(
                    "https://fred.stlouisfed.org/graph/fredgraph.csv",
                    params={"id": fid}, timeout=30)
                resp.raise_for_status()
                import io
                df = pd.read_csv(io.StringIO(resp.text))
                df.columns = ["date", "value"]
                out[fid] = df
        return out

    def _fetch_vintaged(self, fid: str, s, e) -> pd.DataFrame:
        """ALFRED observations for `fid`, chunking the realtime window.

        ALFRED rejects requests spanning more than 2000 vintage dates (a daily series
        over 10 years has ~2600), so the realtime window is walked in <=4-year chunks
        (~1040 weekday vintages each). ALFRED clamps each returned row's
        `realtime_start` to the requested window start, so an observation published in
        an earlier chunk re-appears in later chunks stamped at the chunk boundary;
        keeping the EARLIEST realtime_start per (date, value) reconstructs exactly the
        single-request result — a genuinely revised value keeps its own later vintage.
        """
        import requests

        frames = []
        for cs, ce in _realtime_chunks(s, e):
            params = {
                "series_id": fid, "api_key": self.api_key, "file_type": "json",
                "observation_start": str(s), "observation_end": str(e),
                "realtime_start": str(cs), "realtime_end": str(ce),
            }
            resp = requests.get(
                "https://api.stlouisfed.org/fred/series/observations",
                params=params, timeout=30)
            if resp.status_code == 400 and "does not exist in ALFRED" in resp.text:
                # No vintage archive for this series (e.g. BAMLH0A0HYM2, the VIX
                # family): degrade to revised-FRED history with a synthetic
                # conservative obs+1d vintage, and say so loudly in the audit.
                return self._fetch_unvintaged(fid, s, e)
            resp.raise_for_status()
            obs = resp.json().get("observations", [])
            if obs:
                frames.append(pd.DataFrame(obs))
        if not frames:
            return pd.DataFrame()
        df = pd.concat(frames, ignore_index=True)
        return (df.sort_values("realtime_start")
                  .drop_duplicates(subset=["date", "value"], keep="first")
                  .reset_index(drop=True))

    def _fetch_unvintaged(self, fid: str, s, e) -> pd.DataFrame:
        """Plain-FRED (latest revision) pull for a series ALFRED has no archive for.

        `realtime_start` is synthesized as ``obs_date + 1 day`` — a conservative
        available-from for daily market series (HY OAS, VIX) that are not revised in
        practice. The warning keeps the loss of true vintages out of silent territory.
        """
        import requests

        resp = requests.get(
            "https://api.stlouisfed.org/fred/series/observations",
            params={"series_id": fid, "api_key": self.api_key, "file_type": "json",
                    "observation_start": str(s), "observation_end": str(e)},
            timeout=30)
        resp.raise_for_status()
        df = pd.DataFrame(resp.json().get("observations", []))
        if df.empty:
            return df
        df["realtime_start"] = (
            pd.to_datetime(df["date"]) + pd.Timedelta(days=1)).astype(str)
        self.warnings.append(
            f"{fid}: not in ALFRED — revised FRED history with synthetic obs+1d vintage")
        return df[["date", "value", "realtime_start"]]

    def transform(self, raw) -> pd.DataFrame:
        frames = []
        vintaged = True
        for fid, df in raw.items():
            if df is None or df.empty:
                continue
            sid = ALIAS.get(fid, fid)
            obs = pd.to_datetime(df["date"])
            value = pd.to_numeric(df["value"], errors="coerce")
            out = pd.DataFrame({"obs_date": obs, "series_id": sid, "value": value})
            if "realtime_start" in df.columns:
                out["realtime_start"] = pd.to_datetime(df["realtime_start"])
            else:
                vintaged = False
            out["asset_class"] = "macro"
            frames.append(out.dropna(subset=["value"]))

        if vintaged:
            self.availability_rule = AvailabilityRule("explicit", {"column": "realtime_start"})
        else:
            # No vintages available: stamp conservatively and flag the loss of PIT.
            self.availability_rule = AvailabilityRule("obs_offset", {"offset": pd.Timedelta(days=1)})
            if FALLBACK_WARNING not in self.warnings:
                self.warnings.append(FALLBACK_WARNING)

        if not frames:
            return pd.DataFrame(columns=["obs_date", "series_id", "value", "asset_class"])
        return pd.concat(frames, ignore_index=True)
