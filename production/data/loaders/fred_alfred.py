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
    "BAMLH0A0HYM2": "BAMLH0A0HYM2",
    "VIXCLS": "VIXCLS",
    "VXVCLS": "VXVCLS",
}
FALLBACK_WARNING = "no vintages — fredgraph fallback"


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
        import requests

        out: dict[str, pd.DataFrame] = {}
        s, e = pd.Timestamp(start).date(), pd.Timestamp(end).date()
        for fid in self.series:
            if self.api_key:
                params = {
                    "series_id": fid, "api_key": self.api_key, "file_type": "json",
                    "observation_start": str(s), "observation_end": str(e),
                    "realtime_start": str(s), "realtime_end": "9999-12-31",
                }
                resp = requests.get(
                    "https://api.stlouisfed.org/fred/series/observations",
                    params=params, timeout=30)
                resp.raise_for_status()
                obs = resp.json().get("observations", [])
                out[fid] = pd.DataFrame(obs)
            else:  # keyless fredgraph CSV — no vintages
                resp = requests.get(
                    "https://fred.stlouisfed.org/graph/fredgraph.csv",
                    params={"id": fid}, timeout=30)
                resp.raise_for_status()
                import io
                df = pd.read_csv(io.StringIO(resp.text))
                df.columns = ["date", "value"]
                out[fid] = df
        return out

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
