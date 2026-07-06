"""Fama-French daily factor returns via pandas_datareader (Ken French data library).

VALIDATION-ONLY dataset. These well-known factor returns are never traded and never
fed to a live signal — they exist so `risk/validate_against_french` can sanity-check
that our internally estimated equity factor returns correlate with the canonical
academic series. Vendor values are in percent; we store decimal returns (divide by
100). series_id: FF_MKT_RF, FF_SMB, FF_HML, FF_RF, FF_MOM.

Availability: the library refreshes with a lag of a few business days; we stamp a
conservative `obs_date + 7 calendar days` (>= 5 business days) so no validation date
could ever leak forward.
"""
from __future__ import annotations

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

    FACTOR_SET = "F-F_Research_Data_Factors_daily"
    MOM_SET = "F-F_Momentum_Factor_daily"

    def fetch(self, start, end) -> dict[str, pd.DataFrame]:
        import pandas_datareader.data as web

        out = {}
        for key, name in (("factors", self.FACTOR_SET), ("momentum", self.MOM_SET)):
            bundle = web.DataReader(name, "famafrench",
                                    pd.Timestamp(start), pd.Timestamp(end))
            out[key] = bundle[0]  # [0] is the daily table
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
