"""CFTC Commitments of Traders (legacy, futures-only) positioning.

The COT report observes positions as of Tuesday but is not released until the
following Friday at 15:30 ET (20:30 UTC). That three-day embargo is the classic COT
look-ahead trap, so we stamp `available_from` with a `next_weekday_time` rule: the
first Friday on/after the Tuesday obs_date, at 20:30 UTC.

Futures markets are mapped to the ETF proxy the sleeve actually trades (EURO FX->FXE,
GOLD->GLD, ...); `noncomm_net = noncomm_long - noncomm_short`, plus open interest, is
attributed to the proxy's instrument_id. Socrata JSON is primary; the annual history
zip (annual.txt) is the fallback.
"""
from __future__ import annotations

import pandas as pd

from production.data.base import AvailabilityRule, BaseLoader, asset_class_from_instrument_id

# CFTC market name (prefix, upper-cased) -> proxy ETF symbol. Markets without a
# clean single-ETF proxy (e.g. broad ag baskets) are intentionally omitted and
# documented here rather than force-mapped:
#   - COPPER -> CPER exists but COT "COPPER-GRADE #1" naming is inconsistent; include.
#   - AGS baskets (DBA), base metals (DBB) have no single COT futures market -> skip.
MARKET_MAP = {
    "EURO FX": "FXE",
    "JAPANESE YEN": "FXY",
    "BRITISH POUND": "FXB",
    "AUSTRALIAN DOLLAR": "FXA",
    "CANADIAN DOLLAR": "FXC",
    "SWISS FRANC": "FXF",
    "GOLD": "GLD",
    "SILVER": "SLV",
    "CRUDE OIL, LIGHT SWEET": "USO",
    "NATURAL GAS": "UNG",
    "COPPER": "CPER",
}
SOCRATA_URL = "https://publicreporting.cftc.gov/resource/6dca-aqww.json"


class CftcCotLoader(BaseLoader):
    dataset = "cot"
    vendor = "cftc"
    source = "cftc:cot_legacy"
    asset_classes = ["fx", "commodity"]
    availability_rule = AvailabilityRule("next_weekday_time",
                                         {"weekday": 4, "hour": 20, "minute": 30})
    expectations = {
        "columns": ["noncomm_net", "open_interest"],
        "ranges": {"open_interest": (0, None)},
        "max_null_frac": 0.02,
        "min_rows": 1,
    }

    def fetch(self, start, end) -> list[dict]:
        import requests

        s, e = pd.Timestamp(start).date(), pd.Timestamp(end).date()
        where = (f"report_date_as_yyyy_mm_dd >= '{s}T00:00:00.000' "
                 f"AND report_date_as_yyyy_mm_dd <= '{e}T00:00:00.000'")
        try:
            resp = requests.get(SOCRATA_URL,
                                params={"$where": where, "$limit": 50000}, timeout=60)
            resp.raise_for_status()
            return resp.json()
        except Exception:
            return self._fetch_zip(start, end)

    def _fetch_zip(self, start, end) -> list[dict]:
        import io
        import zipfile

        import requests

        records = []
        for year in range(pd.Timestamp(start).year, pd.Timestamp(end).year + 1):
            try:
                url = f"https://www.cftc.gov/files/dea/history/deacot{year}.zip"
                resp = requests.get(url, timeout=120)
                resp.raise_for_status()
                with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                    with zf.open("annual.txt") as fh:
                        df = pd.read_csv(fh, low_memory=False)
                records.extend(df.to_dict("records"))
            except Exception:
                continue
        return records

    @staticmethod
    def _num(rec, *keys):
        for k in keys:
            if k in rec and rec[k] not in (None, ""):
                try:
                    return float(rec[k])
                except (TypeError, ValueError):
                    continue
        return None

    def transform(self, raw) -> pd.DataFrame:
        rows = []
        for rec in raw:
            name = str(rec.get("market_and_exchange_names")
                       or rec.get("Market_and_Exchange_Names") or "").upper()
            sym = next((v for k, v in MARKET_MAP.items() if name.startswith(k)), None)
            if sym is None:
                continue
            iid = self.resolve(sym, "cftc") or self.resolve(sym)
            if iid is None:
                continue
            date = (rec.get("report_date_as_yyyy_mm_dd")
                    or rec.get("As_of_Date_In_Form_YYMMDD") or rec.get("Report_Date_as_MM_DD_YYYY"))
            nc_long = self._num(rec, "noncomm_positions_long_all", "NonComm_Positions_Long_All")
            nc_short = self._num(rec, "noncomm_positions_short_all", "NonComm_Positions_Short_All")
            oi = self._num(rec, "open_interest_all", "Open_Interest_All")
            if date is None or nc_long is None or nc_short is None:
                continue
            rows.append({
                "obs_date": pd.Timestamp(str(date)).normalize(),
                "instrument_id": iid,
                "noncomm_net": nc_long - nc_short,
                "open_interest": oi,
                "asset_class": asset_class_from_instrument_id(iid),
            })
        if not rows:
            return pd.DataFrame(columns=["obs_date", "instrument_id", "noncomm_net",
                                         "open_interest", "asset_class"])
        return pd.DataFrame(rows)
