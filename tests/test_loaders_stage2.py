"""Stage-2 loader tests — NO NETWORK.

Every loader's `fetch` is monkeypatched to return small hand-built payloads shaped
like the real vendor responses, so the pipeline (transform -> stamp -> audit ->
curated write), the pinned availability lags, and the audit fatal/warning split are
all exercised without touching the network or requiring any API key.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import scripts.ingest as ingest
from production.data.base import IngestError
from production.data.loaders.coingecko import CoinGeckoLoader
from production.data.loaders.stage2.aaii_manual import AaiiManualLoader
from production.data.loaders.stage2.ads import AdsLoader
from production.data.loaders.stage2.cboe_putcall import CboePutCallLoader
from production.data.loaders.stage2.finra_short import FinraShortInterestLoader
from production.data.loaders.stage2.fred_stage2 import FredStage2Loader
from production.data.loaders.stage2.naaim import NaaimLoader

UTC = "UTC"


# --------------------------------------------------------------- canned payloads
def _fred_payload():
    dates = pd.date_range("2020-01-06", periods=4, freq="W-WED")
    def mk(v):
        return pd.DataFrame({"date": dates.astype(str), "value": [v] * len(dates),
                             "realtime_start": dates.astype(str)})
    return {"WEI": mk(1.5), "GDPNOW": mk(2.3), "CFNAI": mk(-0.1)}


def _ads_payload():
    dates = pd.date_range("2020-01-01", periods=6, freq="D")
    return pd.DataFrame({"Date": dates.astype(str),
                         "ADS_Index": np.linspace(-0.5, 0.5, 6)})


def _naaim_payload():
    # Wednesday-dated survey rows.
    dates = pd.date_range("2020-01-08", periods=4, freq="W-WED")
    return pd.DataFrame({"Date": dates.astype(str),
                         "NAAIM Number": [50.0, 60.0, 70.0, 55.0]})


def _cboe_payload():
    dates = pd.date_range("2020-01-06", periods=5, freq="B")
    return pd.DataFrame({"DATE": dates.astype(str),
                         "TOTAL": [0.9, 1.1, 1.0, 0.8, 1.2],
                         "EQUITY": [0.6, 0.7, 0.65, 0.55, 0.75],
                         "INDEX": [1.5, 1.6, 1.4, 1.3, 1.7]})


def _finra_payload():
    return [
        {"settlementDate": "2020-01-15", "symbolCode": "AAA",
         "currentShortPositionQuantity": 1_000_000},
        {"settlementDate": "2020-01-15", "symbolCode": "BBB",
         "currentShortPositionQuantity": 2_500_000},
    ]


def _aaii_payload():
    dates = pd.date_range("2020-01-09", periods=3, freq="W-THU")  # Thursdays
    return pd.DataFrame({"Date": dates.astype(str),
                         "Bullish": [0.35, 0.40, 0.30],
                         "Neutral": [0.40, 0.35, 0.45],
                         "Bearish": [0.25, 0.25, 0.25]})


# ================================================ (1) FRED Stage-2 alias / vintage
def test_fred_stage2_default_series_and_vintage_rule():
    loader = FredStage2Loader()
    assert set(loader.series) == {"WEI", "GDPNOW", "CFNAI"}
    long = loader.transform(_fred_payload())
    # identity aliases -> series_ids unchanged
    assert set(long["series_id"]) == {"WEI", "GDPNOW", "CFNAI"}
    # ALFRED vintages present -> explicit realtime_start availability
    assert loader.availability_rule.kind == "explicit"
    assert loader.availability_rule.params["column"] == "realtime_start"


# ============================================================= (2) ADS ingest_time
def test_ads_parses_colon_separated_vintage_dates(tmp_lake):
    """Regression (live-ingest 2026-07-07): the Philadelphia Fed vintage workbook
    writes dates as '1960:03:01'; naive to_datetime coerces every row to NaT and the
    loader emits an empty frame that fails the schema audit."""
    payload = pd.DataFrame({"Date": ["2020:01:02", "2020:01:03", "2020:01:06"],
                            "ADS_Index": [-0.1, 0.0, 0.1], "RECBARS": [0, 0, 0]})
    out = AdsLoader(tmp_lake).transform(payload)
    assert len(out) == 3
    assert out["obs_date"].tolist() == [pd.Timestamp("2020-01-02"),
                                        pd.Timestamp("2020-01-03"),
                                        pd.Timestamp("2020-01-06")]


def test_ads_transform_and_ingest_time_rule(tmp_lake, monkeypatch):
    loader = AdsLoader(tmp_lake)
    monkeypatch.setattr(loader, "fetch", lambda s, e: _ads_payload())
    res = loader.run("2020-01-01", "2020-01-31", incremental=False)
    assert res.rows == 6

    cur = tmp_lake.read_curated("macro", "macro")
    ads = cur[cur["series_id"] == "ADS_INDEX"]
    assert len(ads) == 6
    # ingest_time -> available_from equals ingested_at (row became knowable at pull)
    row = ads.iloc[0]
    assert row["available_from"] == pd.Timestamp(row["ingested_at"])


# ================================================= (3) NAAIM Wed->Thu availability
def test_naaim_availability_wed_to_thursday_2100(tmp_lake, monkeypatch):
    loader = NaaimLoader(tmp_lake)
    monkeypatch.setattr(loader, "fetch", lambda s, e: _naaim_payload())
    res = loader.run("2020-01-01", "2020-01-31", incremental=False)
    assert res.rows == 4

    cur = tmp_lake.read_curated("macro", "macro")
    naaim = cur[cur["series_id"] == "NAAIM_EXPOSURE"].sort_values("obs_date")
    first = naaim.iloc[0]
    # obs Wednesday 2020-01-08 -> Thursday 2020-01-09 21:00 UTC
    assert first["obs_date"] == pd.Timestamp("2020-01-08")
    assert first["available_from"] == pd.Timestamp("2020-01-09 21:00", tz=UTC)


# ================================================== (4) CBOE obs+1day noon, 3 series
def test_cboe_three_series_and_next_day_noon(tmp_lake, monkeypatch):
    loader = CboePutCallLoader(tmp_lake)
    monkeypatch.setattr(loader, "fetch", lambda s, e: _cboe_payload())
    res = loader.run("2020-01-01", "2020-01-31", incremental=False)
    assert res.rows == 15  # 5 days x 3 series

    cur = tmp_lake.read_curated("macro", "macro")
    assert {"CBOE_TOTAL_PC", "CBOE_EQUITY_PC", "CBOE_INDEX_PC"} <= set(cur["series_id"])
    total = cur[cur["series_id"] == "CBOE_TOTAL_PC"].sort_values("obs_date").iloc[0]
    # obs Monday 2020-01-06 -> 2020-01-07 12:00 UTC
    assert total["obs_date"] == pd.Timestamp("2020-01-06")
    assert total["available_from"] == pd.Timestamp("2020-01-07 12:00", tz=UTC)


# ================================================= (5) FINRA per-symbol + 13d lag
def test_finra_per_symbol_series_and_conservative_lag(tmp_lake, monkeypatch):
    loader = FinraShortInterestLoader(tmp_lake)
    monkeypatch.setattr(loader, "fetch", lambda s, e: _finra_payload())
    res = loader.run("2020-01-01", "2020-02-15", incremental=False)
    assert res.rows == 2

    cur = tmp_lake.read_curated("macro", "macro")
    aaa = cur[cur["series_id"] == "FINRA_SI:AAA"]
    assert len(aaa) == 1
    row = aaa.iloc[0]
    assert row["value"] == 1_000_000
    # settlement 2020-01-15 + 13 days 22:00 -> 2020-01-28 22:00 UTC
    assert row["available_from"] == pd.Timestamp("2020-01-28 22:00", tz=UTC)


# ============================================ (6) AAII manual: file present + absent
def test_aaii_manual_reads_local_file(tmp_path, tmp_lake, monkeypatch):
    path = tmp_path / "sentiment.xls"
    loader = AaiiManualLoader(tmp_lake, path=path)
    # bypass the on-disk xls read; exercise transform + availability directly
    monkeypatch.setattr(loader, "fetch", lambda s, e: _aaii_payload())
    res = loader.run("2020-01-01", "2020-02-15", incremental=False)
    assert res.rows == 9  # 3 weeks x {bull, neutral, bear}

    cur = tmp_lake.read_curated("macro", "macro")
    assert {"AAII_BULL", "AAII_NEUTRAL", "AAII_BEAR"} <= set(cur["series_id"])
    bull = cur[cur["series_id"] == "AAII_BULL"].sort_values("obs_date").iloc[0]
    # obs Thursday 2020-01-09 -> same Thursday 12:00 UTC
    assert bull["obs_date"] == pd.Timestamp("2020-01-09")
    assert bull["available_from"] == pd.Timestamp("2020-01-09 12:00", tz=UTC)


def test_aaii_manual_missing_file_raises_ingest_error(tmp_path, tmp_lake):
    loader = AaiiManualLoader(tmp_lake, path=tmp_path / "does_not_exist.xls")
    with pytest.raises(IngestError) as exc:
        loader.fetch("2020-01-01", "2020-02-15")
    assert "data/manual" in str(exc.value)


# =================================================== (7) audit integration (fatal)
def test_cboe_negative_ratio_is_fatal(tmp_lake, monkeypatch):
    bad = _cboe_payload()
    bad.loc[0, "TOTAL"] = -1.0  # ratios must be >= 0
    loader = CboePutCallLoader(tmp_lake)
    monkeypatch.setattr(loader, "fetch", lambda s, e: bad)
    with pytest.raises(IngestError):
        loader.run("2020-01-01", "2020-01-31", incremental=False)
    # audit trail persisted even though the curated write was blocked
    assert list((tmp_lake.root / "audit").glob("*.json"))


# ===================================================== (8) CoinGecko API-key header
def test_coingecko_demo_key_sets_header(monkeypatch):
    captured = {}

    class _Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return []

    def fake_get(url, params=None, headers=None, timeout=None):
        captured["headers"] = headers
        return _Resp()

    monkeypatch.setenv("COINGECKO_API_KEY", "demo-secret")
    monkeypatch.setattr("requests.get", fake_get)
    loader = CoinGeckoLoader()
    loader.fetch("2020-01-01", "2020-01-02")
    assert captured["headers"] == {"x-cg-demo-api-key": "demo-secret"}


def test_coingecko_no_key_sends_no_header(monkeypatch):
    captured = {}

    class _Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return []

    def fake_get(url, params=None, headers=None, timeout=None):
        captured["headers"] = headers
        return _Resp()

    monkeypatch.delenv("COINGECKO_API_KEY", raising=False)
    monkeypatch.setattr("requests.get", fake_get)
    loader = CoinGeckoLoader()
    loader.fetch("2020-01-01", "2020-01-02")
    assert captured["headers"] is None


# ======================================================= (9) --stage 2 CLI smoke
def test_ingest_stage2_cli_smoke(tmp_lake, monkeypatch):
    """`ingest.py --stage 2` runs the whole Stage-2 batch (fetch mocked, no network)."""
    fetches = {
        "FredStage2Loader": _fred_payload(),
        "AdsLoader": _ads_payload(),
        "NaaimLoader": _naaim_payload(),
        "CboePutCallLoader": _cboe_payload(),
        "FinraShortInterestLoader": _finra_payload(),
    }

    def build(lake, instruments):
        loaders = [
            FredStage2Loader(tmp_lake),
            AdsLoader(tmp_lake),
            NaaimLoader(tmp_lake),
            CboePutCallLoader(tmp_lake),
            FinraShortInterestLoader(tmp_lake),
            AaiiManualLoader(tmp_lake),  # no file -> raises, main keeps going (rc=1)
        ]
        for ld in loaders:
            name = type(ld).__name__
            if name in fetches:
                payload = fetches[name]
                ld.fetch = lambda s, e, _p=payload: _p
        return loaders

    monkeypatch.setattr(ingest, "_make_stage2_loaders", build)
    rc = ingest.main(["--stage", "2", "--start", "2020-01-01", "--end", "2020-02-15",
                      "--lake-root", str(tmp_lake.root)])
    # AAII manual has no file, so the batch reports a non-zero rc but the others land.
    assert rc == 1
    cur = tmp_lake.read_curated("macro", "macro")
    assert {"WEI", "ADS_INDEX", "NAAIM_EXPOSURE", "CBOE_TOTAL_PC"} <= set(cur["series_id"])


# ------------------------------------------- cleveland fed nowcast vintages
def test_cleveland_nowcast_parses_quarter_nodes_with_year_inference():
    """Canned payload of the REAL FusionCharts shape (live probe 2026-07-07): one
    node per target quarter, MM/DD labels that can precede the quarter (a December
    label on a Q1 chart belongs to the prior year), 'Actual ...' series skipped,
    quarter-transition overlaps resolved to the newer target quarter."""
    from production.data.loaders.stage2.cleveland_nowcast import ClevelandNowcastLoader

    raw = [
        {"chart": {"subcaption": "2014:Q1", "_comment": "x"},
         "categories": [{"category": [{"label": "12/30"}, {"label": "01/02"}]}],
         "dataset": [
             {"seriesname": "CPI Inflation",
              "data": [{"value": "1.51"}, {"value": "1.62"}]},
             {"seriesname": "Actual CPI Inflation",
              "data": [{"value": ""}, {"value": "1.60"}]},
         ]},
        {"chart": {"subcaption": "2013:Q4", "_comment": "x"},
         "categories": [{"category": [{"label": "12/30"}]}],
         "dataset": [
             {"seriesname": "CPI Inflation", "data": [{"value": "1.20"}]},
         ]},
    ]
    loader = ClevelandNowcastLoader()
    df = loader.transform(raw)

    assert set(df["series_id"]) == {"CLEV_NOWCAST_CPI"}          # actuals skipped
    by_date = df.set_index("obs_date")["value"]
    # 12/30 on the 2014:Q1 chart -> 2013-12-30 (prior year); overlap with the
    # 2013:Q4 node's same date resolves to the NEWER target quarter (Q1's 1.51,
    # not Q4's 1.20) because nodes arrive oldest-first and keep="last" wins.
    assert by_date[pd.Timestamp("2013-12-30")] == 1.51
    assert by_date[pd.Timestamp("2014-01-02")] == 1.62
    assert len(df) == 2


def test_cleveland_nowcast_monthly_nodes_tag_target_month():
    """Monthly file (probe 2026-07-10): subcaption 'YYYY-M', one node per TARGET
    month, series id carries the target-month tag so simultaneous vintages for two
    months never collide; a January label on a December node is the NEXT year."""
    from production.data.loaders.stage2.cleveland_nowcast import ClevelandNowcastLoader

    raw = {
        "quarter": [],
        "month": [
            {"chart": {"subcaption": "2025-12"},
             "categories": [{"category": [{"label": "12/30"}, {"label": "01/12"}]}],
             "dataset": [
                 {"seriesname": "CPI Inflation",
                  "data": [{"value": "0.31"}, {"value": "0.28"}]},
                 {"seriesname": "Actual CPI Inflation",
                  "data": [{"value": ""}, {"value": "0.30"}]},
             ]},
            {"chart": {"subcaption": "2026-1"},
             "categories": [{"category": [{"label": "12/30"}]}],
             "dataset": [
                 {"seriesname": "Core PCE Inflation", "data": [{"value": "0.22"}]},
             ]},
        ],
    }
    df = ClevelandNowcastLoader().transform(raw)
    assert set(df["series_id"]) == {"CLEV_NOWCAST_CPI_MOM:2025-12",
                                    "CLEV_NOWCAST_COREPCE_MOM:2026-01"}
    dec = df[df["series_id"] == "CLEV_NOWCAST_CPI_MOM:2025-12"].set_index("obs_date")
    # 01/12 on the 2025-12 node -> 2026-01-12 (the pre-release tail of December)
    assert dec.loc[pd.Timestamp("2026-01-12"), "value"] == 0.28
    assert dec.loc[pd.Timestamp("2025-12-30"), "value"] == 0.31
    # the 2026-1 node's 12/30 label stays in 2025 (ramp before the target month)
    jan = df[df["series_id"] == "CLEV_NOWCAST_COREPCE_MOM:2026-01"]
    assert list(jan["obs_date"]) == [pd.Timestamp("2025-12-30")]
