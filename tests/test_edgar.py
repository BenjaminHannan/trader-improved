"""EDGAR fundamentals loader + value/PEAD signals: PIT known-answer tests.

Covers three things the CLAUDE.md rules make non-negotiable for fundamentals:
  * the loader stamps ``available_from`` from the FILING date (21:00 UTC), never the
    fiscal period end — fundamentals are knowable at filing;
  * restatements coexist as later vintages (the original filing survives);
  * the value/drift signals gate on the filing date, so a quarter filed after decision
    date D contributes nothing to D.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from production.data.loaders.edgar import EdgarFundamentalsLoader
from production.signals.value import EarningsYield, PostEarningsDrift

UTC = "UTC"


# --------------------------------------------------------------- test helpers
def _instruments_master(rows: dict[str, str]) -> pd.DataFrame:
    """Minimal instrument master: {ticker: instrument_id} understood by both the
    reference resolver (vendor_symbols json + validity window) and the local fallback."""
    recs = []
    for ticker, iid in rows.items():
        recs.append({
            "instrument_id": iid,
            "symbol": ticker,
            "vendor_symbols": f'{{"sec": "{ticker}"}}',
            "valid_from": pd.NaT,
            "valid_to": pd.NaT,
        })
    return pd.DataFrame(recs)


def _concept(entries: list[dict], unit: str) -> dict:
    return {"units": {unit: entries}}


def _e(end: str, val, filed: str, form: str = "10-Q") -> dict:
    return {"end": end, "val": val, "filed": filed, "form": form,
            "fy": 2020, "fp": "Q1", "accn": f"acc-{end}-{filed}"}


def _canned_companyfacts() -> dict[str, dict]:
    """Two tickers, six quarters of EPS, plus a comparative re-report and a restatement.

    AAPL 2020-03-31 EPS is filed three times: original 1.0 (2020-05-01), an unchanged
    comparative 1.0 (2020-07-31, must dedup away), and a RESTATEMENT to 0.9
    (2021-02-15, must survive as a later vintage). An 8-K EPS entry must be dropped
    (not a 10-Q/10-K). MSFT uses EarningsPerShareBasic (diluted absent) and the revenue
    fallback concept, exercising both fallback chains.
    """
    aapl_eps = [
        _e("2020-03-31", 1.0, "2020-05-01"),
        _e("2020-03-31", 1.0, "2020-07-31"),          # comparative re-report -> dedup
        _e("2020-03-31", 0.9, "2021-02-15"),          # restatement -> later vintage
        _e("2020-06-30", 1.1, "2020-07-31"),
        _e("2020-09-30", 1.2, "2020-10-30"),
        _e("2020-12-31", 1.3, "2021-01-29", form="10-K"),
        _e("2021-03-31", 1.4, "2021-05-01"),
        _e("2021-06-30", 1.5, "2021-07-30"),
        _e("2020-03-31", 9.9, "2020-04-15", form="8-K"),  # wrong form -> dropped
    ]
    aapl = {
        "cik": 320193, "entityName": "Apple Inc.",
        "facts": {
            "us-gaap": {
                "EarningsPerShareDiluted": _concept(aapl_eps, "USD/shares"),
                "Revenues": _concept(
                    [_e("2020-03-31", 5.8e10, "2020-05-01")], "USD"),
            },
            "dei": {
                "CommonStockSharesOutstanding": _concept(
                    [_e("2020-03-31", 1.7e10, "2020-05-01")], "shares"),
            },
        },
    }
    msft_eps = [_e(f"2020-{mm:02d}-{dd}", 2.0 + i * 0.1, filed)
                for i, (mm, dd, filed) in enumerate([
                    (3, "31", "2020-04-29"), (6, "30", "2020-07-22"),
                    (9, "30", "2020-10-27"), (12, "31", "2021-01-26")])]
    msft = {
        "cik": 789019, "entityName": "Microsoft Corp.",
        "facts": {
            "us-gaap": {
                "EarningsPerShareBasic": _concept(msft_eps, "USD/shares"),
                "RevenueFromContractWithCustomerExcludingAssessedTax": _concept(
                    [_e("2020-03-31", 3.5e10, "2020-04-29")], "USD"),
            },
        },
    }
    return {"AAPL": aapl, "MSFT": msft}


AAPL_ID = "EQ:AAPL:1980-12-12"
MSFT_ID = "EQ:MSFT:1986-03-13"


@pytest.fixture()
def edgar_loader(tmp_lake):
    master = _instruments_master({"AAPL": AAPL_ID, "MSFT": MSFT_ID})
    return EdgarFundamentalsLoader(tmp_lake, master, symbols=["AAPL", "MSFT"])


# ---------------------------------------------------------------- transform
def test_transform_resolves_sec_dash_symbol_via_yfinance_vendor_key():
    """Regression (live-ingest 2026-07-07): the REAL minted master keys equities on the
    dotted Wikipedia symbol (BRK.B) with the dash form only under vendor_symbols
    "yfinance" (and no "sec" key at all). Symbols arrive from the SEC ticker directory
    in dash form, so transform's resolve chain must try the yfinance vendor key."""
    import json as _json

    master = pd.DataFrame([{
        "instrument_id": "EQ:BRK.B:1996-05-09",
        "symbol": "BRK.B",                                       # dotted, master form
        "vendor_symbols": _json.dumps(
            {"yfinance": "BRK-B", "stooq": "BRK-B.US", "alpaca": "BRK.B"}),
        "valid_from": pd.NaT, "valid_to": pd.NaT,
    }])
    loader = EdgarFundamentalsLoader(None, master, symbols=["BRK-B"])
    facts = {"BRK-B": {
        "cik": 1067983, "entityName": "Berkshire Hathaway Inc.",
        "facts": {"us-gaap": {"EarningsPerShareDiluted": _concept(
            [_e("2020-03-31", 2.0, "2020-05-01")], "USD/shares")}},
    }}
    df = loader.transform(facts)
    assert not df.empty
    assert set(df["instrument_id"]) == {"EQ:BRK.B:1996-05-09"}


def test_transform_drops_annual_duration_keeps_quarterly():
    """Regression (live-ingest 2026-07-07): Q4 and full-FY values share the same period
    `end` and often the same filing date, colliding on the audit's duplicate key. The
    annual-duration row (>200d span) must be dropped; quarterly and start-less
    (instant / legacy-fixture) rows survive."""
    iid = "EQ:AAA:2000-01-03"
    master = _instruments_master({"AAA": iid})
    loader = EdgarFundamentalsLoader(None, master, symbols=["AAA"])
    facts = {"AAA": {"facts": {"us-gaap": {"EarningsPerShareDiluted": {"units": {
        "USD/shares": [
            {"start": "2020-10-01", "end": "2020-12-31", "val": 1.0,
             "filed": "2021-02-01", "form": "10-K"},          # Q4 duration — keep
            {"start": "2020-01-01", "end": "2020-12-31", "val": 4.0,
             "filed": "2021-02-01", "form": "10-K"},          # FY duration — drop
        ]}}}}}}
    df = loader.transform(facts)
    assert len(df) == 1
    assert df["value"].iloc[0] == 1.0


def test_transform_same_day_refiling_keeps_one_row():
    """A same-day amendment with a different value shares available_from with the
    original; exactly one row (the amendment) must survive the audit key."""
    iid = "EQ:AAA:2000-01-03"
    master = _instruments_master({"AAA": iid})
    loader = EdgarFundamentalsLoader(None, master, symbols=["AAA"])
    facts = {"AAA": {"facts": {"us-gaap": {"EarningsPerShareDiluted": {"units": {
        "USD/shares": [
            {"start": "2020-01-01", "end": "2020-03-31", "val": 1.0,
             "filed": "2020-05-01", "form": "10-Q"},
            {"start": "2020-01-01", "end": "2020-03-31", "val": 1.1,
             "filed": "2020-05-01", "form": "10-Q/A"},
        ]}}}}}}
    df = loader.transform(facts)
    assert len(df) == 1
    assert df["value"].iloc[0] == 1.1


def test_transform_restatement_is_a_later_vintage(edgar_loader):
    df = edgar_loader.transform(_canned_companyfacts())
    aapl_q1 = df[(df["instrument_id"] == AAPL_ID) & (df["field"] == "eps")
                 & (df["obs_date"] == pd.Timestamp("2020-03-31"))]
    # Exactly two vintages survive: original 1.0 and restated 0.9 (comparative deduped).
    got = sorted(zip(aapl_q1["value"], aapl_q1["filed_ts"].dt.date.astype(str)))
    assert got == [(0.9, "2021-02-15"), (1.0, "2020-05-01")]
    # The restatement carries the LATER filing date; the original still exists.
    assert set(aapl_q1["value"]) == {1.0, 0.9}


def test_transform_drops_non_periodic_forms(edgar_loader):
    df = edgar_loader.transform(_canned_companyfacts())
    # The 8-K EPS value (9.9) must never appear.
    assert 9.9 not in set(df["value"])


def test_transform_concept_fallbacks(edgar_loader):
    df = edgar_loader.transform(_canned_companyfacts())
    fields = df.groupby("instrument_id")["field"].agg(set)
    assert fields[AAPL_ID] == {"eps", "revenue", "shares"}
    # MSFT: EPS via Basic fallback, revenue via the RevenueFromContract fallback.
    msft = df[df["instrument_id"] == MSFT_ID]
    assert {"eps", "revenue"} <= set(msft["field"])
    assert 3.5e10 in set(msft.loc[msft["field"] == "revenue", "value"])


def test_filing_date_availability_is_21utc_on_filed(edgar_loader):
    df = edgar_loader.transform(_canned_companyfacts())
    stamped = edgar_loader.stamp_availability(df)
    orig = stamped[(stamped["instrument_id"] == AAPL_ID) & (stamped["field"] == "eps")
                   & (stamped["obs_date"] == pd.Timestamp("2020-03-31"))
                   & (stamped["value"] == 1.0)]
    avail = pd.Timestamp(orig["available_from"].iloc[0])
    assert avail == pd.Timestamp("2020-05-01 21:00", tz=UTC)
    # available_from is the FILING day, categorically after the period end (obs_date).
    assert avail.tz_convert(None).normalize() > pd.Timestamp("2020-03-31")


def test_run_end_to_end_writes_curated(edgar_loader, tmp_lake, monkeypatch):
    monkeypatch.setattr(edgar_loader, "fetch", lambda s, e: _canned_companyfacts())
    res = edgar_loader.run("2020-01-01", "2021-12-31", incremental=False)
    assert res.rows > 0
    cur = tmp_lake.read_curated("fundamentals", "equity")
    q1 = cur[(cur["instrument_id"] == AAPL_ID) & (cur["field"] == "eps")
             & (cur["obs_date"] == pd.Timestamp("2020-03-31"))]
    assert set(np.round(q1["value"], 2)) == {1.0, 0.9}
    restated = q1[q1["value"] == 0.9]
    assert pd.Timestamp(restated["available_from"].iloc[0]) == \
        pd.Timestamp("2021-02-15 21:00", tz=UTC)


# ------------------------------------------------------------- signal fixtures
def _fundamentals(iid: str, quarters: list[tuple[str, float, str]]) -> pd.DataFrame:
    """quarters: list of (period_end, eps, filed_date). available_from = filed @ 21:00 UTC."""
    rows = []
    for end, eps, filed in quarters:
        avail = pd.Timestamp(f"{filed} 21:00", tz=UTC)
        rows.append((pd.Timestamp(end), iid, "eps", eps, avail,
                     "test:fundamentals", avail + pd.Timedelta(minutes=5)))
    return pd.DataFrame(rows, columns=["obs_date", "instrument_id", "field", "value",
                                       "available_from", "source", "ingested_at"])


def _prices(iid: str, dates, closes) -> pd.DataFrame:
    dates = pd.DatetimeIndex(dates)
    avail = dates.tz_localize(UTC) + pd.Timedelta(hours=21, minutes=30)
    return pd.DataFrame({
        "obs_date": dates, "instrument_id": iid,
        "close": np.asarray(closes, dtype=float),
        "volume": 1e6, "dollar_volume": np.asarray(closes, dtype=float) * 1e6,
        "available_from": avail, "source": "test:prices", "ingested_at": avail,
    })


# --------------------------------------------------------- earnings_yield
def test_earnings_yield_trailing_4q_and_pit_discriminator():
    iid = "EQ:TST:2015-01-02"
    quarters = [
        ("2020-03-31", 1.0, "2020-05-01"),
        ("2020-06-30", 2.0, "2020-07-31"),
        ("2020-09-30", 3.0, "2020-10-30"),
        ("2020-12-31", 4.0, "2021-01-29"),
        ("2021-03-31", 5.0, "2021-05-03"),   # filed AFTER D1 -> excluded from D1
    ]
    fund = _fundamentals(iid, quarters)
    dates = ["2019-06-03", "2020-11-02", "2021-03-01", "2021-05-04"]
    closes = [100.0, 99.0, 50.0, 70.0]
    prices = _prices(iid, dates, closes)

    out = EarningsYield().compute({"prices": prices, "fundamentals": fund})
    by_date = out.set_index("obs_date")["value"]

    # D1 = 2021-03-01: only Q1..Q4 filed (Q5 filed 2021-05-03 EXCLUDED) -> 10/50.
    assert by_date.loc[pd.Timestamp("2021-03-01")] == pytest.approx(10.0 / 50.0)
    # D2 = 2021-05-04: Q5 now filed -> trailing four are Q2..Q5 = 14/70.
    assert by_date.loc[pd.Timestamp("2021-05-04")] == pytest.approx(14.0 / 70.0)
    # 2020-11-02: only three quarters filed -> insufficient -> absent.
    assert pd.Timestamp("2020-11-02") not in by_date.index
    # The anchor date long before any filing is absent too.
    assert pd.Timestamp("2019-06-03") not in by_date.index


# ------------------------------------------------------------------- pead
def test_pead_surprise_window_and_pre_filing_absence():
    iid = "EQ:TST:2015-01-02"
    quarters = [
        ("2020-03-31", 1.0, "2020-05-01"),   # year-ago quarter for Q5
        ("2020-06-30", 2.0, "2020-07-31"),
        ("2020-09-30", 3.0, "2020-10-30"),
        ("2020-12-31", 4.0, "2021-01-29"),
        ("2021-03-31", 5.0, "2021-05-03"),   # announcement: surprise vs 2020-03-31
    ]
    fund = _fundamentals(iid, quarters)
    bdays = pd.bdate_range("2021-01-04", "2021-12-31")
    prices = _prices(iid, bdays, np.linspace(100.0, 120.0, len(bdays)))

    out = PostEarningsDrift().compute({"prices": prices, "fundamentals": fund})
    by_date = out.set_index("obs_date")["value"]

    surprise = (5.0 - 1.0) / max(abs(1.0), 0.25)  # = 4.0
    emergence = pd.Timestamp("2021-05-03")        # a business day
    start = int(bdays.get_indexer([emergence])[0])

    # Not emitted before the filing (period end was 2021-03-31; still nothing on 04-15).
    assert pd.Timestamp("2021-04-15") not in by_date.index
    # Prints on the filing date and holds for exactly 63 trading days.
    assert by_date.loc[emergence] == pytest.approx(surprise)
    assert by_date.loc[bdays[start + 62]] == pytest.approx(surprise)
    # Day 64 of the window is past the drift horizon -> absent.
    assert bdays[start + 63] not in by_date.index


def test_pead_not_emitted_without_five_quarters():
    iid = "EQ:TST:2015-01-02"
    # Only four quarters -> no period is four-back -> no surprise event.
    quarters = [
        ("2020-03-31", 1.0, "2020-05-01"),
        ("2020-06-30", 2.0, "2020-07-31"),
        ("2020-09-30", 3.0, "2020-10-30"),
        ("2020-12-31", 4.0, "2021-01-29"),
    ]
    fund = _fundamentals(iid, quarters)
    bdays = pd.bdate_range("2021-01-04", "2021-12-31")
    prices = _prices(iid, bdays, np.linspace(100.0, 120.0, len(bdays)))
    out = PostEarningsDrift().compute({"prices": prices, "fundamentals": fund})
    assert out.empty


# ------------------------------------- signals run on the shared fixture panel
def test_signals_nonempty_on_fixture(signal_data):
    ey = EarningsYield().compute(signal_data)
    pead = PostEarningsDrift().compute(signal_data)
    assert not ey.empty and not pead.empty
    assert list(ey.columns) == ["obs_date", "instrument_id", "value"]
    assert list(pead.columns) == ["obs_date", "instrument_id", "value"]
