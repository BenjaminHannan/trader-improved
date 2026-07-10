"""KalshiHistoryLoader tests — NO NETWORK.

``fetch`` is monkeypatched with a canned payload shaped like the real
``/historical/markets`` + ``/historical/trades`` responses (dollar-string prices,
float-string sizes, taker sides — documented in the loader). We pin: daily-bar
aggregation arithmetic (last/VWAP/volume/taker split), the settlement-row PIT
contract (outcome knowable only at settlement_ts), the two-row-type schema, the
[0,1] range audit, and the end-to-end curated write.
"""
from __future__ import annotations

import pandas as pd
import pytest

from production.data.audit import audit
from production.data.loaders.kalshi_history import KalshiHistoryLoader

UTC = "UTC"


def _payload():
    """One series, two markets.

    M1 (result yes): 3 trades across 2 UTC days — day 1 has two trades (prices
    0.04 then 0.06, sizes 100/300, takers yes/no), day 2 one trade (0.10, 50, yes).
    M2 (result no): a single thin trade. Shapes mirror the probed API exactly.
    """
    markets = [
        {"ticker": "CPIYOY-23JUN-T2.7", "event_ticker": "CPIYOY-23JUN",
         "title": "Will CPI YoY exceed 2.7%?", "status": "finalized",
         "result": "yes", "settlement_ts": "2023-07-12T13:30:00Z",
         "close_time": "2023-07-12T12:25:00Z", "expiration_value": "3.00",
         "strike_type": "greater", "floor_strike": 2.7},
        {"ticker": "CPIYOY-23JUN-T4.0", "event_ticker": "CPIYOY-23JUN",
         "title": "Will CPI YoY exceed 4.0%?", "status": "finalized",
         "result": "no", "settlement_ts": "2023-07-12T13:30:00Z",
         "close_time": "2023-07-12T12:25:00Z", "expiration_value": "3.00",
         "strike_type": "greater", "floor_strike": 4.0},
    ]
    trades = {
        "CPIYOY-23JUN-T2.7": [
            {"created_time": "2023-07-10T14:00:00Z", "yes_price_dollars": "0.0400",
             "no_price_dollars": "0.9600", "count_fp": "100.00", "taker_side": "yes"},
            {"created_time": "2023-07-10T20:52:15Z", "yes_price_dollars": "0.0600",
             "no_price_dollars": "0.9400", "count_fp": "300.00", "taker_side": "no"},
            {"created_time": "2023-07-11T09:00:00Z", "yes_price_dollars": "0.1000",
             "no_price_dollars": "0.9000", "count_fp": "50.00", "taker_side": "yes"},
        ],
        "CPIYOY-23JUN-T4.0": [
            {"created_time": "2023-07-11T10:00:00Z", "yes_price_dollars": "0.0200",
             "no_price_dollars": "0.9800", "count_fp": "68.00", "taker_side": "no"},
        ],
    }
    return {"KXCPIYOY": {"markets": markets, "trades": trades}}


def test_daily_bar_aggregation_arithmetic():
    long = KalshiHistoryLoader().transform(_payload())
    bars = long[(long["instrument_id"] == "EV:kalshi:CPIYOY-23JUN-T2.7")
                & (long["row_type"] == "bar")].sort_values("obs_date")
    assert len(bars) == 2
    d1 = bars.iloc[0]
    assert d1["obs_date"] == pd.Timestamp("2023-07-10")
    assert d1["yes_price"] == pytest.approx(0.06)          # last trade of the day
    # VWAP = (0.04*100 + 0.06*300) / 400
    assert d1["yes_vwap"] == pytest.approx((0.04 * 100 + 0.06 * 300) / 400)
    assert d1["volume"] == pytest.approx(400.0)
    assert d1["n_trades"] == 2
    assert d1["taker_yes_frac"] == pytest.approx(100 / 400)  # contract-weighted
    # knowable the moment the last trade printed
    assert pd.Timestamp(d1["knowable_at"]) == pd.Timestamp("2023-07-10T20:52:15Z")
    d2 = bars.iloc[1]
    assert d2["yes_price"] == pytest.approx(0.10) and d2["n_trades"] == 1


def test_settlement_rows_carry_outcome_at_settlement_time():
    long = KalshiHistoryLoader().transform(_payload())
    st = long[long["row_type"] == "settlement"].set_index("instrument_id")
    assert st.loc["EV:kalshi:CPIYOY-23JUN-T2.7", "yes_price"] == 1.0   # result yes
    assert st.loc["EV:kalshi:CPIYOY-23JUN-T4.0", "yes_price"] == 0.0   # result no
    row = st.loc["EV:kalshi:CPIYOY-23JUN-T2.7"]
    assert pd.Timestamp(row["knowable_at"]) == pd.Timestamp("2023-07-12T13:30:00Z")
    assert row["obs_date"] == pd.Timestamp("2023-07-12")
    assert row["result"] == "yes" and row["expiration_value"] == "3.00"
    # outcome columns never leak onto bar rows
    bars = long[long["row_type"] == "bar"]
    assert bars["result"].isna().all() and bars["expiration_value"].isna().all()


def test_metadata_and_strikes_on_every_row():
    long = KalshiHistoryLoader().transform(_payload())
    assert (long["series_ticker"] == "KXCPIYOY").all()
    assert (long["event_key"] == "CPIYOY-23JUN").all()
    assert (long["venue"] == "kalshi").all()
    t27 = long[long["instrument_id"] == "EV:kalshi:CPIYOY-23JUN-T2.7"]
    assert (t27["floor_strike"] == 2.7).all()
    assert (t27["strike_type"] == "greater").all()


def test_range_audit_and_end_to_end_curated_write(tmp_lake, monkeypatch):
    loader = KalshiHistoryLoader(tmp_lake)
    monkeypatch.setattr(loader, "fetch", lambda start, end: _payload())
    res = loader.run("2023-01-01", "2023-12-31", incremental=False)
    assert res.audit["fatal"] is False
    cur = tmp_lake.read_curated("event_markets_hist", "events")
    # 2 bars + settlement for M1, 1 bar + settlement for M2
    assert len(cur) == 5
    # PIT: available_from equals each row's own knowable_at, tz-aware UTC
    av = pd.to_datetime(cur["available_from"], utc=True)
    kn = pd.to_datetime(cur["knowable_at"], utc=True)
    assert (av == kn).all()
    rep = audit(cur, KalshiHistoryLoader.expectations)
    assert rep.fatal is False


def test_bad_result_and_bad_trades_degrade_not_raise():
    payload = _payload()
    payload["KXCPIYOY"]["markets"].append(
        {"ticker": "CPIYOY-23JUN-XSCALAR", "event_ticker": "CPIYOY-23JUN",
         "title": "scalar oddity", "result": "3.00",   # non-binary result
         "settlement_ts": "2023-07-12T13:30:00Z",
         "close_time": "2023-07-12T12:25:00Z"})
    payload["KXCPIYOY"]["trades"]["CPIYOY-23JUN-XSCALAR"] = [
        {"created_time": None, "yes_price_dollars": "9.99", "count_fp": "1.00"},
    ]
    loader = KalshiHistoryLoader()
    long = loader.transform(payload)
    # no settlement row for the unmapped result, warning recorded, garbage trade dropped
    assert "EV:kalshi:CPIYOY-23JUN-XSCALAR" not in set(
        long[long["row_type"] == "settlement"]["instrument_id"])
    assert any("unmapped result" in w for w in loader.warnings)
    assert long["yes_price"].between(0, 1).all()
