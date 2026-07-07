"""Reference-package tests: instrument master, PIT universe membership, hygiene.

No network. Wikipedia parsing is exercised with injected fixture HTML that mimics the
real page's two tables (current constituents + "Selected changes"). Crypto membership
is exercised with synthetic prices, including a same-day volume spike to prove the
ranking uses shifted (t-1) volume.
"""
from __future__ import annotations

import json

import pandas as pd
import pytest

from tests.conftest import make_gbm_prices
from production.reference import hygiene
from production.reference.instruments import (
    INSTRUMENT_COLUMNS,
    add_equity_instruments,
    build_instrument_master,
    resolve_vendor_symbol,
    vendor_symbol,
)
from production.reference.universe import (
    crypto_membership,
    membership_as_of,
    sp500_membership_from_wikipedia,
    static_membership,
    to_instrument_membership,
)

# --------------------------------------------------------------------- fixtures
# Five current constituents. AAA/BBB were never touched by a change (members "since
# before history"); CCC/DDD/EEE each entered via a change row below.
CURRENT_HTML = """
<table>
<tr><th>Symbol</th><th>Security</th><th>GICS Sector</th></tr>
<tr><td>AAA</td><td>Alpha Inc</td><td>Technology</td></tr>
<tr><td>BBB</td><td>Beta Inc</td><td>Technology</td></tr>
<tr><td>CCC</td><td>Gamma Inc</td><td>Health Care</td></tr>
<tr><td>DDD</td><td>Delta Inc</td><td>Energy</td></tr>
<tr><td>EEE</td><td>Epsilon Inc</td><td>Financials</td></tr>
</table>
"""

# Four change rows (Wikipedia's two-row Added/Removed header). Covers:
#  - add + remove pairs,
#  - a pure ADD (blank removed cell, 2021),
#  - a pure REMOVAL (blank added cell, 2020).
CHANGES_HTML = """
<table>
<tr><th>Date</th><th colspan="2">Added</th><th colspan="2">Removed</th><th>Reason</th></tr>
<tr><th>Date</th><th>Ticker</th><th>Security</th><th>Ticker</th><th>Security</th><th>Reason</th></tr>
<tr><td>June 1, 2023</td><td>CCC</td><td>Gamma Inc</td><td>XXX</td><td>Ex Corp</td><td>r1</td></tr>
<tr><td>March 15, 2022</td><td>DDD</td><td>Delta Inc</td><td>YYY</td><td>Why Corp</td><td>r2</td></tr>
<tr><td>September 10, 2021</td><td>EEE</td><td>Epsilon Inc</td><td></td><td></td><td>r3</td></tr>
<tr><td>January 20, 2020</td><td></td><td></td><td>ZZZ</td><td>Zed Corp</td><td>r4</td></tr>
</table>
"""


# ---------------------------------------------------------------- S&P walk-back
def test_sp500_walkback_reconstructs_intervals():
    m = sp500_membership_from_wikipedia(CURRENT_HTML, CHANGES_HTML)
    assert set(m.columns) == {"symbol", "universe", "effective_from", "effective_to"}
    assert (m["universe"] == "sp500").all()

    def interval(sym):
        row = m[m["symbol"] == sym].iloc[0]
        return row["effective_from"], row["effective_to"]

    # Current members that entered via a change get a known effective_from, open end.
    assert interval("CCC") == (pd.Timestamp("2023-06-01"), pd.NaT)
    assert interval("DDD") == (pd.Timestamp("2022-03-15"), pd.NaT)
    assert interval("EEE") == (pd.Timestamp("2021-09-10"), pd.NaT)  # pure add
    # Untouched current members: member since before history, still current.
    for sym in ("AAA", "BBB"):
        ef, et = interval(sym)
        assert pd.isna(ef) and pd.isna(et)
    # Removed names: unknown start, closed at their removal date.
    assert interval("XXX") == (pd.NaT, pd.Timestamp("2023-06-01"))
    assert interval("YYY") == (pd.NaT, pd.Timestamp("2022-03-15"))
    assert interval("ZZZ") == (pd.NaT, pd.Timestamp("2020-01-20"))  # pure removal
    # No spurious rows from blank cells.
    assert m["symbol"].notna().all()
    assert len(m) == 8


def test_sp500_membership_as_of_three_dates():
    m = sp500_membership_from_wikipedia(CURRENT_HTML, CHANGES_HTML)
    # Before every change: the pre-history members plus everything later removed.
    assert membership_as_of(m, "sp500", "2019-01-01") == ["AAA", "BBB", "XXX", "YYY", "ZZZ"]
    # Mid: after ZZZ removed (2020) and before EEE added (2021-09).
    assert membership_as_of(m, "sp500", "2021-01-01") == ["AAA", "BBB", "XXX", "YYY"]
    # After every change: exactly today's constituent list.
    assert membership_as_of(m, "sp500", "2024-01-01") == ["AAA", "BBB", "CCC", "DDD", "EEE"]


def test_to_instrument_membership_maps_symbols_to_ids():
    m = sp500_membership_from_wikipedia(CURRENT_HTML, CHANGES_HTML)
    master = build_instrument_master()
    memb = pd.DataFrame({
        "symbol": ["AAA", "BBB", "CCC", "DDD", "EEE"],
        "effective_from": ["2018-01-02"] * 5,
    })
    master = add_equity_instruments(master, memb, symbol_meta={})
    inst = to_instrument_membership(m, master)
    assert "instrument_id" in inst.columns
    ids = membership_as_of(inst, "sp500", "2024-01-01")
    assert all(i.startswith("EQ:") for i in ids)
    assert len(ids) == 5


# ------------------------------------------------------------- crypto top-N
CRYPTO_IDS = {f"CR:{s}:2017-01-01": "crypto"
              for s in ("BTC", "ETH", "SOL", "ADA", "DOGE")}


def test_crypto_membership_size_and_warmup():
    prices = make_gbm_prices(CRYPTO_IDS, start="2018-01-02", end="2018-12-31")
    mem = crypto_membership(prices, top_n=3, min_listing_days=90)
    assert set(mem.columns) == {"instrument_id", "universe", "effective_from", "effective_to"}

    # Before any name is listed >= 90 days, membership is empty.
    assert membership_as_of(mem, "crypto", "2018-02-01") == []
    # After warm-up, exactly top_n members on any refresh-covered date.
    members = membership_as_of(mem, "crypto", "2018-07-15")
    assert len(members) == 3
    assert all(i in CRYPTO_IDS for i in members)


def _controlled_crypto_prices():
    """Four steady names + one 'spiker' that explodes on a refresh day (2020-05-01)."""
    dates = pd.date_range("2020-01-01", "2020-06-30", freq="D")
    avail = dates.tz_localize("UTC")

    def mk(iid, dv):
        return pd.DataFrame({
            "obs_date": dates, "instrument_id": iid, "close": 100.0,
            "volume": dv / 100.0, "dollar_volume": float(dv),
            "available_from": avail, "source": "synthetic", "ingested_at": avail,
        })

    frames = [mk("CR:AAA:2017-01-01", 100.0), mk("CR:BBB:2017-01-01", 90.0),
              mk("CR:CCC:2017-01-01", 80.0), mk("CR:LOW:2017-01-01", 10.0)]
    spk = mk("CR:SPK:2017-01-01", 10.0)
    spk.loc[spk["obs_date"] == pd.Timestamp("2020-05-01"), "dollar_volume"] = 1e12
    frames.append(spk)
    return pd.concat(frames, ignore_index=True)


def test_crypto_membership_uses_shifted_volume():
    prices = _controlled_crypto_prices()
    refresh = pd.Timestamp("2020-05-01")  # the spike lands exactly on this refresh day

    mem = crypto_membership(prices, top_n=3, min_listing_days=90)
    members = membership_as_of(mem, "crypto", refresh)

    # Shifted (t-1) ranking excludes the same-day spike -> SPK stays out; the three
    # steadily-high names are the universe.
    assert members == ["CR:AAA:2017-01-01", "CR:BBB:2017-01-01", "CR:CCC:2017-01-01"]
    assert "CR:SPK:2017-01-01" not in members

    # Prove the test is non-vacuous: had the ranking used same-day (obs_date <= R)
    # volume, the 1e12 spike would have made SPK the #1 name and forced it in.
    same_day = prices[(prices["obs_date"] <= refresh)
                      & (prices["obs_date"] >= refresh - pd.Timedelta(days=30))]
    naive_rank = same_day.groupby("instrument_id")["dollar_volume"].mean().sort_values(ascending=False)
    assert naive_rank.index[0] == "CR:SPK:2017-01-01"


# ----------------------------------------------------------- instrument master
def test_master_schema_and_uniqueness():
    m = build_instrument_master()
    assert list(m.columns) == INSTRUMENT_COLUMNS
    assert m["instrument_id"].is_unique
    # 25 crypto + 8 fx + 12 commodity + 8 rates + 12 intl + 11 sector = 76 static.
    assert len(m) == 76
    assert set(m["sleeve"]) == {
        "crypto", "fx_etf", "commodity_etf", "rates_etf", "intl_etf", "sector_etf"}
    # Majors get the earlier listing date.
    assert "CR:BTC:2015-01-01" in set(m["instrument_id"])
    assert "CR:SOL:2017-01-01" in set(m["instrument_id"])
    # ETF proxy_of documents the underlying.
    assert m.loc[m["symbol"] == "GLD", "proxy_of"].iloc[0] == "GOLD_SPOT"
    assert m.loc[m["symbol"] == "FXE", "valid_from"].iloc[0] == pd.Timestamp("2005-12-09")


def test_crypto_vendor_symbols():
    m = build_instrument_master()
    assert vendor_symbol(m, "CR:BTC:2015-01-01", "ccxt") == "BTC/USD"
    assert vendor_symbol(m, "CR:BTC:2015-01-01", "coingecko") == "bitcoin"
    assert vendor_symbol(m, "CR:BTC:2015-01-01", "alpaca") == "BTCUSD"
    assert vendor_symbol(m, "CR:BTC:2015-01-01", "no_such_vendor") is None
    assert vendor_symbol(m, "CR:NOPE:2017-01-01", "ccxt") is None


def test_equity_vendor_symbol_dash_convention():
    m = build_instrument_master()
    memb = pd.DataFrame({"symbol": ["BRK.B", "AAA"],
                         "effective_from": ["2016-03-01", "2010-06-01"]})
    m = add_equity_instruments(m, memb, {"BRK.B": {"sector": "Financials"}})
    iid = "EQ:BRK.B:2016-03-01"
    assert vendor_symbol(m, iid, "yfinance") == "BRK-B"   # "." -> "-"
    assert vendor_symbol(m, iid, "stooq") == "BRK-B.US"
    assert m.loc[m["instrument_id"] == iid, "sector"].iloc[0] == "Financials"


def test_equity_first_seen_date_is_earliest():
    m = build_instrument_master()
    # Same symbol entering twice -> id keyed on the EARLIEST appearance.
    memb = pd.DataFrame({"symbol": ["AAA", "AAA"],
                         "effective_from": ["2015-01-05", "2010-06-01"]})
    m = add_equity_instruments(m, memb, {})
    eq = m[m["sleeve"] == "equity"]
    assert list(eq["instrument_id"]) == ["EQ:AAA:2010-06-01"]


def test_resolve_vendor_symbol_pit_window_match():
    """A reused vendor ticker resolves to different instruments by date."""
    # Two instruments that shared the alpaca ticker "RUSE" across disjoint windows.
    rows = [
        {
            "instrument_id": "EQ:RUSE:2005-01-03", "asset_class": "equity",
            "sleeve": "equity", "symbol": "RUSE",
            "vendor_symbols": json.dumps({"alpaca": "RUSE"}),
            "currency": "USD", "valid_from": pd.Timestamp("2005-01-03"),
            "valid_to": pd.Timestamp("2015-06-30"), "proxy_of": None,
            "sector": None, "meta": "{}",
        },
        {
            "instrument_id": "EQ:RUSE:2018-02-01", "asset_class": "equity",
            "sleeve": "equity", "symbol": "RUSE",
            "vendor_symbols": json.dumps({"alpaca": "RUSE"}),
            "currency": "USD", "valid_from": pd.Timestamp("2018-02-01"),
            "valid_to": pd.NaT, "proxy_of": None, "sector": None, "meta": "{}",
        },
    ]
    master = pd.DataFrame(rows, columns=INSTRUMENT_COLUMNS)

    assert resolve_vendor_symbol(master, "alpaca", "RUSE", "2010-01-01") == "EQ:RUSE:2005-01-03"
    assert resolve_vendor_symbol(master, "alpaca", "RUSE", "2020-01-01") == "EQ:RUSE:2018-02-01"
    # In the gap between the two windows nothing resolves.
    assert resolve_vendor_symbol(master, "alpaca", "RUSE", "2016-01-01") is None
    assert resolve_vendor_symbol(master, "alpaca", "UNKNOWN", "2020-01-01") is None


# ------------------------------------------------------------------- static
def test_static_membership_from_inception():
    fx = static_membership("fx_etf")
    assert (fx["universe"] == "fx_etf").all()
    assert set(fx["instrument_id"]) == {
        "FX:FXE:2005-12-09", "FX:FXY:2007-02-12", "FX:FXB:2006-06-21",
        "FX:FXA:2006-06-21", "FX:FXC:2006-06-21", "FX:FXF:2006-06-21",
        "FX:UUP:2007-02-20", "FX:UDN:2007-02-20",
    }
    # FXE is a member from inception, and not before.
    assert membership_as_of(fx, "fx_etf", "2005-12-31") == ["FX:FXE:2005-12-09"]
    assert "FX:FXE:2005-12-09" not in membership_as_of(fx, "fx_etf", "2005-12-01")


# ------------------------------------------------------------------- hygiene
def test_blocklist_windows():
    assert hygiene.is_blocked("FB", "2022-07-01") is True     # after Meta rename
    assert hygiene.is_blocked("FB", "2021-01-01") is False    # legit Facebook era
    assert hygiene.is_blocked("TWTR", "2023-01-01") is True   # after delisting
    assert hygiene.is_blocked("TWTR", "2020-01-01") is False
    assert hygiene.is_blocked("AAPL", "2023-01-01") is False  # never blocked
    # Boundary is inclusive at the event date.
    assert hygiene.is_blocked("FB", "2022-06-09") is True


def test_price_backstop_drops_sub_floor_rows():
    df = pd.DataFrame({
        "obs_date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"]),
        "instrument_id": ["EQ:X:2000-01-03"] * 4,
        "close": [50.0, 0.05, float("nan"), 0.10],
        "volume": [1, 2, 3, 4],
    })
    out = hygiene.apply_price_backstop(df, min_price=0.10)
    # Drops the $0.05 row and the NaN row; keeps $50 and exactly-$0.10 (>= floor).
    assert list(out["close"]) == [50.0, 0.10]
    # Other columns preserved.
    assert list(out["volume"]) == [1, 4]
    assert "instrument_id" in out.columns


def test_price_backstop_empty_and_missing_close():
    assert hygiene.apply_price_backstop(pd.DataFrame()).empty
    noclose = pd.DataFrame({"obs_date": [pd.Timestamp("2020-01-01")], "x": [1]})
    pd.testing.assert_frame_equal(hygiene.apply_price_backstop(noclose), noclose)


def test_apply_hygiene_drops_penny_and_blocklisted_rows():
    df = pd.DataFrame({
        "obs_date": pd.to_datetime([
            "2020-01-02",  # AAPL, kept
            "2020-01-03",  # penny -> backstop drop
            "2022-07-01",  # FB after Meta rename -> blocklist drop
            "2021-01-04",  # FB legit Facebook era -> kept
        ]),
        "symbol": ["AAPL", "AAPL", "FB", "FB"],
        "close": [150.0, 0.05, 200.0, 250.0],
    })
    clean, counts = hygiene.apply_hygiene(df, symbol_col="symbol")
    assert counts == {"backstop_dropped": 1, "blocklist_dropped": 1,
                      "flap_dropped": 0, "corrupt_series": []}
    # Exactly the two legitimate rows survive, in order.
    assert list(clean["close"]) == [150.0, 250.0]
    assert list(clean["symbol"]) == ["AAPL", "FB"]


def test_apply_hygiene_counts_are_independent():
    # Only a penny row, no blocklisted symbol -> blocklist counter stays 0.
    df = pd.DataFrame({
        "obs_date": pd.to_datetime(["2020-01-02", "2020-01-03"]),
        "symbol": ["AAPL", "AAPL"],
        "close": [10.0, 0.01],
    })
    clean, counts = hygiene.apply_hygiene(df, symbol_col="symbol")
    assert counts == {"backstop_dropped": 1, "blocklist_dropped": 0,
                      "flap_dropped": 0, "corrupt_series": []}
    assert list(clean["close"]) == [10.0]

    # Only a blocklisted row, no penny -> backstop counter stays 0.
    df2 = pd.DataFrame({
        "obs_date": pd.to_datetime(["2021-01-04", "2022-07-01"]),
        "symbol": ["FB", "FB"],
        "close": [250.0, 200.0],
    })
    clean2, counts2 = hygiene.apply_hygiene(df2, symbol_col="symbol")
    assert counts2 == {"backstop_dropped": 0, "blocklist_dropped": 1,
                       "flap_dropped": 0, "corrupt_series": []}
    assert list(clean2["close"]) == [250.0]


def test_apply_hygiene_symbol_col_absent_skips_blocklist():
    # No symbol column: blocklist step is skipped (counted 0); backstop still runs.
    df = pd.DataFrame({
        "obs_date": pd.to_datetime(["2022-07-01", "2020-01-03"]),
        "instrument_id": ["EQ:FB:2012-05-18", "EQ:X:2000-01-03"],
        "close": [200.0, 0.05],
    })
    clean, counts = hygiene.apply_hygiene(df, symbol_col="symbol")
    assert counts == {"backstop_dropped": 1, "blocklist_dropped": 0,
                      "flap_dropped": 0, "corrupt_series": []}
    # The blocklisted FB row survives because there was no symbol column to date it.
    assert list(clean["close"]) == [200.0]


def test_apply_hygiene_empty_frame():
    clean, counts = hygiene.apply_hygiene(pd.DataFrame(), symbol_col="symbol")
    assert clean.empty
    assert counts == {"backstop_dropped": 0, "blocklist_dropped": 0,
                      "flap_dropped": 0, "corrupt_series": []}


def test_flap_screen_drops_regime_flapping_series():
    """Regression (first live French validation, 2026-07-07): dead tickers' vendor
    series flap between price regimes — EQ:TIE oscillated $2 <-> $15,600 day-to-day,
    creating fake +810,000% returns that gave an equal-weight market 1,750% annualized
    vol and ~0.02 correlation with Mkt-RF, and poisoned the equity factors' gate
    validation books. Flap rows sit 20x+ from the rolling median; drop them."""
    dates = pd.date_range("2016-03-01", periods=12, freq="B")
    closes = [2.0, 1.99, 15600.0, 2.01, 2.0, 13400.0, 1.95, 2.02, 2.0, 14500.0, 2.0, 2.01]
    df = pd.DataFrame({"obs_date": dates, "instrument_id": "EQ:TIE:1976-07-01",
                       "close": closes})
    out = hygiene.apply_flap_screen(df)
    assert (out["close"] < 100).all()          # every flap row gone
    assert len(out) == 9                       # the three spikes dropped


def test_flap_screen_keeps_genuine_violent_repricings():
    """GME-style squeeze: large but PERSISTENT moves drag the rolling median with
    them within days, so ratios stay far under the 20x threshold. Real volatility
    must survive; only reverting flaps die."""
    dates = pd.date_range("2021-01-11", periods=12, freq="B")
    closes = [20.0, 20.0, 31.0, 35.0, 39.0, 43.0, 65.0, 88.0, 148.0, 348.0, 194.0, 325.0]
    df = pd.DataFrame({"obs_date": dates, "instrument_id": "EQ:GME:2007-12-13",
                       "close": closes})
    out = hygiene.apply_flap_screen(df)
    assert len(out) == len(df)                 # nothing dropped


def test_flap_screen_passes_short_series_and_counts_in_apply_hygiene():
    dates = pd.date_range("2024-01-02", periods=3, freq="B")
    short = pd.DataFrame({"obs_date": dates, "instrument_id": "EQ:NEW:2024-01-02",
                          "close": [10.0, 10000.0, 10.0], "symbol": "NEW"})
    out = hygiene.apply_flap_screen(short)
    assert len(out) == 3                       # < window obs: nothing to compare against

    dates = pd.date_range("2016-03-01", periods=8, freq="B")
    flappy = pd.DataFrame({"obs_date": dates, "instrument_id": "EQ:TIE:1976-07-01",
                           "close": [2.0, 2.0, 15600.0, 2.0, 2.0, 2.0, 2.0, 2.0],
                           "symbol": "TIE"})
    clean, counts = hygiene.apply_hygiene(flappy)
    assert counts["flap_dropped"] == 1
    assert (clean["close"] < 100).all()


def test_corrupt_series_run_alternator_dropped_wholesale():
    """Regression (2026-07-07): EQ:TIE alternates price regimes in multi-day RUNS
    ($2 stretches <-> $13,400 stretches), so run interiors survive any row-level
    median screen. A series with >5 catastrophic (>400%) day-over-day moves is
    corrupt end-to-end and must be dropped wholesale."""
    dates = pd.date_range("2016-03-01", periods=20, freq="B")
    closes = ([2.0] * 3 + [13400.0] * 3 + [2.0] * 2 + [13400.0] * 4
              + [2.0] * 3 + [13400.0] * 2 + [2.0] * 3)   # 6 regime flips
    tie = pd.DataFrame({"obs_date": dates, "instrument_id": "EQ:TIE:1976-07-01",
                        "close": closes, "symbol": "TIE"})
    # A normal name alongside proves the drop is per-series, not global.
    ok = pd.DataFrame({"obs_date": dates, "instrument_id": "EQ:AAA:2000-01-03",
                       "close": 100.0 + pd.RangeIndex(20) * 0.5, "symbol": "AAA"})
    clean, counts = hygiene.apply_hygiene(pd.concat([tie, ok], ignore_index=True))
    assert counts["corrupt_series"] == ["EQ:TIE:1976-07-01"]
    assert set(clean["instrument_id"]) == {"EQ:AAA:2000-01-03"}


def test_corrupt_series_spares_squeeze_and_crash_names():
    """GME-style squeeze (one +135% day) and a single -75% biotech crash are far
    below both the 400% move size and the >5 repetition bar — never dropped."""
    dates = pd.date_range("2021-01-11", periods=10, freq="B")
    gme = pd.DataFrame({"obs_date": dates, "instrument_id": "EQ:GME:2007-12-13",
                        "close": [20, 39, 43, 65, 88, 148, 348, 194, 325, 225.0]})
    out, corrupt = hygiene.drop_corrupt_series(gme)
    assert corrupt == []
    assert len(out) == len(gme)
