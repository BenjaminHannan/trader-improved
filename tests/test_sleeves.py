"""Tests for the three breadth sleeves (rates_etf, intl_etf, sector_etf) and the
CurveCarry signal.

No network. The instrument master, static membership, config loaders, and the new
curve-carry signal (known-answer + availability-lag) are all exercised against
hand-built frames.
"""
from __future__ import annotations

import pandas as pd

from production.alpha.registry import FactorRegistry
from production.core.config import costs_config, factors_config, risk_config
from production.reference.instruments import build_instrument_master
from production.reference.universe import membership_as_of, static_membership
from production.signals.carry import CurveCarry, RATES_CARRY_SERIES

UTC = "UTC"

# Expected static membership per new sleeve: {instrument_id: inception}.
RATES_EXPECT = {
    "RT:TLT:2002-07-26": "2002-07-26", "RT:IEF:2002-07-26": "2002-07-26",
    "RT:SHY:2002-07-26": "2002-07-26", "RT:LQD:2002-07-26": "2002-07-26",
    "RT:HYG:2007-04-11": "2007-04-11", "RT:EMB:2007-12-19": "2007-12-19",
    "RT:TIP:2003-12-05": "2003-12-05", "RT:BNDX:2013-06-04": "2013-06-04",
}
INTL_EXPECT = {
    "IE:EWJ:1996-03-12": "1996-03-12", "IE:EWG:1996-03-12": "1996-03-12",
    "IE:EWU:1996-03-12": "1996-03-12", "IE:EWQ:1996-03-12": "1996-03-12",
    "IE:EWA:1996-03-12": "1996-03-12", "IE:EWC:1996-03-12": "1996-03-12",
    "IE:EWZ:2000-07-10": "2000-07-10", "IE:INDA:2012-02-02": "2012-02-02",
    "IE:FXI:2004-10-05": "2004-10-05", "IE:EWY:2000-05-09": "2000-05-09",
    "IE:EWT:2000-06-20": "2000-06-20", "IE:EWW:1996-03-12": "1996-03-12",
}
SECTOR_EXPECT = {
    "SE:XLK:1998-12-16": "1998-12-16", "SE:XLF:1998-12-16": "1998-12-16",
    "SE:XLE:1998-12-16": "1998-12-16", "SE:XLV:1998-12-16": "1998-12-16",
    "SE:XLI:1998-12-16": "1998-12-16", "SE:XLP:1998-12-16": "1998-12-16",
    "SE:XLY:1998-12-16": "1998-12-16", "SE:XLU:1998-12-16": "1998-12-16",
    "SE:XLB:1998-12-16": "1998-12-16", "SE:XLRE:2015-10-07": "2015-10-07",
    "SE:XLC:2018-06-18": "2018-06-18",
}
SLEEVE_EXPECT = {"rates_etf": RATES_EXPECT, "intl_etf": INTL_EXPECT,
                 "sector_etf": SECTOR_EXPECT}


# ------------------------------------------------------------- instrument master
def test_master_contains_new_sleeves_with_inception_dates():
    m = build_instrument_master()
    assert m["instrument_id"].is_unique
    for sleeve, expect in SLEEVE_EXPECT.items():
        sub = m[m["sleeve"] == sleeve]
        assert set(sub["instrument_id"]) == set(expect)
        for iid, inception in expect.items():
            row = m[m["instrument_id"] == iid].iloc[0]
            assert row["valid_from"] == pd.Timestamp(inception)
            assert pd.isna(row["proxy_of"])           # not a proxy — traded directly
    # asset_class carried through from universe.yaml.
    assert m.loc[m["sleeve"] == "rates_etf", "asset_class"].iloc[0] == "rates"
    assert m.loc[m["sleeve"] == "intl_etf", "asset_class"].iloc[0] == "intl_equity"
    assert m.loc[m["sleeve"] == "sector_etf", "asset_class"].iloc[0] == "sector_equity"


def test_new_sleeve_vendor_symbols_follow_etf_convention():
    m = build_instrument_master()
    from production.reference.instruments import vendor_symbol
    assert vendor_symbol(m, "RT:TLT:2002-07-26", "yfinance") == "TLT"
    assert vendor_symbol(m, "RT:TLT:2002-07-26", "stooq") == "TLT.US"
    assert vendor_symbol(m, "SE:XLK:1998-12-16", "alpaca") == "XLK"


# ------------------------------------------------------------- static membership
def test_static_membership_each_new_sleeve():
    for sleeve, expect in SLEEVE_EXPECT.items():
        mem = static_membership(sleeve)
        assert (mem["universe"] == sleeve).all()
        assert set(mem["instrument_id"]) == set(expect)

    # A member from inception onward, not before.
    rates = static_membership("rates_etf")
    assert "RT:BNDX:2013-06-04" in membership_as_of(rates, "rates_etf", "2013-06-04")
    assert "RT:BNDX:2013-06-04" not in membership_as_of(rates, "rates_etf", "2013-06-01")
    # Sector: XLC is the newest; before its inception only the older names are live.
    sector = membership_as_of(static_membership("sector_etf"), "sector_etf", "2016-01-01")
    assert "SE:XLC:2018-06-18" not in sector
    assert "SE:XLRE:2015-10-07" in sector


# --------------------------------------------------------------- CurveCarry
def _macro_row(series_id, obs, value, avail):
    avail = pd.Timestamp(avail, tz=UTC)
    return {"obs_date": pd.Timestamp(obs), "series_id": series_id, "value": value,
            "available_from": avail, "source": "test",
            "ingested_at": avail + pd.Timedelta(minutes=5)}


def _tlt_prices(start="2020-01-06", end="2020-04-30"):
    dates = pd.bdate_range(start, end)
    avail = dates.tz_localize(UTC) + pd.Timedelta(hours=21, minutes=30)
    return pd.DataFrame({
        "obs_date": dates, "instrument_id": "RT:TLT:2002-07-26",
        "close": 100.0, "volume": 1e6, "dollar_volume": 1e8,
        "available_from": avail, "source": "test", "ingested_at": avail,
    })


def test_curve_carry_known_answer_yield_spread_exact():
    """Constant DGS20 and DGS3MO_US -> carry == long - cash exactly, every date."""
    prices = _tlt_prices()
    macro = pd.DataFrame([
        _macro_row("DGS20", "2020-01-06", 3.5, "2020-01-06 12:00"),
        _macro_row("DGS3MO_US", "2020-01-06", 1.25, "2020-01-06 12:00"),
    ])
    out = CurveCarry().compute({"prices": prices, "macro": macro})
    assert not out.empty
    # min_history_days=21 floors the first emitted date; every value is the exact spread.
    assert (out["value"] == 3.5 - 1.25).all()
    assert (out["instrument_id"] == "RT:TLT:2002-07-26").all()


def test_curve_carry_respects_availability_lag():
    """A DGS20 jump knowable only from 2020-03-16 cannot move an earlier value."""
    prices = _tlt_prices()
    macro = pd.DataFrame([
        _macro_row("DGS3MO_US", "2020-02-03", 1.0, "2020-02-03 12:00"),
        _macro_row("DGS20", "2020-02-03", 3.0, "2020-02-03 12:00"),
        # A later print that only becomes knowable two weeks after its obs_date.
        _macro_row("DGS20", "2020-03-02", 5.0, "2020-03-16 12:00"),
    ])
    out = CurveCarry().compute({"prices": prices, "macro": macro})
    s = out.set_index("obs_date")["value"]

    # 2020-03-09: after the 5.0 obs_date but BEFORE its availability -> still 3.0.
    assert s.loc[pd.Timestamp("2020-03-09")] == 3.0 - 1.0
    # 2020-03-16: the revision is now knowable -> the value jumps.
    assert s.loc[pd.Timestamp("2020-03-16")] == 5.0 - 1.0


def test_curve_carry_series_map_covers_all_rates_symbols():
    """Every rates-ETF symbol in the master has a carry series mapping."""
    m = build_instrument_master()
    symbols = set(m.loc[m["sleeve"] == "rates_etf", "symbol"])
    assert symbols <= set(RATES_CARRY_SERIES)


# ----------------------------------------------------------------- config gates
def test_factors_config_parses_and_carry_curve_resolves():
    cfg = factors_config()                       # validates the whole file
    assert "carry_curve" in cfg["factors"]
    spec = cfg["factors"]["carry_curve"]
    assert spec["sleeves"] == ["rates_etf"]
    # Status changes once the real gate runs; the invariant is a VALID status.
    assert spec["status"] in ("candidate", "accepted", "rejected")
    # Sleeve extensions landed.
    assert set(cfg["factors"]["tsmom"]["sleeves"]) == {
        "fx_etf", "commodity_etf", "rates_etf", "intl_etf"}
    assert set(cfg["factors"]["mom_12_1"]["sleeves"]) >= {
        "rates_etf", "intl_etf", "sector_etf"}
    assert set(cfg["factors"]["str_reversal_1m"]["sleeves"]) >= {"intl_etf", "sector_etf"}
    # n_trials unchanged by the new candidate.
    # The ledger only ever grows; 7 was the pre-live-gate count (2026-07-07).
    assert cfg["n_trials"] >= 7

    reg = FactorRegistry()
    assert reg.signal_class("carry_curve") is CurveCarry


def test_costs_and_risk_configs_accept_new_sleeves():
    costs = costs_config()
    for sleeve, floor in (("rates_etf", 5.0), ("intl_etf", 5.0), ("sector_etf", 5.0)):
        assert costs["sleeves"][sleeve]["floor_bps"] == floor
    assert costs["sleeves"]["intl_etf"]["half_spread_bps"] == 2.5

    risk = risk_config()
    assert {"rates_etf", "intl_etf", "sector_etf"} <= set(risk["covariance_sleeves"])
