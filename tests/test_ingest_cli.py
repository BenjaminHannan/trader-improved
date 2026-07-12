"""Ingest CLI orchestration tests — NO NETWORK.

Covers the three orchestration fixes in ``scripts/ingest.py``:

1. ``_build_universe`` mints equities from the S&P 500 walk-back and writes a *nonzero*
   ``membership_equity`` (failing loudly if it would be empty), plus all seven sleeves.
2. The ``prices`` dataset fans out across every price sleeve when no ``--sleeve`` is
   given, and scopes to one sleeve when it is.
3. Equity price symbols are derived from the instrument master (the equity sleeve has no
   static ``symbols`` list), and the secondary ``stooq`` feed is wired over the same
   universe.

Wikipedia parsing is exercised with injected fixture HTML (same shape as
``tests/test_universe.py``); the network-fetching reference functions are monkeypatched.
"""
from __future__ import annotations

import json

import pandas as pd
import pytest

import scripts.ingest as ingest
from production.core.lake import Lake
from production.reference import universe as ref_universe

# Reuse the two-table shape from tests/test_universe.py: a current-constituents table
# carrying a GICS Sector column, and a Selected-changes table.
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

CHANGES_HTML = """
<table>
<tr><th>Date</th><th colspan="2">Added</th><th colspan="2">Removed</th><th>Reason</th></tr>
<tr><th>Date</th><th>Ticker</th><th>Security</th><th>Ticker</th><th>Security</th><th>Reason</th></tr>
<tr><td>June 1, 2023</td><td>CCC</td><td>Gamma Inc</td><td>XXX</td><td>Ex Corp</td><td>r1</td></tr>
<tr><td>March 15, 2022</td><td>DDD</td><td>Delta Inc</td><td>YYY</td><td>Why Corp</td><td>r2</td></tr>
</table>
"""


# Capture the unpatched parsers so monkeypatched replacements can delegate to them
# without recursing through the patched module attribute.
_REAL_MEMBERSHIP = ref_universe.sp500_membership_from_wikipedia
_REAL_SECTOR_MAP = ref_universe.sector_map_from_wikipedia


def _membership() -> pd.DataFrame:
    return _REAL_MEMBERSHIP(CURRENT_HTML, CHANGES_HTML)


def _sector_map() -> dict[str, str]:
    return _REAL_SECTOR_MAP(CURRENT_HTML)


# --------------------------------------------------------- Bug 1: _build_universe
def test_build_universe_writes_nonzero_equity_membership(tmp_path, monkeypatch):
    # Inject fixture HTML for both scrapes (no network).
    monkeypatch.setattr(ref_universe, "sp500_membership_from_wikipedia",
                        lambda *a, **k: _membership())
    monkeypatch.setattr(ref_universe, "sector_map_from_wikipedia",
                        lambda *a, **k: _sector_map())

    lake = Lake(str(tmp_path))
    ingest._build_universe(lake)

    master = lake.read_reference("instruments")
    eq = master[master["sleeve"] == "equity"]
    assert not eq.empty
    # The current members carry their GICS sector threaded from the current table.
    sec = dict(zip(eq["symbol"], eq["sector"]))
    assert sec["AAA"] == "Technology"
    assert sec["CCC"] == "Health Care"

    memb = lake.read_reference("membership_equity")
    assert not memb.empty
    assert (memb["instrument_id"].str.startswith("EQ:")).all()

    # All seven price sleeves have a membership table; the three new ETF sleeves too.
    for sleeve in ("equity", "crypto", "fx_etf", "commodity_etf",
                   "rates_etf", "intl_etf", "sector_etf"):
        assert not lake.read_reference(f"membership_{sleeve}").empty


def test_build_universe_fails_loudly_on_empty_membership(tmp_path, monkeypatch):
    # An empty walk-back (e.g. a broken scrape) mints no equities, so
    # to_instrument_membership yields nothing. The builder must raise, never write a
    # silent empty table.
    empty = pd.DataFrame(columns=["symbol", "universe", "effective_from", "effective_to"])
    monkeypatch.setattr(ref_universe, "sp500_membership_from_wikipedia",
                        lambda *a, **k: empty)
    monkeypatch.setattr(ref_universe, "sector_map_from_wikipedia", lambda *a, **k: {})

    lake = Lake(str(tmp_path))
    with pytest.raises(SystemExit):
        ingest._build_universe(lake)


# --------------------------------------------- Bug 2: prices fans out to all sleeves
def test_prices_expands_to_all_seven_sleeves(monkeypatch):
    """With no --sleeve, `--dataset prices` builds one loader per price sleeve."""
    built: list[tuple[str, str | None]] = []

    class _FakeRes:
        vendor, rows = "fake", 0
        start = end = pd.Timestamp("2020-01-01")
        audit = {"warnings": None}

    class _FakeLoader:
        def run(self, *a, **k):
            return _FakeRes()

    def _fake_make_loader(ds, sleeve, lake, instruments):
        built.append((ds, sleeve))
        return _FakeLoader()

    monkeypatch.setattr(ingest, "_make_loader", _fake_make_loader)
    monkeypatch.setattr(ingest, "_load_instruments", lambda lake: None)

    rc = ingest.main(["--dataset", "prices", "--lake-root", "."])
    assert rc == 0
    assert built == [("prices", s) for s in ingest._PRICE_SLEEVES]


def test_prices_scopes_to_single_sleeve_when_given(monkeypatch):
    built: list[tuple[str, str | None]] = []

    class _FakeRes:
        vendor, rows = "fake", 0
        start = end = pd.Timestamp("2020-01-01")
        audit = {"warnings": None}

    def _fake_make_loader(ds, sleeve, lake, instruments):
        built.append((ds, sleeve))
        return type("L", (), {"run": lambda self, *a, **k: _FakeRes()})()

    monkeypatch.setattr(ingest, "_make_loader", _fake_make_loader)
    monkeypatch.setattr(ingest, "_load_instruments", lambda lake: None)

    ingest.main(["--dataset", "prices", "--sleeve", "crypto", "--lake-root", "."])
    assert built == [("prices", "crypto")]


# ------------------------------------- Bug 3: equity symbols derived from the master
def _master_with_equities() -> pd.DataFrame:
    def eq(sym):
        return dict(
            instrument_id=f"EQ:{sym}:2015-01-01", asset_class="equity", sleeve="equity",
            symbol=sym, vendor_symbols=json.dumps(
                {"yfinance": sym.replace(".", "-"), "stooq": f"{sym.replace('.', '-')}.US",
                 "alpaca": sym}),
            currency="USD", valid_from="2015-01-01", valid_to=None,
            proxy_of=None, sector="Technology", meta="{}",
        )
    return pd.DataFrame([eq("AAA"), eq("BRK.B")])


def test_equity_symbols_from_master():
    master = _master_with_equities()
    assert ingest._equity_symbols(master, "yfinance") == ["AAA", "BRK-B"]
    assert ingest._equity_symbols(master, "stooq") == ["AAA.US", "BRK-B.US"]
    # No master -> empty (loaders degrade to a coverage warning, never crash).
    assert ingest._equity_symbols(None, "yfinance") == []


def test_prices_loader_uses_master_equity_symbols():
    master = _master_with_equities()
    loader = ingest._make_loader("prices", "equity", Lake("."), master)
    assert loader.symbols == ["AAA", "BRK-B"]


def test_stooq_loader_wires_secondary_feed():
    master = _master_with_equities()
    loader = ingest._make_loader("stooq", None, Lake("."), master)
    assert loader.vendor == "stooq"
    assert "AAA.US" in loader.symbols and "BRK-B.US" in loader.symbols
