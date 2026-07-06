"""Instrument master — the synthetic-key backbone that kills ticker-reuse bugs.

Every tradable is identified by a synthetic `instrument_id` of the form
`CLASS:SYMBOL:first-listing-date` (e.g. ``CR:BTC:2015-01-01``). That key is *never*
reused: when a vendor recycles a ticker (FB Facebook -> later reuse, TWTR delisted,
an equity symbol handed to a different company), the synthetic key stays distinct and
the reused vendor symbol resolves to the correct instrument *by date* through the
per-row validity window. This is the schema-level defence the reference `trader` repo
lacked.

One row per (entity, validity window). Vendor symbols are stored as a JSON string so
the master survives a round-trip through the parquet lake without column explosion.
"""
from __future__ import annotations

import json

import pandas as pd

from production.core.config import universe_config

# The master schema. `vendor_symbols` and `meta` are JSON strings (see module docstring).
INSTRUMENT_COLUMNS = [
    "instrument_id",
    "asset_class",
    "sleeve",
    "symbol",
    "vendor_symbols",
    "currency",
    "valid_from",
    "valid_to",
    "proxy_of",
    "sector",
    "meta",
]

# class prefix per sleeve — the first field of every instrument_id.
CLASS_PREFIX = {
    "equity": "EQ",
    "crypto": "CR",
    "fx_etf": "FX",
    "commodity_etf": "CO",
}

# Default first-listing date for crypto pairs; the two majors predate the rest.
CRYPTO_DEFAULT_LISTING = "2017-01-01"
CRYPTO_LISTING_OVERRIDES = {"BTC": "2015-01-01", "ETH": "2015-01-01"}

# CoinGecko coin ids (their API keys on a slug, not the ticker). Fallback = lowercase
# ticker for any symbol not enumerated here.
COINGECKO_IDS = {
    "BTC": "bitcoin", "ETH": "ethereum", "SOL": "solana", "ADA": "cardano",
    "DOGE": "dogecoin", "LTC": "litecoin", "BCH": "bitcoin-cash", "LINK": "chainlink",
    "UNI": "uniswap", "AAVE": "aave", "AVAX": "avalanche-2", "DOT": "polkadot",
    "MATIC": "matic-network", "SHIB": "shiba-inu", "XRP": "ripple", "ATOM": "cosmos",
    "ALGO": "algorand", "XLM": "stellar", "ETC": "ethereum-classic", "FIL": "filecoin",
    "NEAR": "near", "GRT": "the-graph", "CRV": "curve-dao-token", "MKR": "maker",
    "SUSHI": "sushi",
}

# Real first-trade (inception) dates for the FX and commodity ETF proxies. These are
# static, non-negotiable facts baked into the master so static membership can be derived
# without a vendor round-trip.
ETF_INCEPTION = {
    # fx_etf
    "FXE": "2005-12-09", "FXY": "2007-02-12", "FXB": "2006-06-21", "FXA": "2006-06-21",
    "FXC": "2006-06-21", "FXF": "2006-06-21", "UUP": "2007-02-20", "UDN": "2007-02-20",
    # commodity_etf
    "GLD": "2004-11-18", "SLV": "2006-04-21", "USO": "2006-04-10", "UNG": "2007-04-18",
    "DBA": "2007-01-05", "DBB": "2007-01-05", "CPER": "2011-11-15", "PPLT": "2010-01-08",
    "PALL": "2010-01-08", "CORN": "2010-06-09", "WEAT": "2011-09-19", "SOYB": "2011-09-19",
}


def _to_dash(symbol: str) -> str:
    """yfinance/stooq class-share convention: BRK.B -> BRK-B."""
    return symbol.replace(".", "-")


def _crypto_vendor_symbols(symbol: str) -> dict[str, str]:
    return {
        "ccxt": f"{symbol}/USD",
        "coingecko": COINGECKO_IDS.get(symbol, symbol.lower()),
        "alpaca": f"{symbol}USD",
    }


def _etf_vendor_symbols(symbol: str) -> dict[str, str]:
    dash = _to_dash(symbol)
    return {
        "yfinance": dash,
        "stooq": f"{dash}.US",
        "alpaca": symbol,
    }


def _equity_vendor_symbols(symbol: str) -> dict[str, str]:
    dash = _to_dash(symbol)
    return {
        "yfinance": dash,
        "stooq": f"{dash}.US",
        "alpaca": symbol,
    }


def _row(instrument_id, asset_class, sleeve, symbol, vendor_symbols, currency,
         valid_from, valid_to, proxy_of, sector, meta) -> dict:
    return {
        "instrument_id": instrument_id,
        "asset_class": asset_class,
        "sleeve": sleeve,
        "symbol": symbol,
        "vendor_symbols": json.dumps(vendor_symbols, sort_keys=True),
        "currency": currency,
        "valid_from": pd.Timestamp(valid_from) if valid_from is not None else pd.NaT,
        "valid_to": pd.Timestamp(valid_to) if valid_to is not None else pd.NaT,
        "proxy_of": proxy_of,
        "sector": sector,
        "meta": json.dumps(meta, sort_keys=True),
    }


def build_instrument_master(lake=None) -> pd.DataFrame:
    """Construct the master from ``configs/universe.yaml``.

    Covers the three statically-known sleeves (crypto pairs, fx-ETF proxies,
    commodity-ETF proxies). Equities are a PIT membership problem — their ids are
    minted from the S&P 500 walk-back at ingest time via :func:`add_equity_instruments`,
    so they are intentionally absent here. `lake` is accepted for signature symmetry
    with the rest of the reference package (a caller may persist the returned frame via
    ``lake.write_reference``); this function itself performs no IO.
    """
    cfg = universe_config()
    sleeves = cfg["sleeves"]
    rows: list[dict] = []

    # ------------------------------------------------------------------ crypto
    crypto = sleeves.get("crypto", {})
    crypto_ac = crypto.get("asset_class", "crypto")
    for symbol in crypto.get("symbols", []):
        listing = CRYPTO_LISTING_OVERRIDES.get(symbol, CRYPTO_DEFAULT_LISTING)
        iid = f"{CLASS_PREFIX['crypto']}:{symbol}:{listing}"
        rows.append(_row(
            iid, crypto_ac, "crypto", symbol, _crypto_vendor_symbols(symbol),
            "USD", listing, None, None, None, {"quote": "USD"},
        ))

    # ------------------------------------------------------ fx / commodity ETFs
    for sleeve in ("fx_etf", "commodity_etf"):
        spec = sleeves.get(sleeve, {})
        asset_class = spec.get("asset_class", sleeve)
        prefix = CLASS_PREFIX[sleeve]
        for symbol, underlying in spec.get("symbols", {}).items():
            if symbol not in ETF_INCEPTION:
                raise KeyError(
                    f"no inception date recorded for ETF {symbol!r} in sleeve {sleeve}")
            inception = ETF_INCEPTION[symbol]
            iid = f"{prefix}:{symbol}:{inception}"
            rows.append(_row(
                iid, asset_class, sleeve, symbol, _etf_vendor_symbols(symbol),
                "USD", inception, None, underlying, None,
                {"proxy_underlying": underlying},
            ))

    return pd.DataFrame(rows, columns=INSTRUMENT_COLUMNS)


def add_equity_instruments(master: pd.DataFrame, membership_df: pd.DataFrame,
                           symbol_meta: dict | None = None) -> pd.DataFrame:
    """Mint ``EQ:SYMBOL:first-seen-date`` rows from an S&P 500 membership table.

    `membership_df` is the symbol-keyed frame produced by
    :func:`production.reference.universe.sp500_membership_from_wikipedia` (columns
    ``symbol`` and ``effective_from``; ``effective_to`` optional). Each symbol's
    first-listing date is the *earliest* date it ever entered the index, so a name
    that leaves and rejoins keeps one stable instrument id. In production this runs
    after the Wikipedia scrape; unit tests feed a synthetic membership frame.

    `symbol_meta` maps ``symbol -> {"sector": ..., "currency": ..., "name": ...}``.
    Missing entries default to USD / unknown sector.
    """
    symbol_meta = symbol_meta or {}
    if membership_df.empty:
        return master.copy()

    mf = membership_df.copy()
    mf["effective_from"] = pd.to_datetime(mf["effective_from"])
    # earliest appearance per symbol -> the immutable first-listing date in the id.
    first_seen = mf.groupby("symbol")["effective_from"].min()

    rows: list[dict] = []
    for symbol, first in first_seen.items():
        # A never-observed effective_from (NaT) would poison the id; fall back to the
        # min of the frame's known dates so the key is still deterministic.
        if pd.isna(first):
            first = mf["effective_from"].min()
        listing = pd.Timestamp(first).strftime("%Y-%m-%d")
        iid = f"{CLASS_PREFIX['equity']}:{symbol}:{listing}"
        meta = symbol_meta.get(symbol, {})
        rows.append(_row(
            iid, "equity", "equity", symbol, _equity_vendor_symbols(symbol),
            meta.get("currency", "USD"), listing, None, None,
            meta.get("sector"), {k: v for k, v in meta.items() if k != "sector"},
        ))

    equities = pd.DataFrame(rows, columns=INSTRUMENT_COLUMNS)
    return pd.concat([master, equities], ignore_index=True)


def _select_window(rows: pd.DataFrame) -> pd.Series:
    """Pick the representative row when an id has several validity windows: the open
    (valid_to is NaT) one, else the latest by valid_from."""
    open_rows = rows[rows["valid_to"].isna()]
    if not open_rows.empty:
        return open_rows.iloc[-1]
    return rows.sort_values("valid_from").iloc[-1]


def vendor_symbol(master: pd.DataFrame, instrument_id: str, vendor: str) -> str | None:
    """The vendor's symbol for an instrument (forward direction).

    Returns ``None`` when the instrument is unknown or the vendor does not cover it.
    """
    rows = master[master["instrument_id"] == instrument_id]
    if rows.empty:
        return None
    row = _select_window(rows)
    mapping = json.loads(row["vendor_symbols"])
    return mapping.get(vendor)


def resolve_vendor_symbol(master: pd.DataFrame, vendor: str, symbol: str,
                          as_of) -> str | None:
    """Reverse direction with PIT window match: which instrument_id did `vendor`'s
    `symbol` refer to on `as_of`?

    This is where ticker reuse is defeated. If two instruments ever shared a vendor
    symbol across disjoint validity windows, the date decides which one wins. Returns
    the matching ``instrument_id`` (or ``None``).
    """
    as_of = pd.Timestamp(as_of)
    matches: list[str] = []
    for row in master.itertuples(index=False):
        mapping = json.loads(row.vendor_symbols)
        if mapping.get(vendor) != symbol:
            continue
        vf, vt = row.valid_from, row.valid_to
        if (pd.isna(vf) or vf <= as_of) and (pd.isna(vt) or as_of <= vt):
            matches.append(row.instrument_id)
    return matches[0] if matches else None
