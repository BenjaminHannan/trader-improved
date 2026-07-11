"""BinanceVisionLoader tests — NO NETWORK.

Canned listing XML and in-memory zips mirror the REAL Binance Vision archive
(live-probed 2026-07-11, see production/data/loaders/binance_vision.py's module
docstring): the S3 listing interleaves each month's ``.zip`` key with a sibling
``.zip.CHECKSUM`` key, monthly klines CSVs carry NO header row (12 comma-separated
fields, first field = open_time), and a recent file's open_time is a 16-digit
MICROSECOND epoch where older files carry a 13-digit MILLISECOND epoch (same probe
session, BTCUSDT 2020-08 vs 2026-06) — a finding this test suite pins directly.

Covers: klines parse + day alignment (values lifted from the real probed SOL
2020-08 file), header-row tolerance, the millisecond/microsecond timestamp
disambiguation, the BACKFILL-ONLY contract (cutoff, dead-asset full history,
ratchet-immunity, refusal with no lake coverage), sub-$0.10 survival, plain-symbol
resolution, per-asset fetch degradation on a 404 and on an empty listing, listing
pagination via continuation-token, and the end-to-end obs+35d curated write.
"""
from __future__ import annotations

import io
import json
import sys
import types
import zipfile

import pandas as pd
import pytest

from production.data.audit import audit
from production.data.base import IngestError
from production.data.loaders.binance_vision import (
    LISTING_URL, ZIP_BASE_URL, BinanceVisionLoader,
)

UTC = "UTC"

_HEADER = ("open_time,open,high,low,close,volume,close_time,quote_asset_volume,"
           "count,taker_buy_volume,taker_buy_quote_volume,ignore")


# --------------------------------------------------------------------- helpers
def _bv_instruments() -> pd.DataFrame:
    """Master with only a 'ccxt' vendor key — no 'binance' key, proving binance
    symbol resolution falls back to the plain `symbol` match (production/data/base.py),
    exactly like CoinMetricsRatesLoader's test fixture."""
    def row(iid, sym):
        return dict(
            instrument_id=iid, asset_class="crypto", sleeve="crypto", symbol=sym,
            vendor_symbols=json.dumps({"ccxt": f"{sym}/USD"}),
            currency="USD", valid_from="1990-01-01", valid_to="2099-01-01",
            proxy_of=None, sector=None, meta="{}",
        )
    return pd.DataFrame([
        row("CR:SOL:2020-08-11", "SOL"),
        row("CR:BTC:2017-08-17", "BTC"),
        row("CR:SHIB:2021-01-01", "SHIB"),
        row("CR:BCC:2017-11-01", "BCC"),   # delisted pair, no live ccxt coverage
    ])


def _seed_ccxt_coverage(lake, iid: str, start: str, days: int = 3) -> None:
    """Write a few ccxt-sourced prices/crypto rows so the loader sees coverage."""
    dates = pd.date_range(start, periods=days, freq="D")
    df = pd.DataFrame({
        "obs_date": dates,
        "instrument_id": iid,
        "close": 100.0,
        "volume": 1.0,
        "dollar_volume": 100.0,
        "available_from": dates.tz_localize(UTC) + pd.Timedelta(hours=24),
        "source": "ccxt:prices",
        "ingested_at": pd.Timestamp("2026-01-01", tz=UTC),
    })
    lake.write_curated(df, "prices", "crypto")


def _day_bounds_ms(date_str: str) -> tuple[int, int]:
    d = pd.Timestamp(date_str, tz="UTC")
    open_ms = int(d.value // 1_000_000)
    close_ms = open_ms + 86_400_000 - 1
    return open_ms, close_ms


def _kline_row(date_str: str, close: float, *, open_=None, high=None, low=None,
              volume: float = 10.0, quote_vol: float = 1000.0, us: bool = False) -> str:
    """One no-header klines CSV line for UTC day `date_str` (real 12-field schema)."""
    open_ms, close_ms = _day_bounds_ms(date_str)
    if us:      # simulate the microsecond-epoch files (module docstring finding)
        open_ms, close_ms = open_ms * 1000, close_ms * 1000 + 999
    o = close if open_ is None else open_
    h = close if high is None else high
    l = close if low is None else low
    return (f"{open_ms},{o},{h},{l},{close},{volume},{close_ms},{quote_vol},"
            f"100,{volume / 2},{quote_vol / 2},0")


def _csv(rows: list[str], header: bool = False) -> str:
    lines = [_HEADER] if header else []
    lines.extend(rows)
    return "\n".join(lines)


def _zip_bytes(csv_text: str, member_name: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(member_name, csv_text)
    return buf.getvalue()


def _listing_xml(pair: str, months: list[str], truncated: bool = False,
                 next_token: str | None = None) -> str:
    """Mirrors the real ListBucketResult shape: each month yields a `.zip` Key
    plus a sibling `.zip.CHECKSUM` Key (live-probed 2026-07-11)."""
    prefix = f"data/spot/monthly/klines/{pair}/1d/"
    contents = "".join(
        f"<Contents><Key>{prefix}{pair}-1d-{m}.zip</Key></Contents>"
        f"<Contents><Key>{prefix}{pair}-1d-{m}.zip.CHECKSUM</Key></Contents>"
        for m in months
    )
    token_xml = (f"<NextContinuationToken>{next_token}</NextContinuationToken>"
                 if next_token else "")
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
        f"<Name>data.binance.vision</Name><Prefix>{prefix}</Prefix>"
        f'<IsTruncated>{"true" if truncated else "false"}</IsTruncated>{token_xml}'
        f"{contents}</ListBucketResult>"
    )


# =============================================================== fetch fakes
class _Resp:
    def __init__(self, content: bytes = b"", text: str | None = None, status: int = 200):
        self.content = content
        self.text = text if text is not None else content.decode("utf-8", "replace")
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class _FakeSession:
    def __init__(self, get_fn):
        self._get_fn = get_fn

    def get(self, url, params=None, timeout=None):
        return self._get_fn(url, params, timeout)


def _install_fake_requests(monkeypatch, get_fn):
    mod = types.ModuleType("requests")
    mod.Session = lambda: _FakeSession(get_fn)
    monkeypatch.setitem(sys.modules, "requests", mod)
    monkeypatch.setattr("time.sleep", lambda s: None)


# ============================================== klines parsing + day alignment
def test_transform_parses_klines_close_volume_and_day_alignment(tmp_lake):
    """Values lifted from the real SOLUSDT-1d-2020-08.zip probe (2026-07-11)."""
    _seed_ccxt_coverage(tmp_lake, "CR:SOL:2020-08-11", "2024-01-01")
    raw = {"SOL": [_csv([
        _kline_row("2020-08-11", 3.29850, volume=1552384.78, quote_vol=4939148.938107),
        _kline_row("2020-08-12", 3.75580, volume=1737042.95, quote_vol=6176153.724638),
    ])]}
    loader = BinanceVisionLoader(tmp_lake, instruments=_bv_instruments())
    long = loader.transform(raw).set_index("obs_date")

    assert long.loc[pd.Timestamp("2020-08-11"), "close"] == pytest.approx(3.29850)
    assert long.loc[pd.Timestamp("2020-08-11"), "volume"] == pytest.approx(1552384.78)
    assert long.loc[pd.Timestamp("2020-08-11"), "dollar_volume"] == pytest.approx(4939148.938107)
    assert long.loc[pd.Timestamp("2020-08-12"), "close"] == pytest.approx(3.75580)
    assert (long["asset_class"] == "crypto").all()


def test_header_row_is_dropped_when_present(tmp_lake):
    """Recent files MAY carry a header row (build brief); tolerate it defensively
    even though every file probed live (2020-08 and 2026-06) had none."""
    _seed_ccxt_coverage(tmp_lake, "CR:SOL:2020-08-11", "2024-01-01")
    raw = {"SOL": [_csv([_kline_row("2020-08-11", 3.2985)], header=True)]}
    loader = BinanceVisionLoader(tmp_lake, instruments=_bv_instruments())
    long = loader.transform(raw)
    assert len(long) == 1
    assert long.iloc[0]["close"] == pytest.approx(3.2985)
    assert long.iloc[0]["obs_date"] == pd.Timestamp("2020-08-11")


def test_microsecond_epoch_open_time_is_detected_and_normalized(tmp_lake):
    """A current-era file (live-probed BTCUSDT 2026-06) carries a 16-digit
    MICROSECOND open_time where the historical files carry 13-digit milliseconds —
    the loader must disambiguate by magnitude, not assume millisecond-only."""
    _seed_ccxt_coverage(tmp_lake, "CR:BTC:2017-08-17", "2027-01-01")
    raw = {"BTC": [_csv([_kline_row("2026-06-01", 71408.90, us=True)])]}
    loader = BinanceVisionLoader(tmp_lake, instruments=_bv_instruments())
    long = loader.transform(raw)
    assert len(long) == 1
    assert long.iloc[0]["obs_date"] == pd.Timestamp("2026-06-01")
    assert long.iloc[0]["close"] == pytest.approx(71408.90)


# ============================================== the backfill-only cutoff contract
def test_rows_stop_strictly_before_existing_ccxt_coverage(tmp_lake):
    _seed_ccxt_coverage(tmp_lake, "CR:SOL:2020-08-11", "2021-01-01")
    raw = {"SOL": [_csv([
        _kline_row("2020-12-29", 27000),
        _kline_row("2020-12-30", 27300),   # cutoff - 1d boundary: 2020-12-31 excluded
        _kline_row("2020-12-31", 28900),
        _kline_row("2021-01-01", 29200),   # covered by ccxt -> never emitted
        _kline_row("2021-06-01", 36000),
    ])]}
    loader = BinanceVisionLoader(tmp_lake, instruments=_bv_instruments())
    long = loader.transform(raw)
    assert long["obs_date"].max() == pd.Timestamp("2020-12-30")
    assert len(long) == 2


def test_sub_ten_cent_crypto_prices_survive(tmp_lake):
    """No equity-style sub-$0.10 hygiene floor here, matching ccxt_prices and
    coinmetrics_rates (module docstring rationale)."""
    _seed_ccxt_coverage(tmp_lake, "CR:SHIB:2021-01-01", "2022-01-01")
    raw = {"SHIB": [_csv([
        _kline_row("2021-01-05", 0.0000059),
        _kline_row("2021-01-06", 0.0000061),
    ])]}
    loader = BinanceVisionLoader(tmp_lake, instruments=_bv_instruments())
    long = loader.transform(raw)
    assert len(long) == 2
    assert long["close"].min() == pytest.approx(0.0000059)


def test_dead_asset_without_coverage_keeps_full_history(tmp_lake):
    # coverage exists for SOL only; BCC (a delisted pair) has none -> full keep
    _seed_ccxt_coverage(tmp_lake, "CR:SOL:2020-08-11", "2021-01-01")
    raw = {"BCC": [_csv([
        _kline_row("2017-11-01", 1500.0),
        _kline_row("2018-11-01", 500.0),
    ])]}
    loader = BinanceVisionLoader(tmp_lake, instruments=_bv_instruments())
    long = loader.transform(raw)
    assert len(long) == 2
    assert long["obs_date"].max() == pd.Timestamp("2018-11-01")


def test_cutoff_ignores_prior_binance_vision_rows_no_ratchet(tmp_lake):
    """Re-run idempotence: the cutoff comes from ccxt-sourced rows ONLY. A prior
    binance:vision backfill's own earlier rows must not ratchet the floor
    backward (the exact bug documented in coinmetrics_rates.py)."""
    _seed_ccxt_coverage(tmp_lake, "CR:SOL:2020-08-11", "2021-01-01")
    old = pd.DataFrame({
        "obs_date": [pd.Timestamp("2020-08-11")],
        "instrument_id": ["CR:SOL:2020-08-11"],
        "close": [3.2985], "volume": [100.0], "dollar_volume": [330.0],
        "available_from": [pd.Timestamp("2020-09-15", tz=UTC)],
        "source": ["binance:vision"],
        "ingested_at": [pd.Timestamp("2026-01-01", tz=UTC)],
    })
    tmp_lake.write_curated(old, "prices", "crypto")
    raw = {"SOL": [_csv([_kline_row("2020-08-11", 3.2985), _kline_row("2020-12-01", 20.0)])]}
    loader = BinanceVisionLoader(tmp_lake, instruments=_bv_instruments())
    long = loader.transform(raw)
    # both rows precede the CCXT floor (2021-01-01): both emitted, no ratchet
    assert len(long) == 2
    assert long["obs_date"].max() == pd.Timestamp("2020-12-01")


def test_refuses_to_emit_when_lake_has_no_crypto_coverage(tmp_lake):
    raw = {"SOL": [_csv([_kline_row("2020-08-11", 3.2985)])]}
    loader = BinanceVisionLoader(tmp_lake, instruments=_bv_instruments())
    long = loader.transform(raw)
    assert long.empty
    assert any("run the ccxt prices ingest first" in w for w in loader.warnings)


# ======================================================== symbol resolution
def test_unresolvable_symbol_is_dropped_silently(tmp_lake):
    _seed_ccxt_coverage(tmp_lake, "CR:SOL:2020-08-11", "2021-01-01")
    raw = {
        "SOL": [_csv([_kline_row("2020-08-11", 3.2985)])],
        "NOTAREALCOIN": [_csv([_kline_row("2020-08-11", 1.0)])],
    }
    loader = BinanceVisionLoader(tmp_lake, instruments=_bv_instruments())
    long = loader.transform(raw)
    assert set(long["instrument_id"]) == {"CR:SOL:2020-08-11"}


# =============================================================== fetch behavior
def test_fetch_requires_assets_configured(tmp_lake):
    loader = BinanceVisionLoader(tmp_lake, instruments=_bv_instruments(), assets=[])
    with pytest.raises(IngestError):
        loader.fetch("2015-01-01", "2022-01-01")


def test_fetch_lists_and_downloads_monthly_zips(tmp_lake, monkeypatch):
    def get_fn(url, params, timeout):
        if url == LISTING_URL:
            return _Resp(text=_listing_xml("SOLUSDT", ["2020-08", "2020-09"]))
        assert url.startswith(ZIP_BASE_URL)
        month = url.rsplit("-1d-", 1)[1][: len("2020-08")]
        csv_text = _csv([_kline_row(f"{month}-01", 3.0)])
        return _Resp(content=_zip_bytes(csv_text, f"SOLUSDT-1d-{month}.csv"))

    _install_fake_requests(monkeypatch, get_fn)
    loader = BinanceVisionLoader(tmp_lake, instruments=_bv_instruments(),
                                 assets=["sol"], pause_s=0)  # lowercase in -> uppercased
    raw = loader.fetch("2015-01-01", "2022-01-01")
    assert set(raw) == {"SOL"}
    assert len(raw["SOL"]) == 2


def test_fetch_degrades_on_zip_404_and_keeps_the_rest(tmp_lake, monkeypatch):
    """Per-asset degradation: one pair's zip 404s, the other asset's fetch
    still succeeds and a warning names the failed pair."""
    def get_fn(url, params, timeout):
        if url == LISTING_URL:
            pair = params["prefix"].split("/")[4]
            return _Resp(text=_listing_xml(pair, ["2020-08"]))
        if "DOESNOTEXIST" in url:
            return _Resp(status=404)
        return _Resp(content=_zip_bytes(_csv([_kline_row("2020-08-01", 3.0)]), "x.csv"))

    _install_fake_requests(monkeypatch, get_fn)
    loader = BinanceVisionLoader(tmp_lake, instruments=_bv_instruments(),
                                 assets=["SOL", "DOESNOTEXIST"], pause_s=0)
    raw = loader.fetch("2015-01-01", "2022-01-01")
    assert set(raw) == {"SOL"}
    assert any("DOESNOTEXIST" in w for w in loader.warnings)


def test_fetch_warns_on_pair_with_no_archived_months(tmp_lake, monkeypatch):
    """An empty listing (KeyCount=0, HTTP 200 — live-probed behavior for a pair
    that was never listed against the quote) warns rather than erroring."""
    def get_fn(url, params, timeout):
        assert url == LISTING_URL
        pair = params["prefix"].split("/")[4]
        return _Resp(text=_listing_xml(pair, []))

    _install_fake_requests(monkeypatch, get_fn)
    loader = BinanceVisionLoader(tmp_lake, instruments=_bv_instruments(),
                                 assets=["NOPE"], pause_s=0)
    raw = loader.fetch("2015-01-01", "2022-01-01")
    assert raw == {}
    assert any("no monthly 1d history" in w for w in loader.warnings)


def test_fetch_paginates_listing_via_continuation_token(tmp_lake, monkeypatch):
    calls = []

    def get_fn(url, params, timeout):
        if url == LISTING_URL:
            calls.append(dict(params))
            if "continuation-token" not in params:
                return _Resp(text=_listing_xml("SOLUSDT", ["2020-08"], truncated=True,
                                               next_token="PAGE2"))
            assert params["continuation-token"] == "PAGE2"
            return _Resp(text=_listing_xml("SOLUSDT", ["2020-09"]))
        return _Resp(content=_zip_bytes(_csv([_kline_row("2020-08-01", 3.0)]), "x.csv"))

    _install_fake_requests(monkeypatch, get_fn)
    loader = BinanceVisionLoader(tmp_lake, instruments=_bv_instruments(),
                                 assets=["SOL"], pause_s=0)
    raw = loader.fetch("2015-01-01", "2022-01-01")
    assert len(calls) == 2                # both listing pages were fetched
    assert len(raw["SOL"]) == 2           # both months' zips downloaded


# ================================================================ end-to-end
def test_end_to_end_curated_write_with_35d_stamp(tmp_lake, monkeypatch):
    _seed_ccxt_coverage(tmp_lake, "CR:SOL:2020-08-11", "2021-01-01")

    def fake_fetch(start, end):
        return {"SOL": [_csv([
            _kline_row("2020-08-11", 3.2985),
            _kline_row("2020-08-12", 3.7558),
        ])]}

    loader = BinanceVisionLoader(tmp_lake, instruments=_bv_instruments())
    monkeypatch.setattr(loader, "fetch", fake_fetch)
    res = loader.run("2014-01-01", "2022-01-01", incremental=False)

    assert res.audit["fatal"] is False
    cur = tmp_lake.read_curated("prices", "crypto")
    bv = cur[cur["source"] == "binance:vision"]
    assert len(bv) == 2
    # available_from = obs_date + 35d (module docstring: worst-case monthly-zip
    # publication lag)
    avail = pd.to_datetime(bv["available_from"], utc=True)
    obs = pd.to_datetime(bv["obs_date"]).dt.tz_localize(UTC)
    assert (avail == obs + pd.Timedelta(days=35)).all()
    # the ccxt seed rows are untouched alongside
    assert (cur["source"] == "ccxt:prices").sum() == 3

    rep = audit(bv, BinanceVisionLoader.expectations)
    assert rep.fatal is False
