"""CoinMetricsRatesLoader tests — NO NETWORK.

Canned payloads mirror the REAL community GitHub CSV (live-probed 2026-07-10 at
https://raw.githubusercontent.com/coinmetrics/data/master/csv/btc.csv): one CSV
per lowercase asset id, a ``time`` column plus many metric columns of which
``PriceUSD`` is the day-close series aligned with our ccxt convention. The same
probe session established (a) the community REST API now serves only a ~7-day
window and silently returns empty when date-filtered — hence CSVs, not the API —
and (b) ``ReferenceRate(D+1) == PriceUSD(D)`` in-file, pinned below as the
day-alignment test.

Covers: PriceUSD -> close mapping and day alignment, the BACKFILL-ONLY contract
(rows stop strictly before each instrument's existing ccxt coverage; full history
for instruments with no coverage; refusal to emit anything when the lake has no
crypto prices at all), plain-symbol resolution fallback, per-asset fetch
degradation, the obs+48h availability stamp, and the end-to-end curated write.
"""
from __future__ import annotations

import json
import sys
import types

import pandas as pd
import pytest

from production.data.audit import audit
from production.data.base import IngestError
from production.data.loaders.coinmetrics_rates import CoinMetricsRatesLoader

UTC = "UTC"


# --------------------------------------------------------------------- helpers
def _crypto_instruments() -> pd.DataFrame:
    """Master with only a 'ccxt' vendor key — no 'coinmetrics' key, proving
    resolution falls back to the plain `symbol` match (production/data/base.py)."""
    def row(iid, sym):
        return dict(
            instrument_id=iid, asset_class="crypto", sleeve="crypto", symbol=sym,
            vendor_symbols=json.dumps({"ccxt": f"{sym}/USD"}),
            currency="USD", valid_from="1990-01-01", valid_to="2099-01-01",
            proxy_of=None, sector=None, meta="{}",
        )
    return pd.DataFrame([
        row("CR:BTC:2013-09-01", "BTC"),
        row("CR:ETH:2015-08-07", "ETH"),
    ])


def _csv(rows: list[tuple[str, float | str, float | str]]) -> str:
    """Minimal community-CSV shape: time, ReferenceRateUSD, PriceUSD (+ noise col
    to prove column selection)."""
    out = ["time,AdrActCnt,ReferenceRateUSD,PriceUSD"]
    for t, rr, p in rows:
        out.append(f"{t},123,{rr},{p}")
    return "\n".join(out)


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


# ============================================== PriceUSD mapping + day alignment
def test_transform_uses_priceusd_with_day_close_alignment(tmp_lake):
    """ReferenceRate(D+1) == PriceUSD(D) in the real file; the loader must take
    PriceUSD at time=D as day-D close (our ccxt convention), NOT ReferenceRate."""
    _seed_ccxt_coverage(tmp_lake, "CR:BTC:2013-09-01", "2021-01-01")
    raw = {"btc": _csv([
        ("2015-01-01", "", "314.25"),
        ("2015-01-02", "314.25", "315.10"),   # RR lags PriceUSD by one day
        ("2015-01-03", "315.10", "281.awkward"),  # unparseable price -> dropped
    ])}
    loader = CoinMetricsRatesLoader(tmp_lake, instruments=_crypto_instruments())
    long = loader.transform(raw)

    got = long.set_index("obs_date")["close"]
    assert got[pd.Timestamp("2015-01-01")] == pytest.approx(314.25)
    assert got[pd.Timestamp("2015-01-02")] == pytest.approx(315.10)
    assert pd.Timestamp("2015-01-03") not in got.index
    assert long["volume"].isna().all() and long["dollar_volume"].isna().all()
    assert (long["asset_class"] == "crypto").all()


# ============================================== the backfill-only cutoff contract
def test_rows_stop_strictly_before_existing_ccxt_coverage(tmp_lake):
    _seed_ccxt_coverage(tmp_lake, "CR:BTC:2013-09-01", "2021-01-01")
    raw = {"btc": _csv([
        ("2020-12-29", "", "27000"),
        ("2020-12-30", "", "27300"),   # cutoff - 1d boundary: 2020-12-31 excluded
        ("2020-12-31", "", "28900"),
        ("2021-01-01", "", "29200"),   # covered by ccxt -> never emitted
        ("2021-06-01", "", "36000"),
    ])}
    loader = CoinMetricsRatesLoader(tmp_lake, instruments=_crypto_instruments())
    long = loader.transform(raw)
    assert long["obs_date"].max() == pd.Timestamp("2020-12-30")
    assert len(long) == 2


def test_sub_ten_cent_crypto_prices_survive(tmp_lake):
    """The equity sub-$0.10 hygiene floor must NOT apply here: ccxt's crypto path
    doesn't apply it either (SHIB at $0.000004 lives in the real lake), and the
    first backfill wrongly ate 5,287 legitimate rows (early BTC, pre-2021 DOGE)."""
    _seed_ccxt_coverage(tmp_lake, "CR:BTC:2013-09-01", "2021-01-01")
    raw = {"btc": _csv([("2010-08-01", "", "0.06"), ("2010-08-02", "", "0.07")])}
    loader = CoinMetricsRatesLoader(tmp_lake, instruments=_crypto_instruments())
    long = loader.transform(raw)
    assert len(long) == 2
    assert long["close"].min() == pytest.approx(0.06)


def test_dead_asset_without_coverage_keeps_full_history(tmp_lake):
    # coverage exists for BTC only; ETH (think: a dead asset) has none -> full keep
    _seed_ccxt_coverage(tmp_lake, "CR:BTC:2013-09-01", "2021-01-01")
    raw = {"eth": _csv([("2016-03-01", "", "12.5"), ("2022-03-01", "", "2900")])}
    loader = CoinMetricsRatesLoader(tmp_lake, instruments=_crypto_instruments())
    long = loader.transform(raw)
    assert len(long) == 2
    assert long["obs_date"].max() == pd.Timestamp("2022-03-01")


def test_cutoff_ignores_prior_cm_rows_no_ratchet(tmp_lake):
    """Re-run idempotence: the cutoff comes from ccxt-sourced rows ONLY. A prior
    CM backfill's own earlier rows must not ratchet the floor backward (the bug
    that shrank the second --full run from 8,266 to 4,513 rows)."""
    _seed_ccxt_coverage(tmp_lake, "CR:BTC:2013-09-01", "2021-01-01")
    # simulate a prior CM backfill row well before ccxt coverage
    old = pd.DataFrame({
        "obs_date": [pd.Timestamp("2015-01-01")],
        "instrument_id": ["CR:BTC:2013-09-01"],
        "close": [314.25], "volume": [float("nan")], "dollar_volume": [float("nan")],
        "available_from": [pd.Timestamp("2015-01-03", tz=UTC)],
        "source": ["coinmetrics:community-csv"],
        "ingested_at": [pd.Timestamp("2026-01-01", tz=UTC)],
    })
    tmp_lake.write_curated(old, "prices", "crypto")
    raw = {"btc": _csv([("2015-01-01", "", "314.25"), ("2020-12-01", "", "19700")])}
    loader = CoinMetricsRatesLoader(tmp_lake, instruments=_crypto_instruments())
    long = loader.transform(raw)
    # both rows precede the CCXT floor (2021-01-01): both emitted, no ratchet
    assert len(long) == 2
    assert long["obs_date"].max() == pd.Timestamp("2020-12-01")


def test_reference_rate_fallback_shifts_one_day_back_and_coalesces(tmp_lake):
    """Newer assets' CSVs carry no PriceUSD (day-START-stamped rates only): the
    fallback must map time=D to obs_date D-1 (in-file identity
    ReferenceRate(D+1) == PriceUSD(D)) and coalesce ReferenceRateUSD with the
    plain ReferenceRate column, which carries the FULL history where the
    USD-suffixed twin is only populated for recent days (sol, probed 2026-07-11)."""
    _seed_ccxt_coverage(tmp_lake, "CR:BTC:2013-09-01", "2021-01-01")
    csv_no_price = ("time,AdrActCnt,ReferenceRate,ReferenceRateUSD\n"
                    "2015-06-02,9,251.5,\n"          # only plain RR populated
                    "2015-06-03,9,260.0,260.0\n")    # both populated (equal)
    loader = CoinMetricsRatesLoader(tmp_lake, instruments=_crypto_instruments())
    long = loader.transform({"eth": csv_no_price})
    got = long.set_index("obs_date")["close"]
    assert got[pd.Timestamp("2015-06-01")] == pytest.approx(251.5)  # coalesced
    assert got[pd.Timestamp("2015-06-02")] == pytest.approx(260.0)  # shifted back
    assert len(long) == 2


def test_refuses_to_emit_when_lake_has_no_crypto_coverage(tmp_lake):
    """Empty lake -> emitting could later displace the tradable-venue series under
    the latest-vintage tie-break; the loader must emit NOTHING and warn."""
    raw = {"btc": _csv([("2015-01-01", "", "314.25")])}
    loader = CoinMetricsRatesLoader(tmp_lake, instruments=_crypto_instruments())
    long = loader.transform(raw)
    assert long.empty
    assert any("run the ccxt prices ingest first" in w for w in loader.warnings)


# ======================================================== symbol resolution
def test_resolves_lowercase_asset_id_via_plain_symbol_fallback(tmp_lake):
    _seed_ccxt_coverage(tmp_lake, "CR:BTC:2013-09-01", "2021-01-01")
    raw = {
        "eth": _csv([("2016-03-01", "", "12.5")]),
        # not on the master at all -> silently dropped, no crash
        "dogecoin": _csv([("2016-03-01", "", "0.05")]),
    }
    loader = CoinMetricsRatesLoader(tmp_lake, instruments=_crypto_instruments())
    long = loader.transform(raw)
    assert set(long["instrument_id"]) == {"CR:ETH:2015-08-07"}


# =============================================================== fetch behavior
class _Resp:
    def __init__(self, text="", status=200):
        self.text = text
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def _install_fake_requests(monkeypatch, fake_get):
    mod = types.ModuleType("requests")
    mod.get = fake_get
    monkeypatch.setitem(sys.modules, "requests", mod)
    monkeypatch.setattr("time.sleep", lambda s: None)


def test_fetch_degrades_on_missing_csv_and_keeps_the_rest(tmp_lake, monkeypatch):
    def fake_get(url, timeout=None):
        if "doesnotexist123" in url:
            return _Resp(status=404)
        return _Resp(_csv([("2016-03-01", "", "12.5")]))

    _install_fake_requests(monkeypatch, fake_get)
    loader = CoinMetricsRatesLoader(tmp_lake, instruments=_crypto_instruments(),
                                    assets=["btc", "doesnotexist123", "eth"],
                                    pause_s=0)
    raw = loader.fetch("2015-01-01", "2022-01-01")
    assert set(raw) == {"btc", "eth"}
    assert any("doesnotexist123" in w for w in loader.warnings)


def test_fetch_requires_assets_configured(tmp_lake):
    loader = CoinMetricsRatesLoader(tmp_lake, instruments=_crypto_instruments(),
                                    assets=[])
    with pytest.raises(IngestError):
        loader.fetch("2015-01-01", "2022-01-01")


# ================================================================ end-to-end
def test_end_to_end_curated_write_with_48h_stamp(tmp_lake, monkeypatch):
    _seed_ccxt_coverage(tmp_lake, "CR:BTC:2013-09-01", "2021-01-01")

    def fake_fetch(start, end):
        return {"btc": _csv([
            ("2015-01-01", "", "314.25"),
            ("2015-01-02", "314.25", "315.10"),
        ])}

    loader = CoinMetricsRatesLoader(tmp_lake, instruments=_crypto_instruments())
    monkeypatch.setattr(loader, "fetch", fake_fetch)
    res = loader.run("2014-01-01", "2022-01-01", incremental=False)

    assert res.audit["fatal"] is False
    cur = tmp_lake.read_curated("prices", "crypto")
    cm = cur[cur["source"] == "coinmetrics:community-csv"]
    assert len(cm) == 2
    # available_from = obs_date + 48h (exists at +24h, repo publishes later)
    avail = pd.to_datetime(cm["available_from"], utc=True)
    obs = pd.to_datetime(cm["obs_date"]).dt.tz_localize(UTC)
    assert (avail == obs + pd.Timedelta(hours=48)).all()
    # the ccxt seed rows are untouched alongside
    assert (cur["source"] == "ccxt:prices").sum() == 3

    rep = audit(cm, CoinMetricsRatesLoader.expectations)
    assert rep.fatal is False
