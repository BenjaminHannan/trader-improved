"""Execution-layer tests — all offline (requests fully monkeypatched, no keys, no network).

Covers: order construction (per-sleeve rounding, min-notional drop, side signs), instrument
-> Alpaca symbol mapping from the master JSON, the dry-run safety property (submission issues
zero HTTP requests), retry-on-429-then-success, 4xx -> ExecutionError, implementation-shortfall
arithmetic, and a --dry-run smoke run of scripts/daily_run.py on the synthetic bundle.
"""
from __future__ import annotations

import pandas as pd
import pytest

import pandas.testing as pdt

from production.execution import alpaca_paper as ap
from production.execution.alpaca_paper import AlpacaPaperClient, ExecutionError
from production.core.lake import Lake
from production.execution.orders import ORDER_COLUMNS, target_weights_to_orders
from production.execution.shortfall import implementation_shortfall
from production.execution.tca import calibrate_overrides, read_overrides, write_overrides
from production.reference.instruments import build_instrument_master


# --------------------------------------------------------------------- helpers
class _FakeResponse:
    def __init__(self, status_code, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload
        self.text = text
        self.content = b"x" if payload is not None else b""

    def json(self):
        return self._payload


def _no_http(*a, **k):  # pragma: no cover - only fires if a dry-run leaks a request
    raise AssertionError("HTTP request issued during a dry-run / offline test")


# ----------------------------------------------------------------- order sizing
def test_orders_equity_rounds_to_whole_shares():
    # equity name @ $100, target 10% of $100k = $10,000 -> 100 shares exactly.
    orders = target_weights_to_orders(
        pd.Series({"EQ:AAA:2000-01-03": 0.10}),
        pd.Series(dtype=float),
        equity_usd=100_000.0,
        prices=pd.Series({"EQ:AAA:2000-01-03": 100.0}),
    )
    assert len(orders) == 1
    row = orders.iloc[0]
    assert row["side"] == "buy"
    assert row["qty"] == 100.0            # whole shares
    assert row["order_type"] == "market"
    assert row["notional_usd"] == pytest.approx(10_000.0)


def test_orders_equity_fractional_share_is_rounded_off():
    # $10,050 / $100 = 100.5 shares -> rounds to whole shares (100 or 101, never fractional).
    orders = target_weights_to_orders(
        pd.Series({"EQ:AAA:2000-01-03": 0.1005}),
        pd.Series(dtype=float),
        equity_usd=100_000.0,
        prices=pd.Series({"EQ:AAA:2000-01-03": 100.0}),
    )
    qty = orders.iloc[0]["qty"]
    assert qty == float(int(qty))         # integral


def test_orders_crypto_rounds_to_six_dp():
    # crypto keeps fractional precision to 6 dp.
    orders = target_weights_to_orders(
        pd.Series({"CR:BTC:2017-01-01": 0.05}),
        pd.Series(dtype=float),
        equity_usd=100_000.0,
        prices=pd.Series({"CR:BTC:2017-01-01": 30_000.0}),
    )
    qty = orders.iloc[0]["qty"]
    # 5000 / 30000 = 0.16666... -> 0.166667
    assert qty == pytest.approx(0.166667, abs=1e-9)


def test_orders_min_notional_drop():
    # $10 delta is below the $25 floor -> dropped entirely.
    orders = target_weights_to_orders(
        pd.Series({"CR:BTC:2017-01-01": 0.10}),
        pd.Series({"CR:BTC:2017-01-01": 0.10 - 10.0 / 100_000.0}),
        equity_usd=100_000.0,
        prices=pd.Series({"CR:BTC:2017-01-01": 100.0}),
        min_order_usd=25.0,
    )
    assert orders.empty


def test_orders_side_signs_buy_and_sell():
    # target below current -> sell; above -> buy.
    orders = target_weights_to_orders(
        pd.Series({"EQ:AAA:2000-01-03": 0.02, "EQ:BBB:2000-01-03": 0.10}),
        pd.Series({"EQ:AAA:2000-01-03": 0.10, "EQ:BBB:2000-01-03": 0.02}),
        equity_usd=100_000.0,
        prices=pd.Series({"EQ:AAA:2000-01-03": 100.0, "EQ:BBB:2000-01-03": 100.0}),
    )
    by_id = orders.set_index("instrument_id")["side"].to_dict()
    assert by_id["EQ:AAA:2000-01-03"] == "sell"
    assert by_id["EQ:BBB:2000-01-03"] == "buy"


# --------------------------------------------------------------- limit orders
def test_orders_limit_price_buy_below_sell_above_known_answer():
    # equity @ $100, 5bp offset: buy limit = 100*(1-0.0005) = 99.95; sell = 100.05.
    orders = target_weights_to_orders(
        pd.Series({"EQ:AAA:2000-01-03": 0.10, "EQ:BBB:2000-01-03": 0.02}),
        pd.Series({"EQ:AAA:2000-01-03": 0.02, "EQ:BBB:2000-01-03": 0.10}),
        equity_usd=100_000.0,
        prices=pd.Series({"EQ:AAA:2000-01-03": 100.0, "EQ:BBB:2000-01-03": 100.0}),
        order_type="limit",
        limit_offset_bps=5.0,
    )
    by_id = orders.set_index("instrument_id")
    assert by_id.loc["EQ:AAA:2000-01-03", "side"] == "buy"
    assert by_id.loc["EQ:AAA:2000-01-03", "limit_price"] == pytest.approx(99.95)
    assert by_id.loc["EQ:BBB:2000-01-03", "side"] == "sell"
    assert by_id.loc["EQ:BBB:2000-01-03", "limit_price"] == pytest.approx(100.05)
    assert set(orders["order_type"]) == {"limit"}


def test_orders_limit_price_equity_rounds_to_two_dp():
    # 123.456 * (1 - 0.0005) = 123.394272 -> 2dp -> 123.39.
    orders = target_weights_to_orders(
        pd.Series({"EQ:AAA:2000-01-03": 0.10}),
        pd.Series(dtype=float),
        equity_usd=100_000.0,
        prices=pd.Series({"EQ:AAA:2000-01-03": 123.456}),
        order_type="limit",
        limit_offset_bps=5.0,
    )
    lp = orders.iloc[0]["limit_price"]
    assert lp == pytest.approx(123.39)
    assert round(lp, 2) == lp        # no sub-cent precision


def test_orders_limit_price_crypto_rounds_to_six_sigfigs():
    # 43210.99 * (1 - 0.0005) = 43189.384505 -> 6 significant figures -> 43189.4.
    orders = target_weights_to_orders(
        pd.Series({"CR:BTC:2017-01-01": 0.10}),
        pd.Series(dtype=float),
        equity_usd=100_000.0,
        prices=pd.Series({"CR:BTC:2017-01-01": 43210.99}),
        order_type="limit",
        limit_offset_bps=5.0,
    )
    assert orders.iloc[0]["limit_price"] == pytest.approx(43189.4)


def test_orders_market_path_bit_identical_when_order_type_omitted():
    # Default (omitted) must equal an explicit market call, with NO limit_price column.
    args = (
        pd.Series({"EQ:AAA:2000-01-03": 0.10, "CR:BTC:2017-01-01": 0.05}),
        pd.Series(dtype=float),
    )
    kwargs = dict(equity_usd=100_000.0,
                  prices=pd.Series({"EQ:AAA:2000-01-03": 100.0,
                                    "CR:BTC:2017-01-01": 30_000.0}))
    default = target_weights_to_orders(*args, **kwargs)
    explicit = target_weights_to_orders(*args, order_type="market", **kwargs)
    assert list(default.columns) == ORDER_COLUMNS
    assert "limit_price" not in default.columns
    pdt.assert_frame_equal(default, explicit)


# --------------------------------------------------------- symbol mapping / submit
def test_symbol_mapping_from_master_json():
    master = build_instrument_master()
    client = AlpacaPaperClient(key_id="k", secret="s", backoff=0.0)
    # A crypto + an ETF from the static master.
    btc = next(i for i in master["instrument_id"] if i.startswith("CR:BTC:"))
    gld = next(i for i in master["instrument_id"] if i.startswith("CO:GLD:"))
    orders = pd.DataFrame({
        "instrument_id": [btc, gld],
        "side": ["buy", "buy"], "qty": [0.1, 3.0],
        "notional_usd": [100.0, 300.0], "order_type": ["market", "market"],
    })
    plan = client.submit_orders(orders, master, dry_run=True)
    m = plan.set_index("instrument_id")["alpaca_symbol"].to_dict()
    assert m[btc] == "BTCUSD"     # crypto BTCUSD-style
    assert m[gld] == "GLD"
    assert set(plan["status"]) == {"dry_run"}


def test_dry_run_submits_nothing(monkeypatch):
    monkeypatch.setattr(ap.requests, "request", _no_http)
    master = build_instrument_master()
    client = AlpacaPaperClient(key_id="k", secret="s", backoff=0.0)
    gld = next(i for i in master["instrument_id"] if i.startswith("CO:GLD:"))
    orders = pd.DataFrame({
        "instrument_id": [gld], "side": ["buy"], "qty": [3.0],
        "notional_usd": [300.0], "order_type": ["market"],
    })
    plan = client.submit_orders(orders, master, dry_run=True)   # must not raise
    assert list(plan["status"]) == ["dry_run"]
    assert plan.iloc[0]["broker_order_id"] is None


def test_submit_order_limit_payload_has_limit_price_and_tif_day(monkeypatch):
    captured = {}

    def fake_request(method, url, **kw):
        captured["json"] = kw.get("json")
        return _FakeResponse(200, payload={"id": "lim1", "status": "accepted"})

    monkeypatch.setattr(ap.requests, "request", fake_request)
    client = AlpacaPaperClient(key_id="k", secret="s", backoff=0.0)
    resp = client.submit_order("GLD", 3, "buy", type_="limit", limit_price=99.95)
    assert resp["id"] == "lim1"
    body = captured["json"]
    assert body["type"] == "limit"
    assert body["time_in_force"] == "day"
    assert body["limit_price"] == "99.95"


def test_submit_order_limit_missing_price_raises(monkeypatch):
    monkeypatch.setattr(ap.requests, "request", _no_http)   # must fail before any HTTP
    client = AlpacaPaperClient(key_id="k", secret="s", backoff=0.0)
    with pytest.raises(ExecutionError):
        client.submit_order("GLD", 3, "buy", type_="limit")


def test_submit_orders_threads_limit_price_from_frame(monkeypatch):
    captured = {}

    def fake_request(method, url, **kw):
        captured["json"] = kw.get("json")
        return _FakeResponse(200, payload={"id": "z", "status": "accepted"})

    monkeypatch.setattr(ap.requests, "request", fake_request)
    master = build_instrument_master()
    client = AlpacaPaperClient(key_id="k", secret="s", backoff=0.0)
    gld = next(i for i in master["instrument_id"] if i.startswith("CO:GLD:"))
    orders = pd.DataFrame({
        "instrument_id": [gld], "side": ["buy"], "qty": [3.0],
        "notional_usd": [300.0], "order_type": ["limit"], "limit_price": [123.39],
    })
    plan = client.submit_orders(orders, master, dry_run=False)
    assert captured["json"]["type"] == "limit"
    assert captured["json"]["limit_price"] == "123.39"
    assert plan.iloc[0]["limit_price"] == pytest.approx(123.39)


def test_retry_on_429_then_success(monkeypatch):
    calls = {"n": 0}

    def fake_request(method, url, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            return _FakeResponse(429, text="rate limited")
        return _FakeResponse(200, payload={"id": "abc", "status": "accepted"})

    monkeypatch.setattr(ap.requests, "request", fake_request)
    monkeypatch.setattr(ap.time, "sleep", lambda *_: None)   # no real backoff wait
    client = AlpacaPaperClient(key_id="k", secret="s", backoff=0.0)
    resp = client.submit_order("GLD", 3, "buy")
    assert calls["n"] == 2
    assert resp["id"] == "abc"


def test_4xx_raises_execution_error(monkeypatch):
    monkeypatch.setattr(ap.requests, "request",
                        lambda *a, **k: _FakeResponse(422, text="bad qty"))
    monkeypatch.setattr(ap.time, "sleep", lambda *_: None)
    client = AlpacaPaperClient(key_id="k", secret="s", backoff=0.0)
    with pytest.raises(ExecutionError):
        client.submit_order("GLD", -1, "buy")


def test_missing_credentials_raises_before_any_request(monkeypatch):
    monkeypatch.delenv("APCA_API_KEY_ID", raising=False)
    monkeypatch.delenv("APCA_API_SECRET_KEY", raising=False)
    monkeypatch.setattr(ap.requests, "request", _no_http)
    client = AlpacaPaperClient(backoff=0.0)
    with pytest.raises(ExecutionError):
        client.account()


def test_5xx_retries_exhausted_raises(monkeypatch):
    monkeypatch.setattr(ap.requests, "request",
                        lambda *a, **k: _FakeResponse(503, text="down"))
    monkeypatch.setattr(ap.time, "sleep", lambda *_: None)
    client = AlpacaPaperClient(key_id="k", secret="s", max_retries=3, backoff=0.0)
    with pytest.raises(ExecutionError):
        client.account()


def test_positions_empty_returns_empty_frame(monkeypatch):
    monkeypatch.setattr(ap.requests, "request",
                        lambda *a, **k: _FakeResponse(200, payload=[]))
    client = AlpacaPaperClient(key_id="k", secret="s", backoff=0.0)
    pos = client.positions()
    assert pos.empty and list(pos.columns) == ["symbol", "qty", "market_value", "side"]


# ------------------------------------------------------------------- shortfall
def test_shortfall_buy_above_decision_is_positive_cost():
    orders = pd.DataFrame({
        "instrument_id": ["EQ:AAA:2000-01-03"], "side": ["buy"],
        "notional_usd": [10_000.0],
    })
    df, agg = implementation_shortfall(
        orders,
        decision_prices=pd.Series({"EQ:AAA:2000-01-03": 100.0}),
        fill_prices=pd.Series({"EQ:AAA:2000-01-03": 101.0}),
    )
    # +1 * (101-100)/100 * 1e4 = +100 bps
    assert df.iloc[0]["shortfall_bps"] == pytest.approx(100.0)
    assert agg == pytest.approx(100.0)


def test_shortfall_sell_below_decision_is_positive_cost():
    orders = pd.DataFrame({
        "instrument_id": ["EQ:AAA:2000-01-03"], "side": ["sell"],
        "notional_usd": [5_000.0],
    })
    df, _ = implementation_shortfall(
        orders,
        decision_prices=pd.Series({"EQ:AAA:2000-01-03": 100.0}),
        fill_prices=pd.Series({"EQ:AAA:2000-01-03": 99.0}),
    )
    # -1 * (99-100)/100 * 1e4 = +100 bps
    assert df.iloc[0]["shortfall_bps"] == pytest.approx(100.0)


def test_shortfall_aggregate_is_notional_weighted():
    orders = pd.DataFrame({
        "instrument_id": ["EQ:AAA:2000-01-03", "EQ:BBB:2000-01-03"],
        "side": ["buy", "buy"],
        "notional_usd": [1_000.0, 3_000.0],
    })
    df, agg = implementation_shortfall(
        orders,
        decision_prices=pd.Series({"EQ:AAA:2000-01-03": 100.0,
                                   "EQ:BBB:2000-01-03": 100.0}),
        fill_prices=pd.Series({"EQ:AAA:2000-01-03": 101.0,      # +100 bps
                               "EQ:BBB:2000-01-03": 100.5}),     # +50 bps
    )
    # weighted: (100*1000 + 50*3000) / 4000 = 62.5 bps
    assert agg == pytest.approx(62.5)


# ------------------------------------------------------------ TCA calibration
def _shortfall_rows(instrument_id: str, abs_bps: float, n: int) -> pd.DataFrame:
    """n fills for one instrument with a fixed |shortfall|, alternating sign (median = abs_bps)."""
    signs = [1.0 if i % 2 == 0 else -1.0 for i in range(n)]
    return pd.DataFrame({
        "instrument_id": [instrument_id] * n,
        "side": ["buy" if s > 0 else "sell" for s in signs],
        "shortfall_bps": [s * abs_bps for s in signs],
        "notional_usd": [10_000.0] * n,
    })


def test_calibrate_overrides_known_answer():
    """20 fills at |shortfall| = 8 bps -> median 8 x safety 1.25 = 10 bps half-spread override."""
    df = _shortfall_rows("EQ:AAA:2000-01-03", abs_bps=8.0, n=20)
    ov = calibrate_overrides(df, min_fills=20, safety=1.25)
    assert set(ov) == {"EQ:AAA:2000-01-03"}
    entry = ov["EQ:AAA:2000-01-03"]
    assert entry["half_spread_bps"] == pytest.approx(10.0)
    assert entry["n_fills"] == 20
    assert entry["median_abs_shortfall_bps"] == pytest.approx(8.0)


def test_calibrate_overrides_never_lowers_cost():
    """Tiny realized shortfall (candidate below the sleeve default half-spread) -> no override.
    Floors are floors — calibration only ever raises cost, never lowers it (CLAUDE.md)."""
    # equity default half_spread = 2.5; 1.0 x 1.25 = 1.25 < 2.5 -> omitted.
    df = _shortfall_rows("EQ:BBB:2000-01-03", abs_bps=1.0, n=40)
    ov = calibrate_overrides(df, min_fills=20, safety=1.25)
    assert "EQ:BBB:2000-01-03" not in ov
    assert ov == {}


def test_calibrate_overrides_min_fills_gate():
    """An instrument below min_fills is not calibrated even with large realized shortfall."""
    df = _shortfall_rows("EQ:CCC:2000-01-03", abs_bps=25.0, n=5)
    assert calibrate_overrides(df, min_fills=20, safety=1.25) == {}
    # ... but clears once it has enough fills.
    df_ok = _shortfall_rows("EQ:CCC:2000-01-03", abs_bps=25.0, n=20)
    ov = calibrate_overrides(df_ok, min_fills=20, safety=1.25)
    assert ov["EQ:CCC:2000-01-03"]["half_spread_bps"] == pytest.approx(31.25)


def test_calibrate_overrides_caps_at_cap_bps():
    """A monster realized shortfall candidate is capped at the cost model cap (100 bps)."""
    df = _shortfall_rows("CR:BTC:2017-01-01", abs_bps=500.0, n=30)
    ov = calibrate_overrides(df, min_fills=20, safety=1.25)
    assert ov["CR:BTC:2017-01-01"]["half_spread_bps"] == pytest.approx(100.0)


def test_overrides_lake_roundtrip(tmp_path):
    """calibrate -> write -> read reproduces the override dict; calibrated_at is stamped."""
    lake = Lake(tmp_path)
    assert read_overrides(lake) == {}                         # absent table -> {}
    df = _shortfall_rows("EQ:AAA:2000-01-03", abs_bps=8.0, n=20)
    ov = calibrate_overrides(df, min_fills=20, safety=1.25)
    write_overrides(ov, lake)

    got = read_overrides(lake)
    assert got == ov

    # calibrated_at column present, non-null, a real timestamp.
    table = lake.read_reference("cost_overrides")
    assert "calibrated_at" in table.columns
    assert table["calibrated_at"].notna().all()
    assert pd.api.types.is_datetime64_any_dtype(pd.to_datetime(table["calibrated_at"]))


# ------------------------------------------------------------------ daily_run
def test_daily_run_dry_run_smoke(monkeypatch, capsys, pregate_engine_registry):
    # No network, no keys: dry-run must produce orders and never touch requests.
    # pregate_engine_registry: the synthetic warmup bundle cannot feed the live
    # registry's accepted-only factor set (fx carry + crypto basis) — pin all-candidate.
    monkeypatch.setattr(ap.requests, "request", _no_http)
    from scripts.daily_run import main

    rc = main(["--synthetic", "--dry-run", "--start", "2019-01-01", "--end", "2019-09-30"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "DRY-RUN" in out


def test_daily_run_dry_run_limit_prints_limit_column(monkeypatch, capsys,
                                                     pregate_engine_registry):
    # --order-type limit dry-run: prints the limit_price column, still zero HTTP.
    monkeypatch.setattr(ap.requests, "request", _no_http)
    from scripts.daily_run import main

    rc = main(["--synthetic", "--dry-run", "--order-type", "limit",
               "--limit-offset-bps", "5", "--start", "2019-01-01", "--end", "2019-09-30"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "DRY-RUN" in out
    assert "limit_price" in out
