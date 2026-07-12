"""Cost model tests — floors, sqrt impact, cap, PIT exclusion of the as_of day."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from production.core.config import costs_config
from production.backtest.cost_model import CostModel, trailing_adv_sigma

SLEEVES = ["equity", "crypto", "fx_etf", "commodity_etf"]


@pytest.fixture()
def model() -> CostModel:
    return CostModel()


@pytest.mark.parametrize("sleeve", SLEEVES)
def test_floor_binds_for_tiny_trades(model, sleeve):
    """A trade tiny relative to ADV yields impact ~ 0, so the per-sleeve floor binds."""
    floor = costs_config()["sleeves"][sleeve]["floor_bps"]
    cost = model.cost_bps(trade_usd=1.0, adv_usd=1e9, sigma_daily=0.01,
                          sleeve=sleeve, instrument_ids=["X"]).iloc[0]
    assert cost == pytest.approx(floor)


def test_sqrt_impact_monotone_increasing(model):
    """Above the floor, cost is strictly increasing in trade size (sqrt impact)."""
    trades = np.array([2e5, 1e6, 5e6, 2e7, 1.5e8])
    # Sized so every point clears the floor yet stays under the cap (strictly increasing).
    costs = model.cost_bps(trade_usd=pd.Series(trades, index=range(len(trades))),
                           adv_usd=1e8, sigma_daily=0.05, sleeve="equity").to_numpy()
    floor = costs_config()["sleeves"]["equity"]["floor_bps"]
    assert np.all(costs > floor)                      # above the floor
    assert np.all(np.diff(costs) > 0)                 # strictly increasing


def test_cap_and_warning(model):
    """Enormous trade/ADV ratio pushes cost past the 100bp cap and records a warning."""
    cap = costs_config()["cap_bps"]
    cost = model.cost_bps(trade_usd=1e12, adv_usd=1e3, sigma_daily=0.5,
                          sleeve="crypto", instrument_ids=["CR:BTC:2017-01-01"]).iloc[0]
    assert cost == pytest.approx(cap)
    assert any(w["reason"] == "cap" for w in model.cap_warnings)


def test_zero_trade_zero_cost(model):
    cost = model.cost_bps(trade_usd=0.0, adv_usd=1e6, sigma_daily=0.01,
                          sleeve="equity", instrument_ids=["X"]).iloc[0]
    assert cost == 0.0


def test_missing_adv_uses_cap(model):
    cap = costs_config()["cap_bps"]
    out = model.cost_bps(trade_usd=pd.Series([1e5, 1e5], index=["A", "B"]),
                         adv_usd=pd.Series([np.nan, 0.0], index=["A", "B"]),
                         sigma_daily=0.01, sleeve="equity")
    assert out.loc["A"] == pytest.approx(cap)
    assert out.loc["B"] == pytest.approx(cap)
    assert all(w["reason"] == "missing_adv" for w in model.cap_warnings)


def test_instrument_override_honored():
    cfg = costs_config().copy()
    cfg = dict(cfg)
    cfg["instrument_overrides"] = {"EQ:PALL:2010-01-08": {"floor_bps": 20.0, "half_spread_bps": 10.0}}
    model = CostModel(cfg)
    # Tiny trade -> floor binds; override floor (20) beats the equity default (5).
    cost = model.cost_bps(trade_usd=1.0, adv_usd=1e9, sigma_daily=0.01,
                          sleeve="equity", instrument_ids=["EQ:PALL:2010-01-08"]).iloc[0]
    assert cost == pytest.approx(20.0)


# ----------------------------------------------------- lake-calibrated overrides_table
_A, _B, _C = "EQ:A:2000-01-03", "EQ:B:2000-01-03", "EQ:C:2000-01-03"


def test_overrides_table_merges_over_yaml_and_is_scoped():
    """overrides_table (TCA-calibrated) merges OVER the yaml overrides and changes
    cost_bps for exactly the instruments it names — others are untouched."""
    cfg = dict(costs_config())
    cfg["instrument_overrides"] = {_B: {"floor_bps": 40.0, "half_spread_bps": 30.0}}
    table = {_A: {"half_spread_bps": 50.0, "n_fills": 25, "median_abs_shortfall_bps": 40.0}}
    model = CostModel(cfg, overrides_table=table)
    base = CostModel(cfg)                       # yaml overrides only, no table

    idx = [_A, _B, _C]
    # trade 1e6 / adv 1e8 / sigma 0.05 -> impact = 1e4*0.15*0.05*sqrt(0.01) = 7.5 bps
    kw = dict(trade_usd=pd.Series([1e6] * 3, index=idx), adv_usd=1e8,
              sigma_daily=0.05, sleeve="equity", instrument_ids=idx)
    out = model.cost_bps(**kw)
    out_base = base.cost_bps(**kw)

    # A: calibrated half_spread 50 wins -> max(5, 50+7.5) = 57.5 (changed vs default 10)
    assert out.loc[_A] == pytest.approx(57.5)
    assert out.loc[_A] != pytest.approx(out_base.loc[_A])
    # B: yaml override survives the merge (table doesn't name it) -> max(40, 30+7.5) = 40
    assert out.loc[_B] == pytest.approx(40.0)
    assert out.loc[_B] == pytest.approx(out_base.loc[_B])
    # C: no override anywhere -> equity default max(5, 2.5+7.5) = 10, unchanged by the table
    assert out.loc[_C] == pytest.approx(10.0)
    assert out.loc[_C] == pytest.approx(out_base.loc[_C])


def test_overrides_table_none_is_bit_identical():
    """overrides_table=None reproduces the yaml-only path bit-for-bit."""
    cfg = dict(costs_config())
    cfg["instrument_overrides"] = {_B: {"floor_bps": 40.0, "half_spread_bps": 30.0}}
    idx = [_A, _B, _C, "D"]
    trades = pd.Series([0.0, 1e6, 5e6, 1e5], index=idx)
    adv = pd.Series([1e8, 1e8, 1e8, np.nan], index=idx)   # D -> missing ADV -> cap
    sigma = pd.Series([0.05, 0.05, 0.05, 0.05], index=idx)
    a = CostModel(cfg).cost_bps(trades, adv, sigma, "equity", instrument_ids=idx)
    b = CostModel(cfg, overrides_table=None).cost_bps(trades, adv, sigma, "equity",
                                                      instrument_ids=idx)
    assert a.to_numpy().tolist() == b.to_numpy().tolist()


# ------------------------------------------------------------- floor enforcement (CLAUDE.md)
def test_overrides_table_half_spread_below_floor_never_lowers_cost():
    """A TCA-calibrated override whose half_spread_bps sits below the sleeve floor cannot
    lower the charged cost. This proves the pre-existing enforcement (cost_bps always
    charges max(floor, half_spread + impact)) rather than re-implementing it — a floor is a
    floor no matter how low a calibrated override's spread candidate is."""
    floor = costs_config()["sleeves"]["equity"]["floor_bps"]
    assert floor > 0.0
    table = {_A: {"half_spread_bps": floor - 1.0, "n_fills": 25,
                 "median_abs_shortfall_bps": floor - 1.0}}
    model = CostModel(overrides_table=table)
    # Tiny trade -> impact ~ 0, so an unclamped half_spread would be the whole charge.
    cost = model.cost_bps(trade_usd=1.0, adv_usd=1e9, sigma_daily=0.01,
                          sleeve="equity", instrument_ids=[_A]).iloc[0]
    assert cost == pytest.approx(floor)


def test_floor_bps_override_below_sleeve_floor_is_clamped_with_warning():
    """A floor_bps override may only RAISE a sleeve's floor, never lower it (CLAUDE.md,
    non-negotiable). One requesting a lower floor is clamped back to the sleeve default and
    recorded in cap_warnings — never silently honored. (test_instrument_override_honored
    above already covers the legitimate raise-the-floor case.)"""
    cfg = dict(costs_config())
    floor = cfg["sleeves"]["equity"]["floor_bps"]
    cfg["instrument_overrides"] = {_A: {"floor_bps": floor - 2.0, "half_spread_bps": 0.0}}
    model = CostModel(cfg)
    cost = model.cost_bps(trade_usd=1.0, adv_usd=1e9, sigma_daily=0.01,
                          sleeve="equity", instrument_ids=[_A]).iloc[0]
    assert cost == pytest.approx(floor)            # never lowered below the sleeve floor
    assert any(w["reason"] == "floor_override_below_sleeve_floor" and w["instrument_id"] == _A
              for w in model.cap_warnings)


def test_floor_bps_override_at_or_above_sleeve_floor_is_not_flagged():
    """A floor_bps override that legitimately raises the floor produces no clamp warning."""
    cfg = dict(costs_config())
    floor = cfg["sleeves"]["equity"]["floor_bps"]
    cfg["instrument_overrides"] = {_A: {"floor_bps": floor + 5.0}}
    model = CostModel(cfg)
    model.cost_bps(trade_usd=1.0, adv_usd=1e9, sigma_daily=0.01,
                   sleeve="equity", instrument_ids=[_A])
    assert not any(w["reason"] == "floor_override_below_sleeve_floor"
                  for w in model.cap_warnings)


# --------------------------------------------------------- trailing_adv_sigma PIT
def _toy_prices(n=60, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    vol = rng.lognormal(13, 0.2, n)
    return pd.DataFrame({
        "obs_date": dates, "instrument_id": "EQ:SYN00:2000-01-03",
        "close": close, "volume": vol, "dollar_volume": close * vol,
    })


def test_trailing_adv_sigma_excludes_as_of_day():
    """Corrupting the as_of-day close/volume must not change ADV or sigma (shift(1))."""
    prices = _toy_prices()
    as_of = prices["obs_date"].iloc[40]
    adv0, sig0 = trailing_adv_sigma(prices, as_of, window=20)

    corrupt = prices.copy()
    mask = corrupt["obs_date"] == as_of
    corrupt.loc[mask, "close"] *= 5.0
    corrupt.loc[mask, "volume"] *= 100.0
    corrupt.loc[mask, "dollar_volume"] = corrupt.loc[mask, "close"] * corrupt.loc[mask, "volume"]
    adv1, sig1 = trailing_adv_sigma(corrupt, as_of, window=20)

    assert adv0.iloc[0] == pytest.approx(adv1.iloc[0])
    assert sig0.iloc[0] == pytest.approx(sig1.iloc[0])


# ------------------------------------------------------ cost ledger + sensitivity
def test_ledger_off_by_default(model):
    """No ledger entry unless record=True (optimizer probes must not pollute the tally)."""
    model.cost_bps(trade_usd=5e6, adv_usd=1e8, sigma_daily=0.05, sleeve="equity",
                   instrument_ids=["X"])
    assert model.ledger == []


def test_ledger_reconciles_charged_bit_exact(model):
    """The ledger's per-name charged vector and its notional-weighted total reconcile
    bit-for-bit with what cost_bps returned (mixed floor / impact / cap / missing-ADV)."""
    idx = ["A", "B", "C", "D", "E"]
    trades = pd.Series([0.0, 1.0, 5e6, 2e7, 1e5], index=idx)      # zero / tiny / mid / big / mid
    adv = pd.Series([1e8, 1e9, 1e8, 1e8, np.nan], index=idx)      # E has missing ADV -> cap
    sigma = pd.Series([0.05, 0.01, 0.05, 0.05, 0.05], index=idx)
    returned = model.cost_bps(trades, adv, sigma, "equity", instrument_ids=idx, record=True)

    assert len(model.ledger) == 1
    e = model.ledger[0]
    # per-name charged vector is a bit-exact copy of the returned Series
    assert e["charged"] == returned.to_numpy().tolist()
    # notional-weighted charged reconciles bit-exact
    notion = trades.abs().to_numpy()
    assert e["charged_bps_x_notional"] == float((returned.to_numpy() * notion).sum())
    assert e["total_notional"] == float(notion.sum())


def test_cost_sensitivity_m1_reconciles_to_one(model):
    """The counterfactual at multiplier 1.0 equals the realized charge exactly."""
    model.cost_bps(pd.Series([5e6, 2e7], index=["A", "B"]), adv_usd=1e8,
                   sigma_daily=0.05, sleeve="equity", record=True)
    model.cost_bps(pd.Series([3e6, 1e7], index=["C", "D"]), adv_usd=8e7,
                   sigma_daily=0.04, sleeve="commodity_etf", record=True)
    sens = model.cost_sensitivity()
    assert sens[1.0] == 1.0


def test_cost_sensitivity_floor_bound_invariant(model):
    """Floor-bound trades (impact ~ 0) are invariant to the impact multiplier."""
    model.cost_bps(pd.Series([1.0, 2.0, 3.0], index=["A", "B", "C"]),
                   adv_usd=1e12, sigma_daily=0.01, sleeve="equity", record=True)
    sens = model.cost_sensitivity(multipliers=(1.0, 10.0 / 3.0, 20.0 / 3.0))
    for r in sens.values():
        assert r == pytest.approx(1.0)


def test_cost_sensitivity_monotone_nondecreasing(model):
    """Above the floor, drag is monotone nondecreasing in the impact multiplier, and
    strictly grows for impact-dominated trades."""
    model.cost_bps(pd.Series([5e6, 2e7, 8e7], index=["A", "B", "C"]),
                   adv_usd=1e8, sigma_daily=0.05, sleeve="equity", record=True)
    ms = (1.0, 2.0, 3.0, 5.0)
    sens = model.cost_sensitivity(multipliers=ms)
    vals = [sens[m] for m in ms]
    assert vals[0] == pytest.approx(1.0)
    assert np.all(np.diff(vals) >= -1e-12)     # nondecreasing
    assert vals[-1] > vals[0]                  # impact-dominated -> genuinely grows


def test_trailing_adv_sigma_ignores_days_after_as_of():
    """Rows dated after as_of are filtered out entirely and cannot leak in."""
    prices = _toy_prices()
    as_of = prices["obs_date"].iloc[40]
    adv0, sig0 = trailing_adv_sigma(prices, as_of, window=20)

    corrupt = prices.copy()
    after = corrupt["obs_date"] > as_of
    corrupt.loc[after, "close"] *= 10.0
    corrupt.loc[after, "dollar_volume"] *= 1000.0
    adv1, sig1 = trailing_adv_sigma(corrupt, as_of, window=20)

    assert adv0.iloc[0] == pytest.approx(adv1.iloc[0])
    assert sig0.iloc[0] == pytest.approx(sig1.iloc[0])
