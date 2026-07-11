"""Q5 risk-exposure additions: ``liquidity`` and ``earnings_yield`` on
``production/risk/exposures.py::build_exposures``
(research/wiki/questions/research-risk-model-validation.md Q5 — capability +
tests only; the falsifiable bias-stat/R^2 adjudication runs later via the
harness once EDGAR fundamentals finish ingesting, and ``configs/risk.yaml`` is
untouched by this build).

All data is synthetic and hand-computable: no lake, no network. Every test
either (a) independently re-derives the expected value from the documented
arithmetic (ADV/mcap log-turnover; TTM-EPS/price) using numpy directly rather
than calling exposures.py's own helpers, or (b) diffs two full ``B`` matrices
the way ``tests/test_risk_model.py::test_exposures_pit_corruption`` already
does for the legacy factors.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from production.alpha.zscore import winsorize
from production.risk.exposures import build_exposures

UTC = "UTC"
AS_OF = pd.Timestamp("2021-06-30")

IDS = ["EQ:A:2000-01-01", "EQ:B:2000-01-01", "EQ:C:2000-01-01"]

# constant (close, dollar_volume) per id -- keeps trailing-ADV/last-close hand-computable
# regardless of how many price rows are supplied.
_PRICE_SPEC = {
    "EQ:A:2000-01-01": (10.0, 100_000.0),
    "EQ:B:2000-01-01": (20.0, 50_000.0),
    "EQ:C:2000-01-01": (25.0, 200_000.0),
}
_SHARES = {  # filed well before AS_OF
    "EQ:A:2000-01-01": 1_000_000.0,
    "EQ:B:2000-01-01": 2_000_000.0,
    "EQ:C:2000-01-01": 4_000_000.0,
}
# four quarters of EPS per id, filed well before AS_OF -> ttm = 4 * quarterly eps
_EPS_QUARTERLY = {
    "EQ:A:2000-01-01": 0.5,   # ttm 2.00 / close 10.0 -> ey 0.20
    "EQ:B:2000-01-01": 1.25,  # ttm 5.00 / close 20.0 -> ey 0.25
    "EQ:C:2000-01-01": 0.75,  # ttm 3.00 / close 25.0 -> ey 0.12
}
_QUARTER_ENDS = [pd.Timestamp("2020-06-30"), pd.Timestamp("2020-09-30"),
                 pd.Timestamp("2020-12-31"), pd.Timestamp("2021-03-31")]


def _cfg(factors: list[str], sleeve: str = "equity") -> dict:
    return {"exposures": {sleeve: {
        "factors": factors,
        "beta_window_days": 252, "size_adv_window_days": 63, "vol_window_days": 63,
    }}}


def _prices(spec: dict[str, tuple[float, float]] = _PRICE_SPEC, as_of=AS_OF,
           n_days: int = 5) -> pd.DataFrame:
    """Constant close/dollar_volume panel over the last ``n_days`` business days
    ending at ``as_of`` -- trailing means and the last-observation close are then
    exactly the constant, by construction."""
    dates = pd.bdate_range(end=as_of, periods=n_days)
    frames = []
    for iid, (close, dv) in spec.items():
        frames.append(pd.DataFrame({
            "obs_date": dates, "instrument_id": iid,
            "close": close, "volume": dv / close, "dollar_volume": dv,
        }))
    out = pd.concat(frames, ignore_index=True)
    avail = pd.to_datetime(out["obs_date"]).dt.tz_localize(UTC) + pd.Timedelta(hours=21, minutes=30)
    out["available_from"] = avail
    out["source"] = "synthetic:test"
    out["ingested_at"] = avail + pd.Timedelta(minutes=5)
    return out.sort_values(["obs_date", "instrument_id"]).reset_index(drop=True)


def _fundamentals(rows: list[tuple]) -> pd.DataFrame:
    """rows: ``(obs_date, instrument_id, field, value, available_from)`` tuples,
    curated-lake shaped (matches ``tests/conftest.py::make_fundamentals``)."""
    df = pd.DataFrame(rows, columns=["obs_date", "instrument_id", "field", "value",
                                     "available_from"])
    df["available_from"] = pd.to_datetime(df["available_from"], utc=True)
    df["source"] = "synthetic:test"
    df["ingested_at"] = df["available_from"] + pd.Timedelta(minutes=5)
    return df


def _baseline_fundamentals(shares=_SHARES, eps=_EPS_QUARTERLY,
                           shares_avail=AS_OF - pd.Timedelta(days=60)) -> pd.DataFrame:
    rows = []
    for iid, val in shares.items():
        rows.append((AS_OF - pd.Timedelta(days=90), iid, "shares", val, shares_avail))
    for iid, q_val in eps.items():
        for q_end in _QUARTER_ENDS:
            filed = q_end + pd.Timedelta(days=40)
            rows.append((q_end, iid, "eps", q_val, filed))
    return _fundamentals(rows)


# --------------------------------------------------------------- 1. arithmetic
def test_liquidity_and_earnings_yield_hand_computable():
    """Independently re-derive both factors from the documented formulas via
    numpy (not by calling exposures.py's internals) and compare."""
    prices = _prices()
    fundamentals = _baseline_fundamentals()
    cfg = _cfg(["liquidity", "earnings_yield"])

    B = build_exposures(prices, "equity", AS_OF, IDS, cfg, fundamentals=fundamentals)

    assert "liquidity" in B.columns and "earnings_yield" in B.columns
    assert not B.isna().any().any()

    close = np.array([_PRICE_SPEC[i][0] for i in IDS])
    adv = np.array([_PRICE_SPEC[i][1] for i in IDS])
    shares = np.array([_SHARES[i] for i in IDS])
    mcap = shares * close
    liq_raw = np.log(adv / mcap)
    liq_w = winsorize(liq_raw)
    liq_expected = (liq_w - liq_w.mean()) / liq_w.std()
    np.testing.assert_allclose(B.loc[IDS, "liquidity"].to_numpy(), liq_expected, atol=1e-10)
    # z-scoring convention matches the rest of the module: mean 0, population std 1.
    assert abs(B["liquidity"].mean()) < 1e-10
    assert abs(B["liquidity"].std(ddof=0) - 1.0) < 1e-10

    ttm = 4.0 * np.array([_EPS_QUARTERLY[i] for i in IDS])
    ey_raw = ttm / close
    ey_w = winsorize(ey_raw)
    ey_expected = (ey_w - ey_w.mean()) / ey_w.std()
    np.testing.assert_allclose(B.loc[IDS, "earnings_yield"].to_numpy(), ey_expected, atol=1e-10)
    assert abs(B["earnings_yield"].mean()) < 1e-10
    assert abs(B["earnings_yield"].std(ddof=0) - 1.0) < 1e-10


# ---------------------------------------------------------------------- 2. PIT
def test_liquidity_pit_lag_share_revision_invisible():
    """A shares revision filed AFTER as_of must not move the factor -- same
    contract as test_risk_model.py::test_exposures_pit_corruption."""
    prices = _prices()
    base = _baseline_fundamentals()
    cfg = _cfg(["liquidity"])
    B_base = build_exposures(prices, "equity", AS_OF, IDS, cfg, fundamentals=base)

    revised = pd.concat([base, _fundamentals([
        (AS_OF - pd.Timedelta(days=90), "EQ:A:2000-01-01", "shares", 9_000_000.0,
         AS_OF + pd.Timedelta(days=30)),  # filed after as_of -> must be invisible
    ])], ignore_index=True)
    B_after = build_exposures(prices, "equity", AS_OF, IDS, cfg, fundamentals=revised)

    pd.testing.assert_frame_equal(B_base, B_after)


def test_earnings_yield_pit_lag_eps_revision_invisible():
    """A restated EPS for an already-filed quarter, filed AFTER as_of, must not
    change the TTM figure used at as_of."""
    prices = _prices()
    base = _baseline_fundamentals()
    cfg = _cfg(["earnings_yield"])
    B_base = build_exposures(prices, "equity", AS_OF, IDS, cfg, fundamentals=base)

    revised = pd.concat([base, _fundamentals([
        (_QUARTER_ENDS[0], "EQ:A:2000-01-01", "eps", 5.0,
         AS_OF + pd.Timedelta(days=30)),  # restatement of an old quarter, filed late
    ])], ignore_index=True)
    B_after = build_exposures(prices, "equity", AS_OF, IDS, cfg, fundamentals=revised)

    pd.testing.assert_frame_equal(B_base, B_after)


# --------------------------------------------------------- 3. missing-input skip
@pytest.mark.parametrize("fundamentals", [None, pd.DataFrame()])
def test_missing_fundamentals_skips_both_factors(fundamentals):
    prices = _prices()
    cfg = _cfg(["liquidity", "earnings_yield"])

    with pytest.warns(UserWarning) as record:
        B = build_exposures(prices, "equity", AS_OF, IDS, cfg, fundamentals=fundamentals)

    assert "liquidity" not in B.columns
    assert "earnings_yield" not in B.columns
    assert list(B.columns) == ["market", "size", "momentum", "vol"]
    assert not B.isna().any().any()

    msgs = " ".join(str(w.message) for w in record)
    assert "liquidity" in msgs and "earnings_yield" in msgs


def test_zero_id_overlap_skips_both_factors():
    """Fundamentals present but for entirely different instruments -> same skip."""
    prices = _prices()
    cfg = _cfg(["liquidity", "earnings_yield"])
    other = _fundamentals([
        (AS_OF - pd.Timedelta(days=90), "EQ:ZZZ:1999-01-01", "shares", 1e6,
         AS_OF - pd.Timedelta(days=60)),
        (pd.Timestamp("2021-03-31"), "EQ:ZZZ:1999-01-01", "eps", 1.0,
         pd.Timestamp("2021-05-10")),
    ])

    with pytest.warns(UserWarning):
        B = build_exposures(prices, "equity", AS_OF, IDS, cfg, fundamentals=other)

    assert "liquidity" not in B.columns
    assert "earnings_yield" not in B.columns
    assert not B.isna().any().any()


def test_partial_coverage_missing_name_gets_zero_not_nan():
    """C has no fundamentals coverage; A and B do. The column is still built
    (coverage exists), C's entry is exactly 0.0 (the module's existing
    NaN->0 convention for a missing instrument), no NaN anywhere."""
    prices = _prices()
    cfg = _cfg(["liquidity", "earnings_yield"])
    shares = {k: v for k, v in _SHARES.items() if k != "EQ:C:2000-01-01"}
    eps = {k: v for k, v in _EPS_QUARTERLY.items() if k != "EQ:C:2000-01-01"}
    fundamentals = _baseline_fundamentals(shares=shares, eps=eps)

    B = build_exposures(prices, "equity", AS_OF, IDS, cfg, fundamentals=fundamentals)

    assert not B.isna().any().any()
    for col in ("liquidity", "earnings_yield"):
        assert B.loc["EQ:C:2000-01-01", col] == 0.0
        others = B.loc[["EQ:A:2000-01-01", "EQ:B:2000-01-01"], col].to_numpy()
        # exactly two finite points -> z-score is always +-1.0, never clipped
        # (winsorize's 3*MAD band around the midpoint always exceeds each
        # point's distance from it for n=2) -- see module docstring.
        np.testing.assert_allclose(sorted(np.abs(others)), [1.0, 1.0], atol=1e-10)
        assert np.isclose(others.sum(), 0.0, atol=1e-10)


# ------------------------------------------------------- 4. config-off bit-identical
def test_config_off_bit_identical_to_before(price_panel, sleeve_of, fundamentals_panel):
    """Real, untouched configs/risk.yaml never lists liquidity/earnings_yield,
    so build_exposures output is byte-identical whether or not a fundamentals
    panel happens to be supplied -- the new code path is strictly config-gated."""
    ids = [i for i, s in sleeve_of.items() if s == "equity"]
    as_of = pd.Timestamp("2021-12-31")

    without = build_exposures(price_panel, "equity", as_of, ids)
    with_fundamentals = build_exposures(price_panel, "equity", as_of, ids,
                                        fundamentals=fundamentals_panel)

    pd.testing.assert_frame_equal(without, with_fundamentals)
    assert list(without.columns) == ["market", "size", "momentum", "vol"]


def test_config_missing_factors_key_also_skips():
    """Defensive: a cfg with no 'factors' key at all under exposures.<sleeve>
    (scfg.get('factors', [])) still never activates the new factors, even with
    fundamentals supplied."""
    prices = _prices()
    fundamentals = _baseline_fundamentals()
    cfg = {"exposures": {"equity": {
        "beta_window_days": 252, "size_adv_window_days": 63, "vol_window_days": 63,
    }}}

    B = build_exposures(prices, "equity", AS_OF, IDS, cfg, fundamentals=fundamentals)

    assert list(B.columns) == ["market", "size", "momentum", "vol"]


# ------------------------------------------------------------ 5. no sector demean
def test_earnings_yield_and_liquidity_unaffected_by_sectors():
    """Sector dummies are appended as extra columns; they must not alter the
    liquidity/earnings_yield values themselves (no sector-demeaning) -- as a
    risk exposure the job is to explain co-movement including sector effects,
    unlike the rejected alpha's sector-neutral framing."""
    prices = _prices()
    fundamentals = _baseline_fundamentals()
    cfg = _cfg(["liquidity", "earnings_yield"])
    sectors = pd.Series({"EQ:A:2000-01-01": "tech", "EQ:B:2000-01-01": "tech",
                         "EQ:C:2000-01-01": "energy"})

    B_plain = build_exposures(prices, "equity", AS_OF, IDS, cfg, fundamentals=fundamentals)
    B_sectored = build_exposures(prices, "equity", AS_OF, IDS, cfg, sectors=sectors,
                                 fundamentals=fundamentals)

    pd.testing.assert_series_equal(B_plain["liquidity"], B_sectored["liquidity"])
    pd.testing.assert_series_equal(B_plain["earnings_yield"], B_sectored["earnings_yield"])
    assert any(c.startswith("sector_") for c in B_sectored.columns)
