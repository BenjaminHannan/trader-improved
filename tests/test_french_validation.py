"""Synthetic tests for scripts/validate_french.py's call-site wiring.

validate_against_french (production/risk/model.py) is a pure function that
expects (a) monthly-frequency data and (b) vendor column names ("Mkt-RF",
"Mom"). The curated lake's "french" dataset is DAILY and keyed by the loader's
internal series_id ("FF_MKT_RF", "FF_MOM", ... -- see
production/data/loaders/ken_french.py's COL_MAP). scripts/validate_french.py's
``wire_french_monthly`` bridges that gap at the call site. These tests exercise
the bridge directly with canned data -- no lake, no network -- and check both
the happy path (known correlation survives the compounding+renaming) and the
failure path (a missing/mismatched series_id raises instead of silently
correlating garbage).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from production.risk.model import validate_against_french
from scripts.validate_french import (FACTOR_TO_VENDOR, _monthly_compound,
                                     extra_factor_correlation,
                                     wire_french_monthly)


def _french_long(series_values: dict[str, pd.Series]) -> pd.DataFrame:
    """Build a canned curated-'french'-shaped long frame: obs_date, series_id,
    value (+ the mandatory lake columns, unused by the functions under test)."""
    frames = []
    for series_id, s in series_values.items():
        frames.append(pd.DataFrame({
            "obs_date": s.index, "series_id": series_id, "value": s.to_numpy(),
        }))
    return pd.concat(frames, ignore_index=True)


# ------------------------------------------------------------- _monthly_compound
def test_monthly_compound_matches_manual_geometric_return():
    dates = pd.bdate_range("2021-01-01", periods=40)
    r = pd.Series(np.linspace(-0.01, 0.01, len(dates)), index=dates, name="x")
    out = _monthly_compound(r.to_frame())
    expected = (1.0 + r).groupby(r.index.to_period("M")).prod() - 1.0
    pd.testing.assert_series_equal(out["x"], expected, check_names=False)


# ------------------------------------------------------------ wire_french_monthly
def test_wire_french_monthly_renames_series_id_and_compounds():
    # One observation per month (Jan..Apr) so the monthly-compounded value is
    # exactly the single day's return -- makes the expected output exact, not
    # just directionally close.
    dates = pd.to_datetime(["2020-01-15", "2020-02-15", "2020-03-15", "2020-04-15"])
    mkt = pd.Series([0.01, -0.02, 0.03, 0.00], index=dates)
    mom = pd.Series([0.005, 0.015, -0.01, 0.02], index=dates)
    french_long = _french_long({"FF_MKT_RF": mkt, "FF_MOM": mom})

    out = wire_french_monthly(french_long, factors={"market": "Mkt-RF", "momentum": "Mom"})

    assert list(out.columns) == ["Mkt-RF", "Mom"]
    assert len(out) == 4
    np.testing.assert_allclose(out["Mkt-RF"].to_numpy(), mkt.to_numpy(), atol=1e-12)
    np.testing.assert_allclose(out["Mom"].to_numpy(), mom.to_numpy(), atol=1e-12)


def test_wire_french_monthly_missing_series_id_raises_loudly():
    """The curated frame lacks FF_MOM entirely (e.g. a partial/broken ingest) --
    wiring must raise, never silently hand back an empty/misaligned momentum
    column that would masquerade as a near-zero correlation."""
    dates = pd.bdate_range("2020-01-01", periods=60)
    french_long = _french_long({"FF_MKT_RF": pd.Series(0.001, index=dates)})

    with pytest.raises(ValueError, match="FF_MOM"):
        wire_french_monthly(french_long, factors={"momentum": "Mom"})


def test_wire_french_monthly_unknown_vendor_name_raises_loudly():
    """A typo'd/unknown factor->vendor mapping (not in ken_french.COL_MAP) must
    raise rather than silently producing a NaN-only column."""
    dates = pd.bdate_range("2020-01-01", periods=60)
    french_long = _french_long({"FF_MKT_RF": pd.Series(0.001, index=dates)})

    with pytest.raises(ValueError, match="COL_MAP"):
        wire_french_monthly(french_long, factors={"market": "Not-A-Real-Column"})


def test_wire_french_monthly_empty_frame_raises():
    with pytest.raises(ValueError, match="empty"):
        wire_french_monthly(pd.DataFrame(columns=["obs_date", "series_id", "value"]))


# ------------------------------------------------------ end-to-end known-corr
def test_wiring_recovers_known_correlation_through_validate_against_french():
    """Canned DAILY estimated factor returns + a canned curated-french frame
    (internal series_id, not vendor names) with a known relationship -> after
    wire_french_monthly + validate_against_french, the recovered correlation
    matches manual computation on the same construction. This is the exact
    call-site path scripts/validate_french.py:run() uses.

    The benchmark side is built with exactly one observation per calendar
    month, so its "monthly compounding" is a no-op (single-element product)
    and the expected value is exact, not just directionally close."""
    rng = np.random.default_rng(9)
    dates = pd.bdate_range("2016-01-01", periods=750)
    mkt = rng.normal(0.0004, 0.01, len(dates))
    mom = rng.normal(0.0002, 0.008, len(dates))
    factor_returns = pd.DataFrame({"market": mkt, "momentum": mom}, index=dates)

    monthly = (1.0 + factor_returns).groupby(factor_returns.index.to_period("M")).prod() - 1.0
    nz = rng.normal(0, 1e-4, monthly.shape)
    bench_dates = monthly.index.to_timestamp()  # one row per month
    bench_mkt = pd.Series(monthly["market"].to_numpy() + nz[:, 0], index=bench_dates)
    bench_mom = pd.Series(monthly["momentum"].to_numpy() + nz[:, 1], index=bench_dates)

    french_long = _french_long({"FF_MKT_RF": bench_mkt, "FF_MOM": bench_mom})

    french_monthly = wire_french_monthly(french_long, factors={"market": "Mkt-RF",
                                                                "momentum": "Mom"})
    out = validate_against_french(factor_returns, french_monthly)

    # Reference: hand-rolled correlation of the same construction, computed
    # independently of both wire_french_monthly and validate_against_french.
    ref_mkt = float(pd.Series(monthly["market"].to_numpy())
                     .corr(pd.Series(monthly["market"].to_numpy() + nz[:, 0])))
    ref_mom = float(pd.Series(monthly["momentum"].to_numpy())
                     .corr(pd.Series(monthly["momentum"].to_numpy() + nz[:, 1])))
    assert out["market"] == pytest.approx(ref_mkt, abs=1e-9)
    assert out["momentum"] == pytest.approx(ref_mom, abs=1e-9)
    assert out["market"] > 0.9
    assert out["momentum"] > 0.9


# ------------------------------------------------------------- size (call-site)
def test_extra_factor_correlation_size_vs_smb_known_negative():
    """size is not in validate_against_french's fixed mapping; the call-site
    helper must recover a known correlation (here: deliberately negative, the
    documented large-cap-universe direction) via the identical monthly-
    compounding logic."""
    dates = pd.to_datetime(["2019-01-15", "2019-02-15", "2019-03-15", "2019-04-15",
                            "2019-05-15"])
    size = pd.Series([0.01, 0.02, -0.01, 0.015, -0.02], index=dates)
    smb = pd.Series([-0.01, -0.02, 0.01, -0.015, 0.02], index=dates)  # exact mirror -> corr = -1

    factor_returns = pd.DataFrame({"size": size})
    french_long = _french_long({"FF_SMB": smb})
    french_monthly = wire_french_monthly(french_long, factors={"size": "SMB"})

    corr = extra_factor_correlation(factor_returns, french_monthly, "size", "SMB")
    assert corr == pytest.approx(-1.0, abs=1e-9)


def test_extra_factor_correlation_missing_column_returns_none():
    factor_returns = pd.DataFrame({"market": [0.01, 0.02]},
                                  index=pd.bdate_range("2020-01-01", periods=2))
    french_monthly = pd.DataFrame({"Mkt-RF": [0.01, 0.02]},
                                  index=pd.PeriodIndex(["2020-01", "2020-02"], freq="M"))
    assert extra_factor_correlation(factor_returns, french_monthly, "size", "SMB") is None


def test_factor_to_vendor_matches_validate_against_french_mapping():
    """Guard against the two mappings silently drifting apart: the vendor names
    this script uses for market/momentum must be exactly what
    validate_against_french itself correlates against."""
    assert FACTOR_TO_VENDOR["market"] == "Mkt-RF"
    assert FACTOR_TO_VENDOR["momentum"] == "Mom"
