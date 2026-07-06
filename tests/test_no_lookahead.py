"""The gold-standard look-ahead corruption harness for signals.

Every registered signal is a pure point-in-time function: its value at date D may use
only inputs knowable by end of day D. This suite proves that mechanically. The core
test (``test_signal_no_lookahead``) is parametrized over the ENTIRE registry, so any
newly registered signal is automatically subjected to it — there is no opt-out.

Mechanics of the corruption test: compute a signal on the full synthetic bundle, pick
a pivot date ~70% through the date range, then rebuild every input frame two ways —
(1) NaN-ing every value column for rows with ``obs_date > pivot`` and (2) DROPPING
those rows entirely — recompute, and assert the output restricted to
``obs_date <= pivot`` is bit-identical under both. If a signal peeked at the future,
mutating the future would move a past value and the assertion would fail.

Additional tests: long-format/no-NaN shape, sleeve isolation, min_history behaviour,
analytic known-answer checks for momentum/reversal, and the COT release-lag check.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from production.signals.base import all_signals, sleeve_from_id

# Every value column that any input frame carries. Corrupting all of them at once is
# the strongest test: a signal must not depend on any future value cell.
VALUE_COLS = ["close", "volume", "dollar_volume", "funding_rate", "value",
              "noncomm_net", "open_interest", "mcap", "tvl"]

REGISTERED = sorted(all_signals())


# ------------------------------------------------------------------- utilities
def _pivot_date(bundle: dict[str, pd.DataFrame]) -> pd.Timestamp:
    """A date ~70% through the union of all obs_dates in the bundle."""
    dates = pd.concat([pd.to_datetime(df["obs_date"]) for df in bundle.values()])
    uniq = np.sort(dates.unique())
    return pd.Timestamp(uniq[int(len(uniq) * 0.70)])


def _corrupt(bundle: dict[str, pd.DataFrame], pivot: pd.Timestamp,
             mode: str) -> dict[str, pd.DataFrame]:
    """Return a copy of the bundle with the future tail (obs_date > pivot) mangled.

    mode="nan": blank every value column for future rows.
    mode="drop": remove future rows entirely.
    """
    out = {}
    for key, df in bundle.items():
        df = df.copy()
        future = pd.to_datetime(df["obs_date"]) > pivot
        if mode == "nan":
            cols = [c for c in VALUE_COLS if c in df.columns]
            df.loc[future, cols] = np.nan
        elif mode == "drop":
            df = df.loc[~future].reset_index(drop=True)
        else:  # pragma: no cover - defensive
            raise ValueError(mode)
        out[key] = df
    return out


def _upto(panel: pd.DataFrame, pivot: pd.Timestamp) -> pd.DataFrame:
    keep = panel.loc[pd.to_datetime(panel["obs_date"]) <= pivot]
    return keep.sort_values(["obs_date", "instrument_id"], kind="stable").reset_index(drop=True)


# --------------------------------------------------------- the corruption gate
@pytest.mark.parametrize("name", REGISTERED)
def test_signal_no_lookahead(name, signal_data):
    """Past values must be invariant to any change in the future tail of the inputs."""
    sig = all_signals()[name]()
    full = sig.compute(signal_data)
    pivot = _pivot_date(signal_data)
    baseline = _upto(full, pivot)

    for mode in ("nan", "drop"):
        corrupted = _corrupt(signal_data, pivot, mode)
        recomputed = _upto(sig.compute(corrupted), pivot)
        pd.testing.assert_frame_equal(
            baseline, recomputed, check_dtype=False,
            obj=f"{name} under future-{mode} corruption",
        )


# ------------------------------------------------------------- shape contracts
@pytest.mark.parametrize("name", REGISTERED)
def test_output_is_clean_long_format(name, signal_data):
    out = all_signals()[name]().compute(signal_data)
    assert list(out.columns) == ["obs_date", "instrument_id", "value"]
    assert not out["value"].isna().any()
    assert out[["obs_date", "instrument_id"]].notna().all().all()


@pytest.mark.parametrize("name", REGISTERED)
def test_signal_only_emits_its_own_sleeves(name, signal_data):
    sig = all_signals()[name]()
    out = sig.compute(signal_data)
    emitted = {sleeve_from_id(i) for i in out["instrument_id"].unique()}
    assert emitted <= set(sig.sleeves)


def test_registry_contains_expected_signals():
    reg = all_signals()
    expected = {"mom_12_1", "str_reversal_1m", "tsmom", "low_vol", "carry_funding",
                "carry_rate_diff", "cot_positioning", "lt_reversal_5y"}
    assert expected <= set(reg)
    # lt_reversal_5y is a code-level candidate — registered but NOT in factors.yaml.
    from production.core.config import factors_config
    assert "lt_reversal_5y" not in factors_config()["factors"]


# ---------------------------------------------------------- min_history bound
def test_min_history_first_valid_date(signal_data):
    """No mom_12_1 value before data_start + min_history_days (calendar), on any sleeve."""
    sig = all_signals()["mom_12_1"]()
    out = sig.compute(signal_data)
    start = pd.to_datetime(signal_data["prices"]["obs_date"]).min()
    floor = start + pd.Timedelta(days=sig.min_history_days)
    assert out["obs_date"].min() >= floor


# ------------------------------------------------------- analytic known answers
def _constant_growth_panel(iid: str, g: float, n: int = 400,
                           start: str = "2015-01-02") -> pd.DataFrame:
    """A single-instrument curated price frame with close = 100*(1+g)**t on bdays."""
    dates = pd.bdate_range(start, periods=n)
    close = 100.0 * (1.0 + g) ** np.arange(n)
    avail = dates.tz_localize("UTC") + pd.Timedelta(hours=21, minutes=30)
    return pd.DataFrame({
        "obs_date": dates, "instrument_id": iid, "close": close,
        "volume": 1e6, "dollar_volume": close * 1e6,
        "available_from": avail, "source": "test", "ingested_at": avail,
    })


def test_momentum_known_answer():
    """Constant daily growth g => mom_12_1 == (1+g)**231 - 1 exactly (231 = 252-21)."""
    g = 0.001
    iid = "EQ:KAT:2015-01-02"
    data = {"prices": _constant_growth_panel(iid, g)}
    out = all_signals()["mom_12_1"]().compute(data)
    assert not out.empty
    expected = (1.0 + g) ** 231 - 1.0
    np.testing.assert_allclose(out["value"].to_numpy(), expected, rtol=1e-12)


def test_short_term_reversal_known_answer():
    """Constant daily growth g => str_reversal_1m == -((1+g)**21 - 1) exactly."""
    g = 0.001
    iid = "EQ:KAT:2015-01-02"
    data = {"prices": _constant_growth_panel(iid, g)}
    out = all_signals()["str_reversal_1m"]().compute(data)
    assert not out.empty
    expected = -((1.0 + g) ** 21 - 1.0)
    np.testing.assert_allclose(out["value"].to_numpy(), expected, rtol=1e-12)


# ------------------------------------------------------------- COT release lag
def test_cot_positioning_release_lag(price_panel, cot_panel):
    """A Tuesday COT obs must be invisible on Wednesday but visible the next Friday.

    We compare the full-data signal against one recomputed with the Tuesday obs (and
    everything after) removed: the Wednesday value is unchanged (does NOT yet use the
    Tuesday obs), while the Friday value DOES change (the release lands on Friday).
    """
    iid = "CO:GLD:2006-01-03"
    prices_sub = price_panel[price_panel["instrument_id"] == iid]
    cot_sub = cot_panel[cot_panel["instrument_id"] == iid]

    tuesdays = np.sort(cot_sub["obs_date"].unique())
    tuesday = pd.Timestamp(tuesdays[110])          # deep enough for the 52-week min_periods
    wednesday = tuesday + pd.Timedelta(days=1)      # between obs and release
    friday = tuesday + pd.Timedelta(days=3)         # the release date

    sig = all_signals()["cot_positioning"]()
    full = sig.compute({"prices": prices_sub, "cot": cot_sub})
    full_s = full.set_index("obs_date")["value"]
    assert wednesday in full_s.index and friday in full_s.index

    trunc_cot = cot_sub[cot_sub["obs_date"] < tuesday]
    trunc = sig.compute({"prices": prices_sub, "cot": trunc_cot})
    trunc_s = trunc.set_index("obs_date")["value"]

    # Wednesday: identical with or without the Tuesday obs -> it is NOT yet reflected.
    assert trunc_s.loc[wednesday] == full_s.loc[wednesday]
    # Friday: the release lands, so removing the Tuesday obs changes the value.
    assert trunc_s.loc[friday] != full_s.loc[friday]


# ----------------------------------------------- macro vintage corruption (carry_rate_diff)
def test_carry_rate_diff_vintage_corruption(price_panel, macro_panel):
    """A macro revision knowable only after the pivot cannot touch pre-pivot signal values.

    Append fake REVISED rows for every macro obs (same obs_dates, values * 100) whose
    availability is shifted +60d — then keep only those whose availability lands strictly
    after the pivot (covering both obs after the pivot, naturally future, and obs on/before
    it whose shifted availability is post-pivot). Revisions knowable only post-pivot must
    leave every value at obs_date <= pivot bit-identical to the unrevised run.
    """
    sig = all_signals()["carry_rate_diff"]()
    baseline = sig.compute({"prices": price_panel, "macro": macro_panel})

    macro_dates = np.sort(pd.to_datetime(macro_panel["obs_date"]).unique())
    pivot = pd.Timestamp(macro_dates[int(len(macro_dates) * 0.70)])

    rev = macro_panel.copy()
    rev["value"] = rev["value"] * 100.0
    rev_avail = pd.to_datetime(rev["available_from"], utc=True) + pd.Timedelta(days=60)
    rev["available_from"] = rev_avail
    if "ingested_at" in rev.columns:
        rev["ingested_at"] = pd.to_datetime(rev["ingested_at"], utc=True) + pd.Timedelta(days=60)
    rev["source"] = "synthetic:macro:revision"
    # Keep only revisions knowable strictly after the pivot (post-pivot availability date).
    keep = rev_avail.dt.normalize().dt.tz_localize(None) > pivot
    rev = rev.loc[keep]
    assert not rev.empty                                   # the corruption is real
    corrupted_macro = pd.concat([macro_panel, rev], ignore_index=True)

    recomputed = sig.compute({"prices": price_panel, "macro": corrupted_macro})

    def _upto_pivot(df):
        return (df[pd.to_datetime(df["obs_date"]) <= pivot]
                .sort_values(["obs_date", "instrument_id"], kind="stable")
                .reset_index(drop=True))

    pd.testing.assert_frame_equal(_upto_pivot(baseline), _upto_pivot(recomputed),
                                  check_dtype=False,
                                  obj="carry_rate_diff under post-pivot macro revisions")
    # Non-vacuous: the revisions DO bite once knowable (post-pivot values move).
    assert not baseline.equals(recomputed)
