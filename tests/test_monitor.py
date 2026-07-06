"""Tests for live IC monitoring.

The decay/sign-flip alarm logic is exercised on hand-built IC series so the arithmetic
is fully determined. The embargo property is exercised end-to-end through
``live_rolling_ic``: corrupting the scores in the *unresolved* forward-return tail must
not move the live IC as of the last resolved date — the same corruption invariant the
``rolling_shrunk_ic`` tests enforce, one layer up.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from production.monitor.ic_monitor import (decay_alarms, live_rolling_ic,
                                           monitor_report, write_monitor_report)


# ------------------------------------------------------------------- decay_alarms
def _live(vals, start="2021-01-01"):
    idx = pd.bdate_range(start, periods=len(vals))
    return pd.Series(np.asarray(vals, dtype=float), index=idx)


def test_decay_alarm_fires_when_edge_collapses():
    live = _live(np.linspace(0.05, 0.0, 80))          # IC fades to zero
    alarms = decay_alarms(live, train_ic=0.05, factor="mom_12_1", ratio=0.25)
    kinds = {a["kind"] for a in alarms}
    assert "decay" in kinds                            # latest 0.0 < 0.25 * 0.05
    assert "sign_flip" not in kinds                    # never went negative
    hit = next(a for a in alarms if a["kind"] == "decay")
    assert hit["factor"] == "mom_12_1"
    assert hit["threshold"] == pytest.approx(0.25 * 0.05)
    assert hit["live_ic"] == pytest.approx(0.0)


def test_healthy_factor_stays_quiet():
    live = _live(np.full(80, 0.04))                    # holds near training IC
    alarms = decay_alarms(live, train_ic=0.05, factor="low_vol", ratio=0.25)
    assert alarms == []                                # 0.04 > 0.0125, never flips


def test_sign_flip_alarm():
    live = _live(np.full(20, -0.03))                   # persistently inverted
    alarms = decay_alarms(live, train_ic=0.05, factor="carry_funding",
                          ratio=0.25, flip_window=10)
    kinds = {a["kind"] for a in alarms}
    assert "sign_flip" in kinds                        # last 10 obs all aligned-negative
    assert "decay" in kinds                            # a flipped factor has also decayed


def test_sign_flip_needs_full_window():
    # only 6 aligned-negative obs but flip_window is 10 -> no persistent-flip alarm
    live = _live(np.full(6, -0.03))
    alarms = decay_alarms(live, train_ic=0.05, factor="tsmom",
                          ratio=0.25, flip_window=10)
    assert "sign_flip" not in {a["kind"] for a in alarms}


def test_same_sign_basis_for_negative_train_ic():
    # A factor whose training IC is negative is healthy when its live IC is also negative.
    healthy = _live(np.full(30, -0.04))
    assert decay_alarms(healthy, train_ic=-0.05, factor="str_reversal_1m",
                        ratio=0.25, flip_window=10) == []
    # ...and it is in trouble when its live IC goes positive (opposite the training sign).
    flipped = _live(np.full(30, 0.04))
    kinds = {a["kind"] for a in decay_alarms(flipped, train_ic=-0.05,
                                             factor="str_reversal_1m",
                                             ratio=0.25, flip_window=10)}
    assert "decay" in kinds and "sign_flip" in kinds


def test_decay_alarms_empty_series_is_quiet():
    assert decay_alarms(pd.Series(dtype=float), train_ic=0.05, factor="x") == []


# ----------------------------------------------------------------- live_rolling_ic
def _equity_panels(n_names=8, n_dates=40, seed=3):
    """Deterministic equity price panel + a rank-stable score panel.

    Scores are the (constant-in-time) name index, so every date has full dispersion and
    a well-defined rank IC against that date's forward returns.
    """
    rng = np.random.default_rng(seed)
    ids = [f"EQ:S{i}" for i in range(n_names)]
    sleeve_map = pd.Series({i: "equity" for i in ids})
    dates = pd.bdate_range("2021-01-01", periods=n_dates)
    price_rows, score_rows = [], []
    for ni, iid in enumerate(ids):
        rets = rng.normal(0.0004, 0.02, n_dates)
        close = 100.0 * (1 + ni) * np.cumprod(1.0 + rets)
        for ti, d in enumerate(dates):
            price_rows.append((d, iid, close[ti]))
            score_rows.append((d, iid, float(ni)))     # rank-stable score
    prices = pd.DataFrame(price_rows, columns=["obs_date", "instrument_id", "close"])
    scores = pd.DataFrame(score_rows, columns=["obs_date", "instrument_id", "value"])
    return scores, prices, sleeve_map


def test_live_rolling_ic_shape_and_embargo_warmup():
    scores, prices, sleeve_map = _equity_panels()
    live = live_rolling_ic(scores, prices, sleeve_map, horizon_days=5, window=100)
    assert list(live.columns) == ["obs_date", "sleeve", "live_ic"]
    assert set(live["sleeve"]) == {"equity"}
    eq = live[live.sleeve == "equity"].set_index("obs_date")["live_ic"].sort_index()
    # first `horizon` IC dates have no resolved history -> NaN warm-up
    assert eq.iloc[:5].isna().all()
    assert np.isfinite(eq.iloc[-1])


def test_live_rolling_ic_embargoes_unresolved_score_tail():
    scores, prices, sleeve_map = _equity_panels()
    horizon = 5
    live = live_rolling_ic(scores, prices, sleeve_map, horizon_days=horizon, window=100)
    eq = live[live.sleeve == "equity"].set_index("obs_date")["live_ic"].sort_index()
    t = eq.index[-1]
    base = eq.loc[t]
    assert np.isfinite(base)

    # The last `horizon` IC dates are exactly the ones embargoed from the live IC at t.
    tail_dates = eq.index[-horizon:]
    corrupt = scores.copy()
    mask = corrupt["obs_date"].isin(tail_dates)
    assert mask.sum() > 0
    # Reverse the scores on the tail dates: flips those dates' IC sign but keeps full
    # dispersion (no date collapses to NaN, so the IC index — and its positions — hold).
    corrupt.loc[mask, "value"] = corrupt.loc[mask, "value"].max() - corrupt.loc[mask, "value"]

    live2 = live_rolling_ic(corrupt, prices, sleeve_map, horizon_days=horizon, window=100)
    eq2 = live2[live2.sleeve == "equity"].set_index("obs_date")["live_ic"].sort_index()
    assert eq2.loc[t] == pytest.approx(base)           # unresolved tail cannot leak in


def test_live_rolling_ic_matches_manual_embargo_mean():
    from production.alpha.ic import forward_returns, rank_ic
    scores, prices, sleeve_map = _equity_panels()
    horizon, window = 5, 100
    live = live_rolling_ic(scores, prices, sleeve_map, horizon_days=horizon, window=window)
    eq = live[live.sleeve == "equity"].set_index("obs_date")["live_ic"].sort_index()

    fwd = forward_returns(prices, horizon)
    ic = rank_ic(scores, fwd, sleeve_map)
    by_date = ic[ic.sleeve == "equity"].set_index("obs_date")["rank_ic"].sort_index()
    t = eq.index[-1]
    i_t = list(by_date.index).index(t)
    eligible = by_date.iloc[: i_t - horizon + 1].iloc[-window:]
    assert eq.loc[t] == pytest.approx(float(eligible.mean()))


# ------------------------------------------------------- monitor_report roundtrip
def test_monitor_report_and_write_roundtrip(tmp_path):
    live = _live(np.linspace(0.05, 0.0, 80))
    alarms = decay_alarms(live, train_ic=0.05, factor="mom_12_1", ratio=0.25)
    table = pd.DataFrame([
        {"factor": "mom_12_1", "sleeve": "equity", "train_ic": 0.05,
         "live_ic": float(live.iloc[-1])},
        {"factor": "low_vol", "sleeve": "equity", "train_ic": 0.04,
         "live_ic": 0.038},
    ])
    report = monitor_report(alarms, table)
    assert report["n_alarms"] == len(alarms)
    assert report["n_factors"] == 2
    assert report["factors_in_alarm"] == ["mom_12_1"]

    path = write_monitor_report(report, out_dir=tmp_path)
    assert path.exists()
    assert path.name.startswith("monitor_") and path.suffix == ".json"
    loaded = json.loads(path.read_text())
    assert loaded["factors_in_alarm"] == ["mom_12_1"]
    assert loaded["alarms"][0]["factor"] == "mom_12_1"
    assert len(loaded["live_vs_train"]) == 2


def test_monitor_report_accepts_dict_table():
    report = monitor_report([], {"mom_12_1": {"train_ic": 0.05, "live_ic": 0.04}})
    assert report["n_alarms"] == 0
    assert report["factors_in_alarm"] == []
    assert report["live_vs_train"][0]["factor"] == "mom_12_1"


def test_write_monitor_report_coerces_non_finite(tmp_path):
    # A NaN live IC must not blow up json.dump(allow_nan=False) — it is coerced to null.
    report = monitor_report([], [{"factor": "x", "train_ic": 0.05,
                                  "live_ic": float("nan")}])
    path = write_monitor_report(report, out_dir=tmp_path)
    loaded = json.loads(path.read_text())
    assert loaded["live_vs_train"][0]["live_ic"] is None
