"""Tests for Combinatorial Purged Cross-Validation and the single-strategy PBO.

Split-invariant tests are corruption-style: they assert directly that no training date
ever lands inside the purge/embargo zone of any test block (the machine-checked PIT
guarantee). The PBO tests plant two factors — a persistently predictive one and one whose
edge is confined to (and reverses outside of) the first part of the sample — and check
that CPCV separates them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from production.backtest.cpcv import (cpcv_factor_eval, cpcv_splits, n_splits, pbo)


# --------------------------------------------------------------------- cpcv_splits
def test_number_of_splits_is_c_n_k():
    dates = pd.bdate_range("2018-01-01", periods=300)
    splits = cpcv_splits(dates, n_groups=6, k_test=2)
    assert len(splits) == 15          # C(6, 2)
    assert n_splits(6, 2) == 15


def test_number_of_splits_other_combinations():
    dates = pd.bdate_range("2018-01-01", periods=300)
    assert len(cpcv_splits(dates, n_groups=5, k_test=2)) == 10   # C(5, 2)
    assert len(cpcv_splits(dates, n_groups=6, k_test=3)) == 20   # C(6, 3)


def test_train_and_test_are_disjoint():
    dates = pd.bdate_range("2018-01-01", periods=300)
    for train, test in cpcv_splits(dates, n_groups=6, k_test=2):
        assert set(train).isdisjoint(set(test))


def test_test_blocks_union_covers_two_groups_worth():
    # With k_test=2 and no purge/embargo, every date is either train or test, and the
    # test side is exactly the union of the two chosen contiguous blocks.
    dates = pd.bdate_range("2018-01-01", periods=300)
    splits = cpcv_splits(dates, n_groups=6, k_test=2, purge_days=0, embargo_days=0)
    for train, test in splits:
        assert len(train) + len(test) == len(dates)


def test_purge_and_embargo_remove_train_dates_around_test():
    dates = pd.bdate_range("2018-01-01", periods=300)
    no_guard = cpcv_splits(dates, n_groups=6, k_test=2, purge_days=0, embargo_days=0)
    guarded = cpcv_splits(dates, n_groups=6, k_test=2, purge_days=21, embargo_days=5)
    # Guarding can only shrink (never grow) the training set on every split.
    for (tr0, _), (tr1, _) in zip(no_guard, guarded):
        assert len(tr1) < len(tr0)


def test_no_train_date_within_purge_or_embargo_of_test():
    # Corruption-style property: reconstruct the positional purge/embargo zone of every
    # contiguous test block and assert training positions never intersect it.
    dates = pd.bdate_range("2018-01-01", periods=300)
    pos_of = {d: i for i, d in enumerate(dates)}
    purge, embargo = 21, 5
    for train, test in cpcv_splits(dates, n_groups=6, k_test=2,
                                   purge_days=purge, embargo_days=embargo):
        train_pos = np.array([pos_of[d] for d in train])
        test_pos = sorted(pos_of[d] for d in test)
        # split the (possibly two) test blocks into contiguous runs
        runs, run = [], [test_pos[0]]
        for p in test_pos[1:]:
            if p == run[-1] + 1:
                run.append(p)
            else:
                runs.append((run[0], run[-1]))
                run = [p]
        runs.append((run[0], run[-1]))
        for lo, hi in runs:
            forbidden_lo = lo - purge                      # purge, backward side
            forbidden_hi = hi + max(purge, embargo)        # purge + embargo, forward side
            leak = train_pos[(train_pos >= forbidden_lo) & (train_pos <= forbidden_hi)]
            assert leak.size == 0


def test_dates_sorted_internally():
    # Unsorted input must not break the contiguous-group logic.
    dates = pd.bdate_range("2018-01-01", periods=120)
    shuffled = pd.DatetimeIndex(np.random.default_rng(0).permutation(dates.values))
    splits = cpcv_splits(shuffled, n_groups=6, k_test=2)
    assert len(splits) == 15
    for train, _ in splits:
        assert train.is_monotonic_increasing


def test_invalid_parameters_raise():
    dates = pd.bdate_range("2018-01-01", periods=100)
    with pytest.raises(ValueError):
        cpcv_splits(dates, n_groups=1, k_test=1)
    with pytest.raises(ValueError):
        cpcv_splits(dates, n_groups=6, k_test=6)      # k_test must be < n_groups
    with pytest.raises(ValueError):
        cpcv_splits(pd.bdate_range("2018-01-01", periods=3), n_groups=6, k_test=2)


# ----------------------------------------------------------------------------- pbo
def test_pbo_all_same_sign_is_zero():
    train = pd.Series([0.1, 0.2, 0.05, 0.3])
    test = pd.Series([0.08, 0.15, 0.02, 0.25])   # all positive both sides
    assert pbo(train, test) == 0.0


def test_pbo_all_flipped_is_one():
    train = pd.Series([0.1, 0.2, 0.3, 0.4])
    test = pd.Series([-0.1, -0.2, -0.05, -0.3])  # every sign flips
    assert pbo(train, test) == 1.0


def test_pbo_half_flipped_is_one_half():
    train = pd.Series([0.1, 0.2, 0.3, 0.4])
    test = pd.Series([0.1, -0.2, 0.3, -0.4])     # two of four flip
    assert pbo(train, test) == pytest.approx(0.5)


def test_pbo_excludes_nan_and_zero_train():
    train = pd.Series([0.1, np.nan, 0.0, 0.3])   # only splits 0 and 3 count
    test = pd.Series([-0.1, 0.5, 0.5, 0.3])      # split 0 flips, split 3 holds
    assert pbo(train, test) == pytest.approx(0.5)


def test_pbo_empty_is_nan():
    assert np.isnan(pbo(pd.Series([np.nan, np.nan]), pd.Series([0.1, 0.2])))


# --------------------------------------------------- cpcv_factor_eval + pbo (planted)
def _planted_factor(regime, seed=0, n_names=15, T=360, horizon=5, noise=0.005):
    """Build (scores, prices, sleeve_map) whose per-date rank IC follows ``regime``.

    ``regime(frac)`` maps the fraction-through-sample ``t/T`` to a sign in {+1, -1}: the
    score at ``t`` is ``sign * (h-day forward return) + small noise``, so the rank IC on
    horizon ``h`` is ~ +1 (or ~ -1) wherever the regime is +1 (or -1).
    """
    rng = np.random.default_rng(seed)
    ids = [f"EQ:S{i:02d}" for i in range(n_names)]
    sleeve_map = pd.Series({i: "equity" for i in ids})
    dates = pd.bdate_range("2018-01-01", periods=T)
    daily = rng.normal(0.0, 0.02, size=(T, n_names))
    close = 100.0 * np.cumprod(1.0 + daily, axis=0)
    prow, srow = [], []
    for ti, d in enumerate(dates):
        for ni, iid in enumerate(ids):
            prow.append((d, iid, close[ti, ni]))
            if ti + horizon < T:
                fret = close[ti + horizon, ni] / close[ti, ni] - 1.0
                srow.append((d, iid, regime(ti / T) * fret + rng.normal(0.0, noise)))
    prices = pd.DataFrame(prow, columns=["obs_date", "instrument_id", "close"])
    scores = pd.DataFrame(srow, columns=["obs_date", "instrument_id", "value"])
    return scores, prices, sleeve_map


def test_persistent_edge_has_low_pbo_and_same_sign_test_ics():
    scores, prices, sleeve_map = _planted_factor(lambda frac: 1.0)
    ev = cpcv_factor_eval(scores, prices, sleeve_map, horizon_days=5,
                          n_groups=6, k_test=2, purge_days=21, embargo_days=5)
    assert len(ev) == 15                       # one row per split, single sleeve
    # a genuine persistent edge keeps a strongly positive IC on every OOS block
    assert (ev["train_ic"] > 0).all()
    assert (ev["test_ic"] > 0).all()
    assert pbo(ev["train_ic"], ev["test_ic"]) < 0.05


def test_train_only_overfit_factor_has_high_pbo():
    # edge (and its reversal) confined to the sample: predictive in the first 60%,
    # anti-predictive afterwards — the hallmark of an in-sample-fitted factor.
    regime = lambda frac: 1.0 if frac < 0.60 else -1.0
    scores, prices, sleeve_map = _planted_factor(regime)
    ev = cpcv_factor_eval(scores, prices, sleeve_map, horizon_days=5,
                          n_groups=6, k_test=2, purge_days=21, embargo_days=5)
    overfit_pbo = pbo(ev["train_ic"], ev["test_ic"])
    assert overfit_pbo > 0.4                    # many OOS sign flips

    # ... and strictly worse than the persistent factor scored the same way.
    s2, p2, m2 = _planted_factor(lambda frac: 1.0)
    persistent_pbo = pbo(*(lambda e: (e["train_ic"], e["test_ic"]))(
        cpcv_factor_eval(s2, p2, m2, horizon_days=5,
                         n_groups=6, k_test=2, purge_days=21, embargo_days=5)))
    assert overfit_pbo > persistent_pbo
