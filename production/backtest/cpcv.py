"""Combinatorial Purged Cross-Validation (CPCV) and Probability of Backtest Overfitting.

Lineage: Marcos López de Prado, *Advances in Financial Machine Learning* (Wiley, 2018),
Ch. 7 (purged/embargoed K-fold CV) and Ch. 12 (CPCV); and Bailey, Borwein, López de
Prado & Zhu, "The Probability of Backtest Overfitting" (*Journal of Computational
Finance*, 2017) for the ``pbo`` estimator.

Why CPCV rather than a single walk-forward split: a lone train/test cut yields exactly
one out-of-sample path and is trivially over-fit by picking the cut. CPCV partitions the
timeline into ``n_groups`` contiguous date blocks and forms *every* ``C(n_groups,
k_test)`` choice of test blocks, so one factor is scored on many distinct OOS paths.

The two leakage guards are strictly positional on the sorted date index (trading
positions, not calendar days), because forward-return labels overlap in trading time:

- **Purge**: a training observation whose forward-return window overlaps a test block
  would leak the test outcome into training. We drop every training position within
  ``purge_days`` trading positions on *either* side of each contiguous test block.
- **Embargo**: serial correlation lets information bleed *forward* from the test block
  into the immediately following training observations. We additionally drop
  ``embargo_days`` training positions *after* each test block.

Both guards operate on positions so a 21-position purge really removes 21 trading days
(a calendar Timedelta would under-purge — 21 calendar days < 21 trading days). For the
purge to fully cover a forward-return label of ``horizon_days`` the caller should use
``purge_days >= horizon_days`` (the default 21 covers the standard weekly/monthly
horizons in this system).
"""
from __future__ import annotations

import itertools
import math
import warnings

import numpy as np
import pandas as pd

from production.alpha.ic import forward_returns, rank_ic


def cpcv_splits(dates: pd.DatetimeIndex, n_groups: int = 6, k_test: int = 2,
                purge_days: int = 21, embargo_days: int = 5
                ) -> list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    """All combinatorial purged/embargoed train-test date splits (López de Prado Ch. 12).

    The sorted ``dates`` index is partitioned into ``n_groups`` contiguous, near-equal
    blocks. For every combination of ``k_test`` blocks used as the test set (there are
    ``C(n_groups, k_test)`` of them) the remaining blocks form the training set, from
    which we then purge and embargo:

    - test positions are the union of the chosen ``k_test`` blocks;
    - adjacent chosen blocks are merged into contiguous test *runs*;
    - for each test run ``[lo, hi]`` (inclusive positions) every position in
      ``[lo - purge_days, hi + purge_days]`` is removed from training (purge, both
      sides), and every position in ``(hi, hi + embargo_days]`` is additionally removed
      (embargo, forward side only);
    - training is every remaining position.

    Returns a list of ``(train_dates, test_dates)`` ``DatetimeIndex`` pairs. Invariants
    (train/test disjoint, and no training position within the purge/embargo zone of any
    test run) are asserted before each split is emitted.
    """
    dates = pd.DatetimeIndex(dates)
    if not dates.is_monotonic_increasing:
        dates = dates.sort_values()
    n = len(dates)
    if n_groups < 2:
        raise ValueError(f"n_groups must be >= 2, got {n_groups}")
    if not 1 <= k_test < n_groups:
        raise ValueError(f"k_test must be in [1, n_groups), got {k_test}")
    if n < n_groups:
        raise ValueError(f"need at least n_groups={n_groups} dates, got {n}")
    if purge_days < 0 or embargo_days < 0:
        raise ValueError("purge_days and embargo_days must be non-negative")

    # Contiguous, near-equal position blocks. array_split guarantees adjacency, so a run
    # of consecutive group indices is one contiguous position range.
    groups = np.array_split(np.arange(n), n_groups)
    bounds = [(int(g[0]), int(g[-1])) for g in groups]  # inclusive (lo, hi) per group

    splits: list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]] = []
    for combo in itertools.combinations(range(n_groups), k_test):
        test_pos = set()
        for gi in combo:
            test_pos.update(range(bounds[gi][0], bounds[gi][1] + 1))

        # Merge adjacent chosen groups into contiguous test runs.
        runs: list[tuple[int, int]] = []
        run = [combo[0]]
        for gi in combo[1:]:
            if gi == run[-1] + 1:
                run.append(gi)
            else:
                runs.append((bounds[run[0]][0], bounds[run[-1]][1]))
                run = [gi]
        runs.append((bounds[run[0]][0], bounds[run[-1]][1]))

        excluded = set(test_pos)
        for lo, hi in runs:
            excluded.update(range(max(0, lo - purge_days), min(n, hi + purge_days + 1)))
            excluded.update(range(hi + 1, min(n, hi + embargo_days + 1)))

        train_pos = [p for p in range(n) if p not in excluded]

        _assert_invariants(train_pos, test_pos, runs, purge_days, embargo_days, n)

        splits.append((dates[train_pos], dates[sorted(test_pos)]))
    return splits


def _assert_invariants(train_pos: list[int], test_pos: set[int],
                       runs: list[tuple[int, int]], purge_days: int,
                       embargo_days: int, n: int) -> None:
    """Guard rails: train/test disjoint, and no training leak into any purge/embargo zone.

    This is the machine-checked statement of the PIT contract. The forward (embargo) side
    of each test run excludes ``max(purge_days, embargo_days)`` positions; the backward
    side excludes ``purge_days``.
    """
    train_set = set(train_pos)
    assert train_set.isdisjoint(test_pos), "train and test positions overlap"
    fwd = max(purge_days, embargo_days)
    for lo, hi in runs:
        zone = range(max(0, lo - purge_days), min(n - 1, hi + fwd) + 1)
        leak = train_set.intersection(zone)
        assert not leak, f"training positions {sorted(leak)} leak into purge/embargo of {(lo, hi)}"


def _mean_ic_by_sleeve(scores: pd.DataFrame, fwd: pd.DataFrame,
                       sleeve_map: pd.Series, dates: pd.DatetimeIndex) -> pd.Series:
    """Mean per-sleeve rank IC over the given date subset (empty Series if no valid IC)."""
    date_set = pd.DatetimeIndex(dates)
    s = scores[scores["obs_date"].isin(date_set)]
    f = fwd[fwd["obs_date"].isin(date_set)]
    ic = rank_ic(s, f, sleeve_map)
    if ic.empty:
        return pd.Series(dtype=float)
    return ic.groupby("sleeve")["rank_ic"].mean()


def cpcv_factor_eval(scores: pd.DataFrame, prices: pd.DataFrame,
                     sleeve_map: pd.Series, horizon_days: int,
                     **split_kw) -> pd.DataFrame:
    """Score one factor's rank IC on every CPCV train/test split, per sleeve.

    ``scores`` is a long panel ``[obs_date, instrument_id, value]``; the CPCV date grid is
    the sorted set of distinct score dates. Forward returns are computed once over the
    full price history (``production.alpha.ic.forward_returns``) and then sliced to the
    train and test dates of each split. Purging (``purge_days >= horizon_days``) is what
    keeps a training date's forward window from overlapping the test block.

    Returns ``[split, sleeve, train_ic, test_ic]`` — one row per (split, sleeve) for which
    at least one of train/test produced a valid IC; the missing side is ``NaN``. Feed the
    ``train_ic``/``test_ic`` columns to :func:`pbo`.
    """
    purge = split_kw.get("purge_days", 21)
    if purge < horizon_days:
        warnings.warn(
            f"cpcv_factor_eval: purge_days={purge} < horizon_days={horizon_days} — "
            "training forward-return windows can overlap test blocks (leakage)",
            stacklevel=2)
    grid = pd.DatetimeIndex(sorted(pd.unique(scores["obs_date"])))
    splits = cpcv_splits(grid, **split_kw)
    fwd = forward_returns(prices, horizon_days)

    rows: list[tuple] = []
    for si, (train_dates, test_dates) in enumerate(splits):
        train_ic = _mean_ic_by_sleeve(scores, fwd, sleeve_map, train_dates)
        test_ic = _mean_ic_by_sleeve(scores, fwd, sleeve_map, test_dates)
        for sleeve in sorted(set(train_ic.index) | set(test_ic.index)):
            rows.append((si, sleeve,
                         float(train_ic.get(sleeve, np.nan)),
                         float(test_ic.get(sleeve, np.nan))))
    return pd.DataFrame(rows, columns=["split", "sleeve", "train_ic", "test_ic"])


def pbo(train_metric, test_metric) -> float:
    """Probability of Backtest Overfitting — single-strategy sign-consistency variant.

    The canonical CSCV estimator of Bailey et al. (2017) needs a *family* of strategy
    configurations per split: it selects the in-sample winner and measures how often that
    winner lands below the out-of-sample median, i.e. how often ranking is not preserved
    OOS. With a **single** strategy scored per split there is no cross-sectional rank to
    take, so we use the natural degenerate case of the same idea: rank preservation of a
    lone strategy collapses to *sign* preservation of its metric.

    Concretely, given paired per-split metrics (here train and test rank IC), this returns
    the fraction of splits on which the test-metric sign flips relative to the train-metric
    sign::

        PBO = mean_over_splits[ sign(test_metric) != sign(train_metric) ]

    A persistent edge keeps its sign OOS on (almost) every split -> PBO near 0. An edge
    that was fit to the in-sample period reverses or vanishes OOS -> many sign flips ->
    PBO near 1. Pairs with a NaN metric, or with a zero train metric (no in-sample
    position to preserve), are excluded from the denominator; ``NaN`` is returned if no
    pair survives.
    """
    tr = np.asarray(train_metric, dtype=float)
    te = np.asarray(test_metric, dtype=float)
    mask = np.isfinite(tr) & np.isfinite(te) & (np.sign(tr) != 0)
    tr, te = tr[mask], te[mask]
    if len(tr) == 0:
        return float("nan")
    flips = np.sign(te) != np.sign(tr)
    return float(np.mean(flips))


def n_splits(n_groups: int = 6, k_test: int = 2) -> int:
    """Number of CPCV splits: ``C(n_groups, k_test)`` (15 for the default 6-choose-2)."""
    return math.comb(n_groups, k_test)
