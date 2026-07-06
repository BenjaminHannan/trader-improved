"""Signal purification: planted-contamination orthogonality, IC uplift, exclude, fallback.

No production.signals import — the score cross-section and exposure matrix B are built
directly so the test isolates purify_scores' cross-sectional OLS.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from production.alpha.purify import FACTOR_STYLE, exclude_for, purify_scores


def _pearson(a: pd.Series, b: pd.Series) -> float:
    x = a.to_numpy(dtype=float)
    y = b.reindex(a.index).to_numpy(dtype=float)
    return float(np.corrcoef(x, y)[0, 1])


def _setup(seed: int = 11, n: int = 50):
    rng = np.random.default_rng(seed)
    ids = [f"EQ:S{i:02d}" for i in range(n)]
    B = pd.DataFrame(
        {"market": rng.normal(size=n),
         "size": rng.normal(size=n),
         "vol": rng.normal(size=n)},
        index=ids,
    )
    true_alpha = pd.Series(rng.normal(size=n), index=ids)
    return rng, ids, B, true_alpha


# ============================================================ orthogonality
def test_purify_strips_planted_market_contamination():
    """z = 0.6*alpha + 0.8*market: raw is strongly correlated with market; the purified
    residual is orthogonal (|corr| < 0.05) to EVERY used exposure column."""
    _rng, _ids, B, true_alpha = _setup()
    z_raw = 0.6 * true_alpha + 0.8 * B["market"]

    purified = purify_scores(z_raw, B, exclude=[])

    assert abs(_pearson(z_raw, B["market"])) > 0.5          # planted contamination present
    for c in B.columns:                                     # OLS residual orthogonal to all
        assert abs(_pearson(purified, B[c])) < 0.05, c


def test_purify_restandardizes_mean_zero_std_one():
    _rng, _ids, B, true_alpha = _setup()
    z_raw = 0.6 * true_alpha + 0.8 * B["market"]
    purified = purify_scores(z_raw, B, exclude=[])
    assert abs(purified.mean()) < 1e-9
    assert abs(purified.std(ddof=0) - 1.0) < 1e-9


# ============================================================ IC uplift
def test_purify_ic_beats_raw_against_beta_neutral_returns():
    """Forward returns = alpha + beta*market_shock + noise; beta-neutralized for scoring.
    The purified score (market stripped) ranks the beta-neutral returns strictly better
    than the market-contaminated raw score."""
    rng, ids, B, true_alpha = _setup(seed=7)
    z_raw = 0.6 * true_alpha + 0.8 * B["market"]
    purified = purify_scores(z_raw, B, exclude=[])

    market_shock = 1.5
    noise = pd.Series(rng.normal(scale=0.5, size=len(ids)), index=ids)
    fwd_raw = true_alpha + B["market"] * market_shock + noise
    fwd_neutral = fwd_raw - B["market"] * market_shock       # strip the paid-for market move

    ic_raw = spearmanr(z_raw.to_numpy(), fwd_neutral.to_numpy())[0]
    ic_pure = spearmanr(purified.reindex(ids).to_numpy(), fwd_neutral.to_numpy())[0]
    assert ic_pure > ic_raw
    # report handle: the measured uplift on this planted case
    globals()["_IC_UPLIFT"] = (ic_raw, ic_pure)


# ============================================================ exclude respected
def test_purify_excluded_style_survives():
    """A momentum-family signal excludes the 'momentum' column: its own style must NOT be
    regressed out — the purified score stays correlated with momentum."""
    rng, ids, _B, true_alpha = _setup(seed=3)
    B = pd.DataFrame(
        {"market": rng.normal(size=len(ids)),
         "momentum": rng.normal(size=len(ids)),
         "vol": rng.normal(size=len(ids))},
        index=ids,
    )
    z_raw = 0.6 * true_alpha + 0.8 * B["momentum"]

    purified = purify_scores(z_raw, B, exclude=["momentum"])
    # excluded style survives (still strongly correlated), other columns stripped to ~0.
    assert abs(_pearson(purified, B["momentum"])) > 0.5
    assert abs(_pearson(purified, B["market"])) < 0.05
    assert abs(_pearson(purified, B["vol"])) < 0.05

    # contrast: NOT excluding momentum strips it out entirely.
    stripped = purify_scores(z_raw, B, exclude=[])
    assert abs(_pearson(stripped, B["momentum"])) < 0.05


# ============================================================ small-sample fallback
def test_purify_small_sample_returns_input_unchanged():
    """N < K_used + min_excess -> documented fallback: input returned bit-identical."""
    ids = [f"EQ:S{i}" for i in range(6)]            # N = 6
    B = pd.DataFrame(np.random.default_rng(1).normal(size=(6, 4)),
                     index=ids, columns=["market", "size", "momentum", "vol"])  # K_used = 4
    z = pd.Series(np.arange(6, dtype=float), index=ids)
    out = purify_scores(z, B, exclude=[], min_excess=3)   # 6 < 4 + 3 -> fallback
    pd.testing.assert_series_equal(out, z)


def test_purify_no_columns_to_neutralize_returns_input():
    """Excluding every column leaves nothing to regress on -> input unchanged."""
    ids = [f"EQ:S{i}" for i in range(20)]
    B = pd.DataFrame(np.random.default_rng(2).normal(size=(20, 2)),
                     index=ids, columns=["market", "vol"])
    z = pd.Series(np.random.default_rng(3).normal(size=20), index=ids)
    out = purify_scores(z, B, exclude=["market", "vol"])
    pd.testing.assert_series_equal(out, z)


# ============================================================ style map
def test_factor_style_map_and_exclude_helper():
    assert FACTOR_STYLE["mom_12_1"] == "momentum"
    assert FACTOR_STYLE["low_vol"] == "vol"
    assert FACTOR_STYLE["carry_funding"] is None
    assert exclude_for("mom_12_1") == ["momentum"]
    assert exclude_for("tsmom") == ["momentum"]
    assert exclude_for("str_reversal_1m") == ["momentum"]
    assert exclude_for("low_vol") == ["vol"]
    assert exclude_for("carry_funding") == []          # neutralize against all of B
    assert exclude_for("unknown_factor") == []
