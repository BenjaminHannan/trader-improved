"""The risk-model scoring harness — bias statistics + MVP horse race.

WHY: research/wiki/questions/research-risk-model-validation.md Q1 ("the scoring
harness") — the "build once, use forever" instrument for adjudicating every future
risk-model change (Q2-Q5 in that page are questions the harness itself answers
later; this module implements none of them, only the harness). Likelihood and
Frobenius error are NOT the accepted way to score a risk model used for portfolio
construction — bias statistics on families of test portfolios, plus a
minimum-variance realized-vol horse race, are (Menchero-Orr-Wang USE4;
Clarke-de Silva-Thorley JPM 2006/2011; Ledoit & Wolf, J. Empirical Finance 2011).

Risk-model changes carry ZERO n_trials cost (no alpha selection is touched by
anything in this module) but are logged with the same discipline as a factor gate
verdict — into a SEPARATE ledger (``diagnostics/risk_harness/``, not
``configs/factors.yaml``).

PIT contract (CLAUDE.md: every rolling statistic is trailing-only and shift(1)-ed
where it feeds a same-day decision). Both entry points that walk a panel over time
— :func:`bias_stats` and :func:`mvp_horse_race` — enforce this THEMSELVES rather
than trusting the caller: at evaluation date ``t`` they slice
``returns_panel.iloc[:pos]`` (strictly ``obs_date < t``, ``pos`` excluded) before
handing that window to the caller-supplied ``sigma_fn``. A Sigma built from that
window cannot see day ``t`` or later — this IS the shift(1) semantics the hard
rules require, enforced by construction instead of by convention. No further
``shift(1)`` is applied on top: unlike the ADV-weight pattern in
``production/risk/exposures.py`` (which shifts because its trailing window is
INCLUSIVE of day ``t``), Sigma_t here is already exclusive of ``t`` by the slice,
so an extra lag would just waste a day of information.

Bias-statistic sampling design (a deliberate reading of the wiki spec, documented
here because the wiki's phrasing admits two readings): the core loop scores
NON-OVERLAPPING h=21-day windows so z_t are approximately iid across TIME
(T ~= 2600/21 ~= 120 per the wiki). Pooling many test portfolios evaluated at the
SAME window against the SAME Sigma_t and the SAME realized market move would NOT
add T independent observations per portfolio -- they are correlated through the
shared forecast and the shared realized path. So for the "many random draws"
families (1 long-only, 2 dollar-neutral) :func:`bias_stats` consumes exactly ONE
portfolio per evaluation window, cycling through the pre-built batch from
:func:`build_test_portfolios` -- T ends up counted on the time axis, matching the
wiki's T~=120, not batch_size x windows. Families with only one or two PERSISTENT
portfolio definitions (3 equal-weight, 4 MVP variants) contribute one z per
variant per window, which is what "MVP w proportional to Sigma^{-1} 1 ... ALSO the
horse-race portfolio" in the spec literally describes: a portfolio that is itself
a function of Sigma_t, recomputed every window.

Families 5 (pure-factor books) and 6 (cross-sleeve hedged books) need the live
factor model / a cross-sleeve instrument map that does not exist as a clean
callable yet (see the CLI's docstring and the final report of the build for what
specifically is missing in ``production/risk/model.py``). :func:`build_test_portfolios`
implements their INTERFACE only -- pass pre-built weights via ``external`` -- and
returns an empty frame (never raises, never silently fabricates data) when they are
absent, so the harness degrades gracefully instead of crashing when those two
families are simply not available yet.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from production.backtest.bootstrap import politis_white_block_length
from production.core.config import REPO_ROOT

_TRADING_DAYS = 252
_HORIZON_DAYS = 21          # h — the forecast/realization horizon (Q1 spec)
_EVAL_STEP_DAYS = 5         # MVP horse-race rebalance cadence (a continuous book,
                            # no non-overlap constraint — see module docstring)
_MRAD_WINDOW = 12           # USE4 "mean rolling |B-1|" window, in z observations
_BASELINE_MIN_FRACTION = 0.90

SigmaFn = Callable[[pd.DataFrame], pd.DataFrame]
_LEDGER_SUBDIR = "diagnostics/risk_harness"   # committed (not gitignored) — see CLAUDE.md


class RiskValidationError(Exception):
    """Harness misuse: unknown family, a required input missing, ..."""


# ------------------------------------------------------------- MVP primitive
def _mvp_weights(sigma: pd.DataFrame, long_only: bool = False) -> pd.Series | None:
    """``w ∝ Sigma^{-1} 1``, normalized to sum to 1 — the classical unconstrained MVP.

    ``long_only=True`` clips negative raw weights to zero and renormalizes (family
    4's long-only-constrained variant). Returns ``None`` when the raw solve is
    degenerate (a long-only clip that zeroes everything, or a non-finite sum) —
    callers must treat that as "no portfolio for this Sigma", not raise.
    """
    ids = sigma.index
    S = sigma.to_numpy(dtype=float)
    ones = np.ones(len(ids))
    try:
        raw = np.linalg.solve(S, ones)
    except np.linalg.LinAlgError:
        raw = np.linalg.lstsq(S, ones, rcond=None)[0]
    if long_only:
        raw = np.clip(raw, 0.0, None)
    total = float(raw.sum())
    if not np.isfinite(total) or abs(total) < 1e-12:
        return None
    return pd.Series(raw / total, index=ids)


# ------------------------------------------------------------ test portfolios
def build_test_portfolios(returns_panel: pd.DataFrame, sleeve: str, family: int,
                          rng: np.random.Generator, n: int = 100,
                          sigma: pd.DataFrame | None = None,
                          external: pd.DataFrame | None = None) -> pd.DataFrame:
    """Build one family's test-portfolio weight batch: rows=portfolio label, cols=ids.

    Family 1 — random long-only Dirichlet(1) draws (``n`` rows).
    Family 2 — random dollar-neutral long/short: a random long/short split per
    draw, Dirichlet(1) magnitudes on each side, so ``sum(w)=0`` and
    ``sum(|w|)=2`` exactly.
    Family 3 — the equal-weight sleeve book (1 row).
    Family 4 — MVP ``w ∝ Sigma^{-1} 1`` plus the long-only-clipped renormalized
    variant (needs ``sigma``; up to 2 rows — fewer if a variant is degenerate).
    Families 5/6 — INTERFACE ONLY: pure-factor books (5) and cross-sleeve hedged
    books (6) need the live factor model / a cross-sleeve instrument map this
    build does not have a clean callable for yet. Pass pre-built weights via
    ``external``; absent that, returns an empty (0-row) frame — a documented
    not-computed-yet fallback, never a crash.
    """
    ids = list(returns_panel.columns)
    if family == 1:
        draws = rng.dirichlet(np.ones(len(ids)), size=n)
        return pd.DataFrame(draws, columns=ids,
                            index=[f"{sleeve}_f1_{i:03d}" for i in range(n)])

    if family == 2:
        rows: dict[str, np.ndarray] = {}
        for i in range(n):
            # At least one long AND one short name, else it's not dollar-neutral.
            while True:
                long_side = rng.integers(0, 2, size=len(ids)).astype(bool)
                if long_side.any() and (~long_side).any():
                    break
            w = np.zeros(len(ids))
            w[long_side] = rng.dirichlet(np.ones(int(long_side.sum())))
            w[~long_side] = -rng.dirichlet(np.ones(int((~long_side).sum())))
            rows[f"{sleeve}_f2_{i:03d}"] = w
        return pd.DataFrame(rows, index=ids).T.reindex(columns=ids)

    if family == 3:
        w = np.full(len(ids), 1.0 / len(ids))
        return pd.DataFrame([w], columns=ids, index=[f"{sleeve}_f3_equal_weight"])

    if family == 4:
        if sigma is None:
            raise RiskValidationError(f"family 4 (MVP) requires `sigma` ({sleeve!r})")
        Sig = sigma.reindex(index=ids, columns=ids)
        if Sig.isna().any().any():
            raise RiskValidationError(
                f"family 4 (MVP): sigma does not cover every id for sleeve {sleeve!r}")
        rows = {}
        w_mvp = _mvp_weights(Sig)
        if w_mvp is not None:
            rows[f"{sleeve}_f4_mvp"] = w_mvp.reindex(ids).to_numpy()
        w_lo = _mvp_weights(Sig, long_only=True)
        if w_lo is not None:
            rows[f"{sleeve}_f4_mvp_long_only"] = w_lo.reindex(ids).to_numpy()
        if not rows:
            return pd.DataFrame(columns=ids)
        return pd.DataFrame(rows, index=ids).T.reindex(columns=ids)

    if family in (5, 6):
        if external is None:
            # NOT COMPUTED YET (see module + CLI docstrings) — degrade gracefully.
            return pd.DataFrame(columns=ids)
        return external.reindex(columns=ids).fillna(0.0)

    raise RiskValidationError(f"unknown test-portfolio family: {family!r}")


# ------------------------------------------------------------------ bias stats
def evaluation_dates(index: pd.DatetimeIndex, h: int = _HORIZON_DAYS,
                     min_obs: int = 252) -> pd.DatetimeIndex:
    """Non-overlapping evaluation dates for the bias-statistic z-sample.

    A pure stride-``h`` grid starting once ``min_obs`` trailing observations
    exist: consecutive kept dates are exactly ``h`` positions apart, so the
    ``[pos, pos+h)`` realization windows used downstream never overlap by
    construction. ``T ~= (len(index) - min_obs) / h`` — matches the wiki's
    "T ~= 2600/21 ~= 120 obs per cell" note.
    """
    n = len(index)
    if n < min_obs + h:
        return index[:0]
    positions = np.arange(min_obs, n - h + 1, h)
    return index[positions]


def _mrad(z: np.ndarray, window: int = _MRAD_WINDOW) -> float:
    """Mean rolling |B-1| over ``window``-obs windows of the (time-ordered) z sample."""
    if len(z) < window:
        return float("nan")
    roll_b = pd.Series(z).rolling(window).std(ddof=0)
    return float((roll_b - 1.0).abs().mean())


def _empty_cell(n_dates: int, n_portfolios: int = 0, skipped: int = 0) -> dict:
    return {"B": float("nan"), "band": (float("nan"), float("nan")), "in_band": False,
            "T": 0, "MRAD": float("nan"), "n_dates": n_dates,
            "n_portfolios": n_portfolios, "skipped_dates": skipped}


def bias_stats(returns_panel: pd.DataFrame, sigma_fn: SigmaFn,
               weights: pd.DataFrame | Callable[[pd.DataFrame], pd.DataFrame],
               h: int = _HORIZON_DAYS, min_obs: int = 252,
               mrad_window: int = _MRAD_WINDOW) -> dict:
    """The Q1 core loop for one (sleeve x family) cell.

    At each non-overlapping evaluation date t (:func:`evaluation_dates`):
    forecast ``sigma_hat_t(h) = sqrt(h * w' Sigma_t w)`` where ``Sigma_t =
    sigma_fn(returns_panel.iloc[:pos])`` — STRICTLY the rows before t (the PIT
    enforcement; see module docstring); realize ``r_{t->t+h} = sum`` of the h
    daily returns starting at t; ``z = r / sigma_hat``.

    ``weights`` is either:
      - a static (P x N) DataFrame from :func:`build_test_portfolios` (families
        1/2/3, or externally-supplied 5/6): exactly ONE row is scored per
        evaluation window, cycling through the P rows in order
        (``row = window_index % P``) — see module docstring for why pooling all
        P rows at every window is NOT done (it would not add independent
        observations);
      - a callable ``Sigma_t -> (P x N) DataFrame`` for portfolios that are
        themselves a function of the forecast (family 4's MVP / long-only-MVP,
        typically P=2) — ALL rows it returns are scored every window, since
        these are named, persistent portfolio definitions, not interchangeable
        random draws.

    Returns ``{B, band, in_band, T, MRAD, n_dates, n_portfolios, skipped_dates}``.
    A ``sigma_fn`` failure (e.g. too few observations) at some date is skipped,
    not raised — that date just does not contribute a z (degrade gracefully).
    """
    ids = list(returns_panel.columns)
    index = returns_panel.index
    dates = evaluation_dates(index, h=h, min_obs=min_obs)
    R = returns_panel.to_numpy(dtype=float)

    is_dynamic = callable(weights)
    static_W = None
    if not is_dynamic:
        static_W = weights.reindex(columns=ids).fillna(0.0)
        if static_W.empty:
            return _empty_cell(len(dates))

    z_rows: list[np.ndarray] = []
    skipped = 0
    n_portfolios = 0
    for window_idx, t in enumerate(dates):
        pos = index.get_loc(t)
        window = returns_panel.iloc[:pos]           # STRICTLY before t — the PIT contract.
        try:
            Sigma = sigma_fn(window)
        except Exception:                            # noqa: BLE001 - a failing forecast
            skipped += 1                              # for this date is data, not a crash.
            continue
        Sig = Sigma.reindex(index=ids, columns=ids)
        if Sig.isna().any().any():
            skipped += 1
            continue

        if is_dynamic:
            W = weights(Sig)
            if W is None or W.empty:
                skipped += 1
                continue
            W = W.reindex(columns=ids).fillna(0.0)
        else:
            row = window_idx % static_W.shape[0]
            W = static_W.iloc[[row]]
        n_portfolios = max(n_portfolios, W.shape[0])

        Wm = W.to_numpy(dtype=float)
        Sig_np = Sig.to_numpy(dtype=float)
        var_p = np.einsum("pi,ij,pj->p", Wm, Sig_np, Wm)
        sigma_hat = np.sqrt(np.clip(h * var_p, 0.0, None))
        realized = R[pos: pos + h, :].sum(axis=0)     # h daily returns starting AT t
        r_p = Wm @ realized
        with np.errstate(invalid="ignore", divide="ignore"):
            z = np.where(sigma_hat > 0, r_p / sigma_hat, np.nan)
        z_rows.append(z)

    if not z_rows:
        return _empty_cell(len(dates), n_portfolios, skipped)

    z = np.concatenate(z_rows)
    z = z[np.isfinite(z)]
    T = int(z.size)
    if T < 2:
        return _empty_cell(len(dates), n_portfolios, skipped)

    B = float(np.std(z, ddof=0))
    band = (1.0 - np.sqrt(2.0 / T), 1.0 + np.sqrt(2.0 / T))
    in_band = bool(band[0] <= B <= band[1])
    return {"B": B, "band": band, "in_band": in_band, "T": T,
            "MRAD": _mrad(z, window=mrad_window),
            "n_dates": len(dates) - skipped, "n_portfolios": n_portfolios,
            "skipped_dates": skipped}


# --------------------------------------------------------------- MVP horse race
def _paired_stationary_resample(n: int, block: float, rng: np.random.Generator) -> np.ndarray:
    """Geometric-block circular index draw — the same scheme as
    ``production.backtest.bootstrap.stationary_bootstrap`` — returned as raw
    indices (instead of a reduced Sharpe) so the identical resample can be
    applied to BOTH the candidate and incumbent MVP return paths at once,
    preserving their shared-market-day correlation. Only this short index loop
    is duplicated; the block-length SELECTION (the hard part) is reused as-is
    from :func:`production.backtest.bootstrap.politis_white_block_length`.
    """
    idx = np.empty(n, dtype=int)
    i = int(rng.integers(0, n))
    p = 1.0 / max(block, 1.0)
    for k in range(n):
        idx[k] = i
        i = int(rng.integers(0, n)) if rng.random() < p else (i + 1) % n
    return idx


def lw2011_equal_variance_test(r_candidate, r_incumbent, n_boot: int = 1000,
                               seed: int = 0) -> dict:
    """Two-sided stationary-bootstrap test of H0: Var(r_candidate) == Var(r_incumbent).

    A documented, practical form of the Ledoit & Wolf (2011, J. Empirical
    Finance) robust-variance test: the block length is selected data-drivenly
    from the *differential* series via
    :func:`production.backtest.bootstrap.politis_white_block_length` (so serial
    dependence in either return path sets the block size), the paired series
    are block-resampled together (:func:`_paired_stationary_resample`), the
    variance difference is recomputed on each resample, and a two-sided
    p-value is read off how extreme the observed difference is against the
    resample distribution RECENTERED at zero (the standard bootstrap device
    for testing a point null against an empirical, not null, resampling DGP).
    This is a simplification of LW-2011's exact studentized statistic (no
    closed-form asymptotic variance-of-the-variance is fit) — it reuses the
    same block-bootstrap machinery this repo already trusts for the Sharpe CI.
    A `+1` continuity correction keeps ``p`` from ever reading exactly 0.
    """
    joint = pd.concat([pd.Series(r_candidate).astype(float),
                       pd.Series(r_incumbent).astype(float)],
                      axis=1, join="inner").dropna()
    n = len(joint)
    if n < 4:
        return {"observed_diff": float("nan"), "p_value": float("nan"),
                "block_length": float("nan"), "n_boot": n_boot, "n": n}

    rc = joint.iloc[:, 0].to_numpy()
    ri = joint.iloc[:, 1].to_numpy()
    block = politis_white_block_length(rc - ri)
    obs_diff = float(np.var(rc, ddof=0) - np.var(ri, ddof=0))

    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = _paired_stationary_resample(n, block, rng)
        boots[b] = np.var(rc[idx], ddof=0) - np.var(ri[idx], ddof=0)
    centered = boots - boots.mean()
    count = int(np.sum(np.abs(centered) >= abs(obs_diff)))
    p_value = (count + 1) / (n_boot + 1)
    return {"observed_diff": obs_diff, "p_value": p_value, "block_length": block,
            "n_boot": n_boot, "n": n}


def _mvp_return_path(sigma_fn: SigmaFn, returns_panel: pd.DataFrame, step: int,
                     min_obs: int, long_only: bool) -> pd.Series:
    """Walk-forward MVP daily-return path: rebalance every ``step`` business days
    using ``Sigma_t = sigma_fn(returns strictly before t)`` (same PIT contract
    as :func:`bias_stats`), hold the resulting weights until the next
    rebalance. No extra ``shift(1)`` beyond the strict ``< t`` slice — see the
    module docstring.
    """
    index = returns_panel.index
    n = len(index)
    ids = list(returns_panel.columns)
    rows: dict[pd.Timestamp, np.ndarray] = {}
    for pos in np.arange(min_obs, n, step):
        t = index[pos]
        window = returns_panel.iloc[:pos]
        try:
            Sigma = sigma_fn(window)
        except Exception:                            # noqa: BLE001 - see bias_stats
            continue
        Sig = Sigma.reindex(index=ids, columns=ids)
        if Sig.isna().any().any():
            continue
        w = _mvp_weights(Sig, long_only=long_only)
        if w is not None:
            rows[t] = w.reindex(ids).to_numpy()

    if not rows:
        raise RiskValidationError(
            "mvp_horse_race: sigma_fn produced no usable MVP weights over the "
            "panel (insufficient trailing history for every rebalance date?)")

    W = pd.DataFrame(rows, index=ids).T.reindex(columns=ids)
    W_daily = W.reindex(index, method="ffill").fillna(0.0)
    port_ret = (returns_panel.reindex(columns=ids).fillna(0.0) * W_daily).sum(axis=1)
    return port_ret.loc[port_ret.index >= W.index[0]]  # drop the pre-first-rebalance prefix


def mvp_horse_race(candidate_sigma_fn: SigmaFn, incumbent_sigma_fn: SigmaFn,
                   returns_panel: pd.DataFrame, step: int = _EVAL_STEP_DAYS,
                   min_obs: int = 60, long_only: bool = False,
                   n_boot: int = 1000, seed: int = 0) -> dict:
    """Realized-vol horse race of the family-4 MVP book under two Sigma models
    (Q1 pass criterion 3): walk both candidate and incumbent forward
    (:func:`_mvp_return_path`), compare realized annualized vol, and run the
    LW-2011 stationary-bootstrap equal-variance test
    (:func:`lw2011_equal_variance_test`) on the two return paths.
    """
    r_candidate = _mvp_return_path(candidate_sigma_fn, returns_panel, step, min_obs, long_only)
    r_incumbent = _mvp_return_path(incumbent_sigma_fn, returns_panel, step, min_obs, long_only)
    vol_candidate = float(r_candidate.std(ddof=0) * np.sqrt(_TRADING_DAYS))
    vol_incumbent = float(r_incumbent.std(ddof=0) * np.sqrt(_TRADING_DAYS))
    test = lw2011_equal_variance_test(r_candidate, r_incumbent, n_boot=n_boot, seed=seed)
    return {
        "candidate_ann_vol": vol_candidate,
        "incumbent_ann_vol": vol_incumbent,
        "candidate_lower": vol_candidate < vol_incumbent,
        "p_value": test["p_value"],
        "observed_var_diff": test["observed_diff"],
        "block_length": test["block_length"],
        "n_obs": test["n"],
        "n_boot": n_boot,
    }


# --------------------------------------------------------------------- verdicts
def baseline_health_check(cells: dict) -> dict:
    """The one-time sanity check run at harness birth (Q1 spec): >= 90% of
    random-book (families 1 and 2) cells must land inside the acceptance band.
    ``cells``: ``{(sleeve, family): bias_stats(...) result}``.
    """
    random_cells = {k: v for k, v in cells.items() if k[1] in (1, 2)}
    total = len(random_cells)
    in_band = sum(1 for v in random_cells.values() if v.get("in_band"))
    frac = (in_band / total) if total else float("nan")
    failing = sorted(k for k, v in random_cells.items() if not v.get("in_band"))
    return {"passed": bool(total and frac >= _BASELINE_MIN_FRACTION),
            "fraction_in_band": frac, "n_cells": total, "failing_cells": failing}


def adoption_verdict(incumbent_cells: dict, candidate_cells: dict, horse_race: dict,
                     pit_harness_green: bool, candidate_is_simpler: bool = False) -> dict:
    """The 4 pass criteria for adopting a risk-model change, verbatim from
    research/wiki/questions/research-risk-model-validation.md Q1:

    1. no NEW calibration failures (every cell in-band under the incumbent
       stays in-band under the candidate);
    2. |B-1| improves weakly in >= 60% of (common) cells;
    3. MVP realized vol: candidate <= incumbent with LW-2011 p < 0.05 for a
       claimed improvement (p >= 0.05 -> adopt only if strictly simpler);
    4. the PIT corruption harness (tests/test_no_lookahead.py) is still green.

    Criterion 4 is NOT re-run here — it is a pytest suite, not a stat this
    module can compute — so it is asserted externally via
    ``pit_harness_green``. ``candidate_is_simpler`` is likewise a judgment call
    outside this module's scope (criterion 3's escape clause).

    ``incumbent_cells`` / ``candidate_cells``: ``{(sleeve, family):
    bias_stats(...) result}``, compared on their common keys.
    """
    common = sorted(set(incumbent_cells) & set(candidate_cells))
    reasons: list[str] = []

    new_failures = [k for k in common
                    if incumbent_cells[k].get("in_band", False)
                    and not candidate_cells[k].get("in_band", False)]
    c1 = len(new_failures) == 0
    if not c1:
        reasons.append(f"criterion 1 failed: new calibration failures in {new_failures}")

    improved = 0
    n_comparable = 0
    for k in common:
        b_in, b_cand = incumbent_cells[k].get("B"), candidate_cells[k].get("B")
        if b_in is None or b_cand is None or not (np.isfinite(b_in) and np.isfinite(b_cand)):
            continue
        n_comparable += 1
        if abs(b_cand - 1.0) <= abs(b_in - 1.0) + 1e-12:
            improved += 1
    frac_improved = (improved / n_comparable) if n_comparable else 0.0
    c2 = frac_improved >= 0.60
    if not c2:
        reasons.append(f"criterion 2 failed: only {frac_improved:.0%} of cells "
                       f"improved |B-1| (need >= 60%)")

    p = horse_race.get("p_value", float("nan"))
    cand_lower = bool(horse_race.get("candidate_lower", False))
    if cand_lower and np.isfinite(p) and p < 0.05:
        c3 = True
    elif candidate_is_simpler:
        c3 = True
        reasons.append("criterion 3: horse race inconclusive (p>=0.05 or vol not "
                       "lower) but candidate is claimed strictly simpler — adopt anyway")
    else:
        c3 = False
        reasons.append(f"criterion 3 failed: candidate_lower={cand_lower}, "
                       f"p={p!r} (need <0.05), and candidate is not claimed simpler")

    c4 = bool(pit_harness_green)
    if not c4:
        reasons.append("criterion 4 failed: PIT corruption harness not green "
                       "(tests/test_no_lookahead.py)")

    passed = c1 and c2 and c3 and c4
    if passed:
        reasons.append("all 4 criteria passed")
    return {"passed": passed, "reasons": reasons,
            "criteria": {"no_new_failures": c1, "frac_improved": frac_improved,
                        "improved_ge_60pct": c2, "mvp_horse_race": c3,
                        "pit_harness_green": c4},
            "new_failures": new_failures}


# ---------------------------------------------------------------------- ledger
def _coerce(o):
    """JSON-safe coercion (numpy scalars, NaN/inf -> None, tuples -> lists) —
    same discipline as production/backtest/report.py's report writer."""
    if isinstance(o, dict):
        return {str(k): _coerce(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_coerce(v) for v in o]
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return None if not np.isfinite(f) else f
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    return o


def _stringify_cells(cells: dict | None) -> dict | None:
    """``{(sleeve, family): result}`` -> ``{"sleeve/family<N>": result}`` (JSON keys
    must be strings; tuple keys are how every cell dict in this module is indexed)."""
    if cells is None:
        return None
    return {f"{k[0]}/family{k[1]}": _coerce(v) for k, v in cells.items()}


def build_ledger_record(*, run_id: str | None = None, incumbent_cells: dict,
                        candidate_cells: dict | None = None,
                        horse_race: dict | None = None, verdict: dict | None = None,
                        baseline_health: dict | None = None,
                        notes: list[str] | None = None) -> dict:
    """Plain-dict ledger record for one harness run.

    Logged like a factor-gate verdict but in a SEPARATE ledger — risk-model
    changes carry zero n_trials cost (CLAUDE.md; no alpha selection is
    touched), so this never writes to ``configs/factors.yaml``.
    """
    ts = datetime.now(timezone.utc)
    return {
        "run_id": run_id or ts.strftime("%Y%m%dT%H%M%SZ"),
        "generated_at": ts.isoformat(),
        "incumbent_cells": _stringify_cells(incumbent_cells),
        "candidate_cells": _stringify_cells(candidate_cells),
        "horse_race": _coerce(horse_race) if horse_race is not None else None,
        "verdict": _coerce(verdict) if verdict is not None else None,
        "baseline_health": _coerce(baseline_health) if baseline_health is not None else None,
        "notes": list(notes) if notes else [],
    }


def write_ledger(record: dict, out_dir: str | Path = _LEDGER_SUBDIR) -> Path:
    """Write ``risk_harness_<run_id>.json`` into ``diagnostics/risk_harness/``
    (committed — not the gitignored ``data/`` or ``reports/`` — see CLAUDE.md's
    "no report files in data/" rule, which this directory does not violate)."""
    out = Path(out_dir)
    if not out.is_absolute():
        out = REPO_ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    run_id = record.get("run_id") or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = out / f"risk_harness_{run_id}.json"
    with open(path, "w") as f:
        json.dump(record, f, indent=2, allow_nan=False)
    return path
