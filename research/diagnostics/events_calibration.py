"""Backlog #21 — events-sleeve calibration curve (market price -> realized
probability), fit from RESOLVED Kalshi outcomes with a pre-registered
out-of-sample adoption test.

Closes the iteration-25 documented approximation ("signal scores pass through
as believed edge; calibration is the first real-data improvement") and is the
prerequisite signal model for the maker-side execution idea (#19).

=============================== PRE-REGISTRATION (verbatim) ================================
Written 2026-07-11 BEFORE the calibration was first fitted. Data: curated
event_markets_hist (12 US macro series, 2021-07 -> 2026-07): row_type='bar' daily
trade bars and row_type='settlement' outcomes (yes_price 1.0/0.0). Universe: binary
settled markets with life volume >= 500 contracts; entry price = last bar yes_price
on a day in [close-10d, close-1d] (the T-1 convention of the 2026-07-10 longshot
diagnostic). Category caveat: this panel is macro-economics only; the fitted curve
is tagged category='economics' and any production consumption must fall back to
identity for other categories.

Fit (train = settlements before 2025-07-01):
  g = isotonic regression of outcome on entry price (monotone non-decreasing,
  pool-adjacent-violators), plus per-decile Wilson 95% intervals as the
  uncertainty record. No other features in the primary fit.

Pre-registered OOS adoption test (test = settlements on/after 2025-07-01):
  A1: Brier(g(price)) < Brier(price) on the test set, with a paired one-sided
      cluster-bootstrap p < 0.05 (resample event_key clusters, 2000 draws,
      fixed seed 7).
  A2 (sanity): g is not degenerate — at least 4 distinct fitted values across
      the price deciles on train.
  PASS = A1 & A2 -> propose consumption in events sizing (execution layer, no
  n_trials); FAIL -> file the curve as a diagnostic record only.
  Any spec change after seeing results is a NEW pre-registration.
=============================================================================================

Entries frame: reuses kalshi_longshot_fade.load_panel() verbatim (identical
curated-lake read of event_markets_hist -> row_type in {bar, settlement},
settlement rows deduped keep-last per instrument_id) and REPLICATES its
build_entries() below with the life-volume floor changed from 1,000 to 500
contracts per this backlog's spec. Everything else in build_entries — the T-1
entry-window logic ([close-10d, close-1d], last bar in that window), the
close_date derivation, the output columns — is unchanged from
kalshi_longshot_fade.build_entries.

Isotonic fit: pool-adjacent-violators implemented directly in numpy (no sklearn
dependency in this repo). Duplicate entry prices are aggregated to a weighted
mean outcome first (weight = count) — this is the standard reduction and gives
an identical solution to running PAV on the raw disaggregated pairs, since the
L2 isotonic-regression optimum only depends on the multiset of (x, y) through
the within-x weighted means. The fitted curve is stored as pooled blocks
(xmin, xmax, value); prediction is a right-continuous step function keyed on
each block's upper edge (xmax): g(p) = value[i] where i is the first block with
xmax[i] >= p. Extrapolation: p below the lowest train price maps to the first
block's value; p above the highest train price maps to the last block's value
(both are the pre-registered "clamp to the boundary fitted value" rule).

Usage: uv run python research/diagnostics/events_calibration.py
Artifact: diagnostics/events_calibration.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "research" / "diagnostics"))

from kalshi_longshot_fade import load_panel  # noqa: E402

MIN_LIFE_VOLUME = 500.0                       # this backlog's floor (vs 1,000 in #1)
ENTRY_WINDOW_DAYS = 10                        # [close-10d, close-1d], same as #1
TRAIN_TEST_CUTOFF = pd.Timestamp("2025-07-01")
N_DECILES = 10
N_BOOT = 2000
BOOT_SEED = 7
CATEGORY = "economics"
WILSON_Z = 1.959963984540054                  # 95% two-sided


def build_entries(bars: pd.DataFrame, st: pd.DataFrame,
                   min_life_volume: float = MIN_LIFE_VOLUME,
                   window_days: int = ENTRY_WINDOW_DAYS) -> pd.DataFrame:
    """One row per market: T-1 entry price, outcome, metadata.

    Replicated from kalshi_longshot_fade.build_entries with the life-volume
    floor parameterized (500 here vs 1,000 there); all other logic identical.
    """
    st = st.set_index("instrument_id")
    close_date = (pd.to_datetime(st["close_time"], utc=True, format="ISO8601")
                  .dt.tz_localize(None).dt.normalize())
    life_vol = bars.groupby("instrument_id")["volume"].sum()

    rows = []
    for iid, g in bars.groupby("instrument_id"):
        if iid not in st.index:
            continue
        cd = close_date.get(iid)
        if pd.isna(cd):
            continue
        pre = g[(g["obs_date"] < cd)
                & (g["obs_date"] >= cd - pd.Timedelta(days=window_days))]
        if pre.empty:
            continue
        last = pre.sort_values("obs_date").iloc[-1]
        srow = st.loc[iid]
        rows.append({
            "instrument_id": iid,
            "event_key": srow["event_key"],
            "series_ticker": srow["series_ticker"],
            "entry_yes": float(last["yes_price"]),
            "entry_date": last["obs_date"],
            "close_date": cd,
            "settle_date": srow["obs_date"],
            "terminal": float(srow["yes_price"]),
            "life_volume": float(life_vol.get(iid, 0.0)),
            "entry_day_volume": float(last["volume"]),
        })
    ent = pd.DataFrame(rows)
    return ent[ent["life_volume"] >= min_life_volume].reset_index(drop=True)


def wilson_interval(k: float, n: float, z: float = WILSON_Z) -> tuple[float, float]:
    """Standard Wilson score interval for a binomial proportion."""
    if n <= 0:
        return (float("nan"), float("nan"))
    phat = k / n
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2.0 * n)) / denom
    halfwidth = (z * np.sqrt(phat * (1.0 - phat) / n + z * z / (4.0 * n * n))) / denom
    return (center - halfwidth, center + halfwidth)


def pav_isotonic(x: np.ndarray, y: np.ndarray,
                  w: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pool-adjacent-violators isotonic regression (monotone non-decreasing).

    Aggregates duplicate x-values to a weighted mean first, then pools blocks
    bottom-up via a stack. Returns (xmin, xmax, value) arrays for the pooled
    blocks, sorted ascending by x — the fitted curve's breakpoints.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.ones_like(y) if w is None else np.asarray(w, dtype=float)

    order = np.argsort(x, kind="mergesort")
    xs, ys, ws = x[order], y[order], w[order]
    ux, inv = np.unique(xs, return_inverse=True)
    sum_wy = np.zeros(len(ux))
    sum_w = np.zeros(len(ux))
    np.add.at(sum_wy, inv, ys * ws)
    np.add.at(sum_w, inv, ws)
    uy = sum_wy / sum_w

    stack: list[list[float]] = []          # each block: [xmin, xmax, sum_wy, sum_w]
    for xi, yi, wi in zip(ux, uy, sum_w):
        stack.append([xi, xi, yi * wi, wi])
        while len(stack) >= 2 and (stack[-2][2] / stack[-2][3]
                                    > stack[-1][2] / stack[-1][3] + 1e-15):
            b = stack.pop()
            a = stack.pop()
            stack.append([a[0], b[1], a[2] + b[2], a[3] + b[3]])

    xmin = np.array([b[0] for b in stack])
    xmax = np.array([b[1] for b in stack])
    value = np.array([b[2] / b[3] for b in stack])
    return xmin, xmax, value


def make_predictor(xmax: np.ndarray, value: np.ndarray):
    """Right-continuous step function keyed on pooled-block upper edges;
    clamps to the boundary fitted value outside the train price range."""
    def g(p):
        p_arr = np.atleast_1d(np.asarray(p, dtype=float))
        idx = np.clip(np.searchsorted(xmax, p_arr, side="left"), 0, len(value) - 1)
        out = value[idx]
        return out if np.ndim(p) else float(out[0])
    return g


def brier(outcome: np.ndarray, phat: np.ndarray) -> np.ndarray:
    return (np.asarray(outcome, dtype=float) - np.asarray(phat, dtype=float)) ** 2


def cluster_bootstrap_pvalue(diff: np.ndarray, clusters: np.ndarray,
                              n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> tuple[float, np.ndarray]:
    """Paired one-sided cluster bootstrap on per-market diff = Brier(price) -
    Brier(g(price)). Resamples event_key clusters with replacement; p = frac of
    resamples with mean diff <= 0, +1 continuity correction (one-sided test that
    g's Brier is strictly lower than the raw-price Brier)."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"diff": diff, "cluster": clusters})
    groups = df.groupby("cluster")["diff"].apply(lambda s: s.to_numpy())
    cluster_ids = groups.index.to_numpy()
    n_clusters = len(cluster_ids)
    cluster_arrays = groups.to_numpy()

    boot_means = np.empty(n_boot)
    for b in range(n_boot):
        draw_idx = rng.integers(0, n_clusters, size=n_clusters)
        boot_means[b] = np.concatenate(cluster_arrays[draw_idx]).mean()

    p_value = (1 + int(np.sum(boot_means <= 0))) / (n_boot + 1)
    return p_value, boot_means


def main() -> int:
    lake_bars, lake_st = load_panel()
    if len(lake_bars) == 0 or len(lake_st) == 0:
        raise RuntimeError(
            "event_markets_hist lake read returned no bar/settlement rows — "
            "lake missing or empty; refusing to fit on nothing.")

    ent = build_entries(lake_bars, lake_st)

    out: dict = {
        "pre_registration": {
            "category": CATEGORY,
            "production_fallback": "identity function for any non-economics category",
            "fit": "isotonic regression (PAV) of outcome on entry price, "
                   f"train = settlements before {TRAIN_TEST_CUTOFF.date()}",
            "A1": "Brier(g(price)) < Brier(price) on test, paired one-sided "
                  f"cluster bootstrap (event_key clusters, {N_BOOT} draws, "
                  f"seed {BOOT_SEED}), p < 0.05",
            "A2": "sanity: g not degenerate — >=4 distinct fitted values "
                  "across the price deciles on train",
            "decision_rule": "PASS = A1 & A2 -> propose consumption in events "
                              "sizing (execution layer, no n_trials); "
                              "FAIL -> file as a diagnostic record only",
            "filters": {"min_life_volume": MIN_LIFE_VOLUME,
                        "entry_window_days": ENTRY_WINDOW_DAYS,
                        "train_test_cutoff": str(TRAIN_TEST_CUTOFF.date())},
        },
        "entries_frame_source": (
            "kalshi_longshot_fade.load_panel() reused verbatim (same curated-lake "
            "read); build_entries() replicated locally with the life-volume floor "
            "changed 1000 -> 500 contracts, all other T-1 entry-window logic "
            "unchanged."),
        "n_markets": int(len(ent)),
        "n_events": int(ent["event_key"].nunique()) if len(ent) else 0,
        "settle_range": ([str(ent["settle_date"].min()), str(ent["settle_date"].max())]
                          if len(ent) else None),
    }

    if len(ent) < 100:
        out["verdict"] = "INSUFFICIENT DATA (<100 qualifying markets)"
        print(json.dumps(out, indent=2, default=str))
        return 1

    train = ent[ent["settle_date"] < TRAIN_TEST_CUTOFF].reset_index(drop=True)
    test = ent[ent["settle_date"] >= TRAIN_TEST_CUTOFF].reset_index(drop=True)
    out["train"] = {"n": int(len(train)), "n_clusters": int(train["event_key"].nunique()),
                     "settle_range": [str(train["settle_date"].min()),
                                      str(train["settle_date"].max())] if len(train) else None}
    out["test"] = {"n": int(len(test)), "n_clusters": int(test["event_key"].nunique()),
                    "settle_range": [str(test["settle_date"].min()),
                                     str(test["settle_date"].max())] if len(test) else None}

    if len(train) < 50 or len(test) < 30:
        out["verdict"] = "INSUFFICIENT DATA (train<50 or test<30 after split)"
        print(json.dumps(out, indent=2, default=str))
        return 1

    # ------------------------------------------------------------- fit g(p)
    xmin, xmax, value = pav_isotonic(train["entry_yes"].to_numpy(), train["terminal"].to_numpy())
    g = make_predictor(xmax, value)

    out["fitted_curve"] = {
        "extrapolation_rule": (
            "right-continuous step function keyed on pooled-block upper edges "
            "(xmax); g(p) = value[i] for the first block with xmax[i] >= p; "
            "p below the lowest train price clamps to the first block's value, "
            "p above the highest train price clamps to the last block's value"),
        "n_blocks": int(len(value)),
        "block_xmin": xmin.tolist(),
        "block_xmax_breakpoints": xmax.tolist(),
        "block_value": value.tolist(),
    }

    # --------------------------------------------------------- decile table
    deciles = pd.qcut(train["entry_yes"], N_DECILES, duplicates="drop")
    train_d = train.assign(decile=deciles)
    decile_rows = []
    decile_fitted_vals = []
    for interval, b in train_d.groupby("decile", observed=True):
        n = int(len(b))
        k = float(b["terminal"].sum())
        mean_price = float(b["entry_yes"].mean())
        win_rate = float(b["terminal"].mean())
        lo, hi = wilson_interval(k, n)
        fitted_at_mean = float(g(mean_price))
        decile_fitted_vals.append(round(fitted_at_mean, 10))
        decile_rows.append({
            "price_range": f"({interval.left:.4g}, {interval.right:.4g}]",
            "n": n, "mean_price": mean_price, "win_rate": win_rate,
            "wilson_lo": float(lo), "wilson_hi": float(hi),
            "g_at_mean_price": fitted_at_mean,
        })
    out["decile_table"] = decile_rows

    # ------------------------------------------------------------------ A2
    n_distinct = len(set(decile_fitted_vals))
    out["A2"] = {"n_deciles": len(decile_rows), "n_distinct_fitted_values": n_distinct,
                 "pass": bool(n_distinct >= 4)}

    # ------------------------------------------------------------------ A1
    test_price = test["entry_yes"].to_numpy()
    test_outcome = test["terminal"].to_numpy()
    test_g = np.asarray(g(test_price), dtype=float)

    brier_price_i = brier(test_outcome, test_price)
    brier_g_i = brier(test_outcome, test_g)
    diff_i = brier_price_i - brier_g_i          # >0 means g's Brier is lower (better)

    brier_price = float(brier_price_i.mean())
    brier_g = float(brier_g_i.mean())
    point_diff = brier_price - brier_g

    p_value, boot_means = cluster_bootstrap_pvalue(diff_i, test["event_key"].to_numpy())

    out["A1"] = {
        "brier_price": brier_price, "brier_g": brier_g,
        "diff_price_minus_g": point_diff,
        "n_boot": N_BOOT, "seed": BOOT_SEED,
        "p_value": float(p_value),
        "boot_mean_diff": float(np.mean(boot_means)),
        "boot_std_diff": float(np.std(boot_means)),
        "pass": bool(brier_g < brier_price and p_value < 0.05),
    }

    out["verdict"] = "PASS" if out["A1"]["pass"] and out["A2"]["pass"] else "FAIL"
    out["verdict_note"] = (
        "propose consumption in events sizing (execution layer, no n_trials)"
        if out["verdict"] == "PASS"
        else "file the curve as a diagnostic record only")

    out["by_series"] = {s: int(n) for s, n in ent["series_ticker"].value_counts().items()}

    artifact = REPO_ROOT / "diagnostics" / "events_calibration.json"
    artifact.write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps(out, indent=2, default=str))
    print(f"\nartifact -> {artifact}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
