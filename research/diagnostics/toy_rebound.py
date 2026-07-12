"""PRE-REGISTERED diagnostic — turn-of-year tax-loss-selling rebound (wiki backlog #17).

Mechanism (practitioner-mechanism-scan #7): tax-loss-selling pressure on prior-year
losers reverses at the turn of the year. Well-covered lineage; the residual effect
in the modern era is documented mainly in small/illiquid, low-institutional-ownership
names. Our investable universe is PIT S&P 500 members, which skews large-cap/liquid —
this diagnostic therefore tests whether the effect survives AT TRADEABLE SCALE IN OUR
UNIVERSE, not whether the mechanism exists at all in the broader literature.

=============================== PRE-REGISTRATION (verbatim) ================================
Written 2026-07-10 BEFORE the panel was first read. Mechanism: tax-loss-selling
pressure on prior-year losers reverses at the turn of the year (long lineage; the
residual effect is documented in small/illiquid low-institutional-ownership names).
Our investable universe is PIT S&P 500 members — this diagnostic therefore tests
whether the effect exists AT TRADEABLE SCALE IN OUR UNIVERSE, not whether the
mechanism exists at all. A null kills usability here, not the literature.

Universe & formation, per TOY event year Y (Y = 2016..2025):
  - members: PIT S&P 500 membership on the formation date (membership intervals
    from the lake reference tables); prices from curated prices/equity only
    (hygiene layer already applied at ingest).
  - formation date F(Y) = the 4th-to-last trading day of December Y (NYSE grid
    from the price panel itself).
  - prior-year return = total close-to-close return from the first trading day of
    January Y to F(Y), requiring >= 200 price observations in that span.
  - deciles by prior-year return (decile 1 = losers, decile 10 = winners).
  - TOY window = last 3 trading days of Dec Y through first 5 trading days of
    Jan Y+1 (8 trading days), entered at F(Y) close.

Outcome per event year: TOY_spread(Y) = equal-weight decile-1 TOY-window return
minus equal-weight decile-10 TOY-window return (the loser-rebound L/S), and
TOY_alpha(Y) = decile-1 return minus cross-sectional mean return (long-only tilt).

Incrementality controls (both required):
  C1 (placebo windows): for each Y, 40 placebo 8-trading-day windows with start
     dates drawn uniformly from Oct 1 of Y to Mar 1 of Y+1 EXCLUDING any window
     overlapping the TOY window; same formation/deciles. Effect must be
     concentrated: TOY_spread(Y) percentile within its year's placebo distribution.
  C2 (reversal control): pooled cross-sectional regression within each window
     (TOY and placebo): r_i,window = a + b1 * prior_year_ret_i + b2 *
     prior_month_ret_i + e (prior_month = 21 trading days ending at F(Y);
     b2 absorbs generic short-term reversal). The TOY effect is b1(TOY windows)
     vs the placebo distribution of b1.

DECISION RULES (pre-registered):
  T1: mean over years of TOY_spread > 0 with one-sided t >= 1.645 across the ~10
      yearly observations (each year one obs; plain t-test, small-N caveat stands).
  T2: pooled TOY b1 is NEGATIVE (losers rebound => negative coefficient on
      prior-year return) AND below the 10th percentile of the placebo b1
      distribution (concentration + incrementality vs reversal).
  T3 (cost realism): T1's spread survives 2 x 5bp per-side costs on the 8-day
      round trip (i.e., mean spread > 20bp).
  PASS = T1 & T2 & T3 -> propose a December-seasonal overlay study as a candidate
  (registering it later = a new trial at the gate; this diagnostic burns none).
  Any spec change after seeing results is a NEW pre-registration.
=============================================================================================

IMPLEMENTATION NOTES (decisions made where the pre-registration is silent on mechanics;
flagged here rather than silently reinterpreted):
  - "Pooled TOY b1" (T2) = a single cluster-robust OLS run on the union of all TOY-window
    cross-sections across years (one row per (year, instrument)), clustered by year via
    `_common.cluster_ols`. This is the literal reading of "pooled" as one regression over
    stacked observations, as opposed to averaging ten separate per-year b1 estimates.
  - "The placebo distribution of b1" (T2) = the collection of per-window b1 estimates (one
    OLS per placebo window, NOT stacked) pooled across every (year, draw) — i.e. up to
    10 * 40 = 400 values — from which the 10th percentile is taken. This mirrors C1's
    per-window statistic (spread) with the same "one regression per window" unit, just
    aggregated across years for the single "placebo b1 distribution" T2 refers to (whereas
    C1's percentile check is explicitly per-year).
  - C1's per-year percentile is computed and reported for every year but is NOT folded into
    PASS/FAIL directly — the DECISION RULES section only gates on T1, T2, T3, and T2 already
    carries the incrementality-vs-placebo test (via C2's b1 distribution). C1 is reported as
    a descriptive concentration check alongside the gates, per the literal text of the
    decision rules.
  - T3 checks the raw (pre-existing) T1 mean spread against the 20bp threshold; it does not
    separately re-run T1's t-test net of costs — "T1's spread survives ... costs" is read as
    a magnitude check on the same mean-spread statistic T1 already computes.
  - Entry timing for placebo windows mirrors the TOY convention: entered at the close one
    trading day before the window's first day (paralleling "entered at F(Y) close", since
    F(Y) is exactly one trading day before the TOY window's first day).

Usage: uv run python research/diagnostics/toy_rebound.py
Artifact: diagnostics/toy_rebound.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "research" / "diagnostics"))

from production.core.lake import Lake, LakeError  # noqa: E402
from production.reference.universe import membership_as_of  # noqa: E402
from _common import cluster_ols  # noqa: E402

# ------------------------------------------------------------------- config
EVENT_YEARS = list(range(2016, 2026))       # Y = 2016..2025
N_PLACEBO = 40
PLACEBO_SEED = 17                           # np.random.default_rng(17), documented per task
FORMATION_OFFSET = 4                        # F(Y) = 4th-to-last trading day of December
TOY_DEC_DAYS = 3
TOY_JAN_DAYS = 5
TOY_WINDOW_LEN = TOY_DEC_DAYS + TOY_JAN_DAYS  # 8 trading days
PRIOR_MONTH_DAYS = 21
MIN_PRIOR_YEAR_OBS = 200
N_DECILES = 10
LOSER_DECILE = 1
WINNER_DECILE = 10
COST_PER_SIDE_BP = 5.0                      # equity floor, configs/costs.yaml
T3_THRESHOLD = 4 * COST_PER_SIDE_BP / 1e4   # 2 legs x 2 sides x 5bp = 20bp round trip
T_CRIT = 1.645                              # one-sided, ~90% confidence


# ------------------------------------------------------------------- grid helpers
def trading_grid(prices: pd.DataFrame) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(sorted(prices["obs_date"].unique()))


def month_days(grid: pd.DatetimeIndex, year: int, month: int) -> pd.DatetimeIndex:
    return grid[(grid.year == year) & (grid.month == month)]


def build_year_spec(grid: pd.DatetimeIndex, Y: int) -> dict | None:
    """Formation date, TOY window, and lookback anchors for event year Y."""
    dec = month_days(grid, Y, 12)
    jan_next = month_days(grid, Y + 1, 1)
    jan_this = month_days(grid, Y, 1)
    if len(dec) < FORMATION_OFFSET or len(jan_next) < TOY_JAN_DAYS or len(jan_this) == 0:
        return None
    f_date = dec[-FORMATION_OFFSET]
    idx_f = grid.get_loc(f_date)
    if idx_f < PRIOR_MONTH_DAYS:
        return None
    toy_days = dec[-TOY_DEC_DAYS:].append(jan_next[:TOY_JAN_DAYS])
    assert len(toy_days) == TOY_WINDOW_LEN
    # TOY window must be contiguous immediately after F(Y) in the grid (sanity check
    # of the pre-registration's own internal consistency: F(Y) is the 4th-to-last
    # trading day of Dec, so the "last 3 trading days of Dec" are exactly the 3
    # trading days following F(Y), and the window is entered at F(Y) close).
    assert grid.get_loc(toy_days[0]) == idx_f + 1
    exit_date = toy_days[-1]
    idx_toy_start = idx_f + 1
    idx_toy_end = idx_f + TOY_WINDOW_LEN  # inclusive index of exit_date
    return {
        "Y": Y, "f_date": f_date, "idx_f": idx_f,
        "exit_date": exit_date, "toy_days": toy_days,
        "idx_toy_start": idx_toy_start, "idx_toy_end": idx_toy_end,
        "prior_year_start": jan_this[0],
        "prior_month_start": grid[idx_f - PRIOR_MONTH_DAYS],
    }


def placebo_start_indices(grid: pd.DatetimeIndex, spec: dict, rng: np.random.Generator) -> list[int]:
    """Up to N_PLACEBO grid indices for 8-day windows in [Oct 1 Y, Mar 1 Y+1],
    excluding any window overlapping the TOY window, with a full day of history
    before the window (entry) and 8 trading days inside the panel (exit)."""
    Y = spec["Y"]
    oct1 = pd.Timestamp(year=Y, month=10, day=1)
    mar1 = pd.Timestamp(year=Y + 1, month=3, day=1)
    candidates = grid[(grid >= oct1) & (grid <= mar1)]
    eligible = []
    for d in candidates:
        idx_s = grid.get_loc(d)
        if idx_s - 1 < 0 or idx_s + TOY_WINDOW_LEN - 1 >= len(grid):
            continue
        idx_e = idx_s + TOY_WINDOW_LEN - 1
        overlap = not (idx_e < spec["idx_toy_start"] or idx_s > spec["idx_toy_end"])
        if overlap:
            continue
        eligible.append(idx_s)
    n = min(N_PLACEBO, len(eligible))
    if n == 0:
        return []
    chosen = rng.choice(np.array(eligible), size=n, replace=False)
    return sorted(int(i) for i in chosen)


# ------------------------------------------------------------------- stats helpers
def one_sample_t(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 2:
        return {"mean": float(x.mean()) if n else float("nan"), "se": float("nan"),
                "t": float("nan"), "n": n}
    mean = float(x.mean())
    sd = float(x.std(ddof=1))
    se = sd / np.sqrt(n)
    t = mean / se if se > 0 else float("nan")
    return {"mean": mean, "sd": sd, "se": se, "t": t, "n": n}


def percentile_rank(value: float, dist: np.ndarray) -> float | None:
    dist = np.asarray(dist, dtype=float)
    dist = dist[~np.isnan(dist)]
    if len(dist) == 0 or np.isnan(value):
        return None
    return float((dist <= value).mean() * 100.0)


def ols_b1(y: np.ndarray, x1: np.ndarray, x2: np.ndarray) -> float | None:
    """Single-window cross-sectional OLS b1 (no clustering needed: one window)."""
    mask = ~(np.isnan(y) | np.isnan(x1) | np.isnan(x2))
    if mask.sum() < 5:
        return None
    X = np.column_stack([np.ones(mask.sum()), x1[mask], x2[mask]])
    beta, *_ = np.linalg.lstsq(X, y[mask], rcond=None)
    return float(beta[1])


# ------------------------------------------------------------------- per-year core
def compute_year(prices: pd.DataFrame, membership: pd.DataFrame, grid: pd.DatetimeIndex,
                  Y: int, rng: np.random.Generator) -> dict | None:
    spec = build_year_spec(grid, Y)
    if spec is None:
        return None

    members = membership_as_of(membership, "sp500", spec["f_date"])
    present = set(prices["instrument_id"].unique())
    members = [m for m in members if m in present]
    if len(members) < 50:
        return None

    date_lo = spec["prior_year_start"]
    date_hi = pd.Timestamp(year=Y + 1, month=4, day=30)
    sub = prices.loc[prices["instrument_id"].isin(members)
                      & (prices["obs_date"] >= date_lo)
                      & (prices["obs_date"] <= date_hi),
                      ["obs_date", "instrument_id", "close"]]
    wide = sub.pivot_table(index="obs_date", columns="instrument_id", values="close")
    date_range = grid[(grid >= date_lo) & (grid <= date_hi)]
    wide = wide.reindex(date_range)

    # ---- prior-year return + >=200-obs eligibility filter
    py_span = sub.loc[(sub["obs_date"] >= spec["prior_year_start"])
                       & (sub["obs_date"] <= spec["f_date"])]
    obs_count = py_span.groupby("instrument_id").size()

    start_px = wide.loc[spec["prior_year_start"]]
    f_px = wide.loc[spec["f_date"]]
    pm_px = wide.loc[spec["prior_month_start"]]
    prior_year_ret = f_px / start_px - 1.0
    prior_month_ret = f_px / pm_px - 1.0

    members_df = pd.DataFrame({
        "instrument_id": members,
    }).set_index("instrument_id")
    members_df["obs_count"] = obs_count.reindex(members_df.index)
    members_df["prior_year_ret"] = prior_year_ret.reindex(members_df.index)
    members_df["prior_month_ret"] = prior_month_ret.reindex(members_df.index)
    n_before_filter = len(members_df)
    members_df = members_df[members_df["obs_count"] >= MIN_PRIOR_YEAR_OBS]
    members_df = members_df.dropna(subset=["prior_year_ret", "prior_month_ret"])
    n_eligible = len(members_df)
    if n_eligible < 50:
        return None

    members_df["decile"] = pd.qcut(members_df["prior_year_ret"], N_DECILES,
                                    labels=False, duplicates="drop") + 1
    loser_ids = members_df.index[members_df["decile"] == LOSER_DECILE]
    winner_ids = members_df.index[members_df["decile"] == WINNER_DECILE]

    def window_return(entry_date, exit_date) -> pd.Series:
        entry_px = wide.loc[entry_date, members_df.index]
        exit_px = wide.loc[exit_date, members_df.index]
        return exit_px / entry_px - 1.0

    # ---- TOY window outcome
    toy_ret = window_return(spec["f_date"], spec["exit_date"])
    toy_spread = float(toy_ret[loser_ids].mean() - toy_ret[winner_ids].mean())
    toy_alpha = float(toy_ret[loser_ids].mean() - toy_ret.mean())
    b1_toy = ols_b1(toy_ret.to_numpy(), members_df["prior_year_ret"].to_numpy(),
                     members_df["prior_month_ret"].to_numpy())

    # ---- C1/C2 placebo windows
    idxs = placebo_start_indices(grid, spec, rng)
    placebo_spreads, placebo_b1s = [], []
    for idx_s in idxs:
        entry_date = grid[idx_s - 1]
        exit_date = grid[idx_s + TOY_WINDOW_LEN - 1]
        ret = window_return(entry_date, exit_date)
        sp = ret[loser_ids].mean() - ret[winner_ids].mean()
        if not np.isnan(sp):
            placebo_spreads.append(float(sp))
        b1 = ols_b1(ret.to_numpy(), members_df["prior_year_ret"].to_numpy(),
                     members_df["prior_month_ret"].to_numpy())
        if b1 is not None:
            placebo_b1s.append(b1)

    c1_percentile = percentile_rank(toy_spread, np.array(placebo_spreads))

    return {
        "Y": Y,
        "f_date": str(spec["f_date"].date()),
        "exit_date": str(spec["exit_date"].date()),
        "n_members_pit": len(members),
        "n_eligible": n_eligible,
        "n_before_prior_year_filter": n_before_filter,
        "n_loser_decile": int(len(loser_ids)),
        "n_winner_decile": int(len(winner_ids)),
        "n_placebo_windows": len(idxs),
        "TOY_spread": toy_spread,
        "TOY_alpha": toy_alpha,
        "b1_TOY": b1_toy,
        "C1_placebo_percentile": c1_percentile,
        "placebo_spreads": placebo_spreads,
        "placebo_b1s": placebo_b1s,
        # stacked rows for the pooled TOY regression (T2)
        "_toy_regression_rows": pd.DataFrame({
            "ret": toy_ret.to_numpy(),
            "prior_year_ret": members_df["prior_year_ret"].to_numpy(),
            "prior_month_ret": members_df["prior_month_ret"].to_numpy(),
            "Y": Y,
        }).dropna(subset=["ret"]),
    }


# ------------------------------------------------------------------- main
def main() -> int:
    t0 = time.time()
    lake = Lake()

    try:
        prices = lake.read_curated("prices", "equity")
    except LakeError as exc:
        print(f"FATAL: curated prices/equity unavailable: {exc}", file=sys.stderr)
        return 1
    if prices.empty:
        print("FATAL: curated prices/equity is empty", file=sys.stderr)
        return 1
    prices = prices[["obs_date", "instrument_id", "close"]].copy()
    prices["obs_date"] = pd.to_datetime(prices["obs_date"])

    try:
        membership = lake.read_reference("membership_equity")
    except LakeError as exc:
        print(f"FATAL: reference membership_equity unavailable: {exc}", file=sys.stderr)
        return 1
    if membership.empty or "sp500" not in set(membership["universe"].unique()):
        print("FATAL: membership_equity has no sp500 rows", file=sys.stderr)
        return 1

    n_rows, n_instr = len(prices), prices["instrument_id"].nunique()
    date_span = (str(prices["obs_date"].min().date()), str(prices["obs_date"].max().date()))
    print(f"data sanity: prices/equity rows={n_rows:,} instruments={n_instr} "
          f"span={date_span[0]}..{date_span[1]} membership_rows={len(membership)}")
    if n_rows < 100_000 or n_instr < 100:
        print("FATAL: prices/equity looks too small for a PIT S&P 500 panel "
              f"(rows={n_rows}, instruments={n_instr})", file=sys.stderr)
        return 1

    grid = trading_grid(prices)
    rng = np.random.default_rng(PLACEBO_SEED)

    years_out = []
    toy_rows = []
    all_placebo_b1 = []
    for Y in EVENT_YEARS:
        rec = compute_year(prices, membership, grid, Y, rng)
        if rec is None:
            print(f"  year {Y}: skipped (insufficient data)")
            continue
        toy_rows.append(rec.pop("_toy_regression_rows"))
        all_placebo_b1.extend(rec["placebo_b1s"])
        years_out.append(rec)
        print(f"  year {Y}: F(Y)={rec['f_date']} n_eligible={rec['n_eligible']} "
              f"TOY_spread={rec['TOY_spread']*1e4:+.1f}bp "
              f"C1_pct={rec['C1_placebo_percentile']:.1f} "
              f"placebo_n={rec['n_placebo_windows']}")

    out: dict = {
        "pre_registration": {
            "T1": "mean(TOY_spread) > 0, one-sided t >= 1.645 across event years",
            "T2": "pooled TOY b1 < 0 AND < 10th pct of pooled placebo b1 distribution",
            "T3": f"mean(TOY_spread) > {T3_THRESHOLD*1e4:.0f}bp (2x5bp per-side, 8-day round trip)",
            "PASS": "T1 & T2 & T3",
        },
        "config": {
            "event_years": EVENT_YEARS, "n_placebo": N_PLACEBO, "placebo_seed": PLACEBO_SEED,
            "formation_offset": FORMATION_OFFSET, "toy_window_days": TOY_WINDOW_LEN,
            "prior_month_days": PRIOR_MONTH_DAYS, "min_prior_year_obs": MIN_PRIOR_YEAR_OBS,
        },
        "data_sanity": {
            "n_price_rows": n_rows, "n_instruments": n_instr, "date_span": date_span,
            "n_membership_rows": int(len(membership)),
        },
        "n_years_used": len(years_out),
    }

    if len(years_out) < 5:
        out["verdict"] = "INSUFFICIENT DATA (<5 usable event years)"
        print(json.dumps(out, indent=2, default=str))
        return 1

    # ---- per-year table (drop the large placebo arrays from the printed table;
    # keep them in a separate section of the artifact for auditability)
    year_table = [{k: v for k, v in r.items()
                   if k not in ("placebo_spreads", "placebo_b1s")} for r in years_out]
    out["years"] = year_table
    out["placebo_detail"] = {r["Y"]: {"spreads": r["placebo_spreads"], "b1s": r["placebo_b1s"]}
                              for r in years_out}

    # ---------------------------------------------------------------- T1
    spreads = np.array([r["TOY_spread"] for r in years_out])
    t1_stat = one_sample_t(spreads)
    t1_pass = bool(t1_stat["n"] >= 2 and t1_stat["mean"] > 0 and t1_stat["t"] >= T_CRIT)
    out["T1"] = {**t1_stat, "pass": t1_pass}

    # ---------------------------------------------------------------- T2
    toy_stack = pd.concat(toy_rows, ignore_index=True)
    X = np.column_stack([np.ones(len(toy_stack)), toy_stack["prior_year_ret"].to_numpy(),
                         toy_stack["prior_month_ret"].to_numpy()])
    pooled = cluster_ols(toy_stack["ret"].to_numpy(), X, toy_stack["Y"].to_numpy())
    b1_pooled = float(pooled["beta"][1])
    placebo_b1_arr = np.array(all_placebo_b1)
    b1_10th_pct = float(np.percentile(placebo_b1_arr, 10)) if len(placebo_b1_arr) else float("nan")
    t2_pass = bool(len(placebo_b1_arr) > 0 and b1_pooled < 0 and b1_pooled < b1_10th_pct)
    out["T2"] = {
        "b1_pooled_TOY": b1_pooled, "se": float(pooled["se"][1]), "t": float(pooled["t"][1]),
        "n": pooled["n"], "n_year_clusters": pooled["n_clusters"],
        "placebo_b1_distribution": {
            "n": int(len(placebo_b1_arr)),
            "mean": float(np.mean(placebo_b1_arr)) if len(placebo_b1_arr) else None,
            "sd": float(np.std(placebo_b1_arr, ddof=1)) if len(placebo_b1_arr) > 1 else None,
            "p05": float(np.percentile(placebo_b1_arr, 5)) if len(placebo_b1_arr) else None,
            "p10": b1_10th_pct if len(placebo_b1_arr) else None,
            "p25": float(np.percentile(placebo_b1_arr, 25)) if len(placebo_b1_arr) else None,
            "p50": float(np.percentile(placebo_b1_arr, 50)) if len(placebo_b1_arr) else None,
            "p75": float(np.percentile(placebo_b1_arr, 75)) if len(placebo_b1_arr) else None,
            "p95": float(np.percentile(placebo_b1_arr, 95)) if len(placebo_b1_arr) else None,
        },
        "pass": t2_pass,
    }

    # ---------------------------------------------------------------- T3
    t3_pass = bool(t1_stat["n"] >= 2 and t1_stat["mean"] > T3_THRESHOLD)
    out["T3"] = {"mean_spread_bp": t1_stat["mean"] * 1e4, "threshold_bp": T3_THRESHOLD * 1e4,
                 "pass": t3_pass}

    out["verdict"] = "PASS" if (t1_pass and t2_pass and t3_pass) else "FAIL"
    out["runtime_sec"] = round(time.time() - t0, 1)

    artifact = REPO_ROOT / "diagnostics" / "toy_rebound.json"
    artifact.write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps(out, indent=2, default=str))
    print(f"\nartifact -> {artifact}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
