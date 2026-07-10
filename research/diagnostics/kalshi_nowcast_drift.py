"""PRE-REGISTERED diagnostic #2 — Kalshi inflation-ladder drift toward the
Cleveland Fed nowcast.

Mechanism (practitioner scan idea #2; Diercks-Katz-Wright FEDS 2026-010 establish
nowcast accuracy for these ladders, no drift regression; Jia et al. 2604.20421
hint the MARKET may lead the nowcast — the sign of beta below is decisive either
way). PIT: nowcast vintages are the as-published daily paths (loader stamps
publication day 15:00 UTC); the ladder side uses daily trade bars.

=============================== PRE-REGISTRATION ================================
Written 2026-07-10 BEFORE the joined panel was first read. Universe: settled
events of the MoM inflation ladders KXCPI / KXCPICORE / KXPCECORE (exact-unit
match with CLEV_NOWCAST_{CPI,CORECPI,COREPCE}_MOM:<target-month> vintages).

Implied market mean per (event, day):
  - markets with strike_type == "greater" and finite floor_strike;
  - per-market last bar price forward-filled up to 5 calendar days;
  - survival curve s(x_k) = ffilled YES price at strike x_k, strikes ascending,
    monotonicity enforced by running minimum from the lowest strike;
  - validity: >= 5 live strikes, s(min strike) >= 0.85, s(max strike) <= 0.15
    (tails covered), else the day is skipped;
  - E[print] = (x_1 - d) + trapezoid integral of s over [x_1 - d, x_K + d] with
    s = 1 below x_1 - d and 0 above x_K + d, d = median strike spacing.

Regression (pooled over events e, days t in the last 10 calendar days before each
event's close): dm_{e,t} = alpha + beta * (nowcast_{e,t-1} - m_{e,t-1}) + eps,
where nowcast_{e,t-1} is the latest vintage for e's TARGET MONTH with
available_from <= end of day t-1 (UTC). SE clustered by event (CR1).

Q1: beta > 0 with |t| >= 1.96 -> pre-release drift toward the nowcast confirmed
    (under-reaction; tradeable direction) -> propose candidate execution study.
Q2: beta < 0 with |t| >= 1.96 -> market leads the nowcast (Jia et al. sign) ->
    mechanism dead as a fade; file as anti-edge.
Else: inconclusive -> dead by default. Any spec change after seeing results is a
NEW pre-registration.
=================================================================================

Secondary descriptives (no pass/fail): terminal-accuracy horse race
MAE(final implied mean vs print) vs MAE(final nowcast vs print); per-series betas.

Usage: uv run python research/diagnostics/kalshi_nowcast_drift.py
Artifact: diagnostics/kalshi_nowcast_drift.json
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "research" / "diagnostics"))

from production.core.lake import Lake  # noqa: E402
from _common import cluster_ols  # noqa: E402

SERIES_TO_NOWCAST = {
    "KXCPI": "CLEV_NOWCAST_CPI_MOM",
    "KXCPICORE": "CLEV_NOWCAST_CORECPI_MOM",
    "KXPCECORE": "CLEV_NOWCAST_COREPCE_MOM",
}
_MONTHS = {m: i + 1 for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
     "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"])}
FFILL_DAYS = 5
MIN_STRIKES = 5
TAIL_LO, TAIL_HI = 0.85, 0.15
WINDOW_DAYS = 10


def target_month(event_key: str) -> str | None:
    """'KXCPI-25JUL' / 'CPI-23JUN' -> '2025-07' / '2023-06'."""
    m = re.search(r"-(\d{2})([A-Z]{3})$", str(event_key))
    if not m or m.group(2) not in _MONTHS:
        return None
    return f"20{m.group(1)}-{_MONTHS[m.group(2)]:02d}"


def implied_mean_curve(day_prices: pd.Series) -> float | None:
    """day_prices: index=floor_strike (ascending), values=YES price. See header."""
    s = day_prices.dropna()
    if len(s) < MIN_STRIKES:
        return None
    s = s.sort_index()
    surv = np.minimum.accumulate(s.to_numpy())          # enforce non-increasing
    if surv[0] < TAIL_LO or surv[-1] > TAIL_HI:
        return None
    x = s.index.to_numpy(dtype=float)
    d = float(np.median(np.diff(x))) if len(x) > 1 else 0.1
    xs = np.concatenate([[x[0] - d], x, [x[-1] + d]])
    ss = np.concatenate([[1.0], surv, [0.0]])
    return float(xs[0] + np.trapezoid(ss, xs))


def build_event_panel(bars: pd.DataFrame, st: pd.DataFrame) -> pd.DataFrame:
    """Rows: (event_key, series, obs_date, implied_mean, close_date, print)."""
    st_idx = st.drop_duplicates("instrument_id", keep="last").set_index("instrument_id")
    rows = []
    for (series, event), g in bars.groupby(["series_ticker", "event_key"]):
        if series not in SERIES_TO_NOWCAST:
            continue
        tag = target_month(event)
        if tag is None:
            continue
        meta = st_idx[st_idx["event_key"] == event]
        if meta.empty:
            continue
        close = pd.to_datetime(meta["close_time"].iloc[0], utc=True, format="ISO8601")
        close_date = close.tz_localize(None).normalize()
        try:
            print_val = float(meta["expiration_value"].iloc[0])
        except (TypeError, ValueError):
            print_val = np.nan
        g = g[np.isfinite(g["floor_strike"]) & (g["strike_type"] == "greater")]
        if g.empty:
            continue
        px = (g.pivot_table(index="obs_date", columns="floor_strike",
                            values="yes_price", aggfunc="last")
                .sort_index().ffill(limit=FFILL_DAYS))
        px = px[(px.index >= close_date - pd.Timedelta(days=WINDOW_DAYS))
                & (px.index < close_date)]
        for day, row in px.iterrows():
            m = implied_mean_curve(row)
            if m is None:
                continue
            rows.append({"event_key": event, "series_ticker": series, "tag": tag,
                         "obs_date": day, "implied_mean": m,
                         "close_date": close_date, "print": print_val})
    return pd.DataFrame(rows)


def main() -> int:
    lake = Lake()
    df = lake.read_curated("event_markets_hist", "events")
    bars = df[df["row_type"] == "bar"].copy()
    st = df[df["row_type"] == "settlement"].copy()
    macro = lake.read_curated("macro", "macro")
    nc = macro[macro["series_id"].astype(str).str.contains("_MOM:")].copy()
    nc["available_from"] = pd.to_datetime(nc["available_from"], utc=True)

    panel = build_event_panel(bars, st)
    out: dict = {"pre_registration": {
        "regression": "dm_t = a + b*(nowcast_{t-1} - m_{t-1}), cluster by event",
        "Q1": "b > 0, |t| >= 1.96 -> drift toward nowcast (promote)",
        "Q2": "b < 0, |t| >= 1.96 -> market leads nowcast (anti-edge)",
        "filters": {"ffill_days": FFILL_DAYS, "min_strikes": MIN_STRIKES,
                    "tails": [TAIL_LO, TAIL_HI], "window_days": WINDOW_DAYS},
    }, "n_event_days": int(len(panel)),
       "n_events": int(panel["event_key"].nunique()) if len(panel) else 0}

    if len(panel) < 60:
        out["verdict"] = "INSUFFICIENT DATA (<60 valid event-days)"
        print(json.dumps(out, indent=2, default=str))
        return 1

    # join the PIT nowcast: latest vintage for the event's target month known by
    # the END of day t-1 (UTC)
    reg_rows = []
    for (event, series, tag), g in panel.groupby(["event_key", "series_ticker", "tag"]):
        sid = f"{SERIES_TO_NOWCAST[series]}:{tag}"
        vintages = nc[nc["series_id"] == sid].sort_values("available_from")
        if vintages.empty:
            continue
        g = g.sort_values("obs_date")
        m = g.set_index("obs_date")["implied_mean"]
        for t_prev, t_cur in zip(g["obs_date"].iloc[:-1], g["obs_date"].iloc[1:]):
            if (t_cur - t_prev).days > FFILL_DAYS:
                continue
            known = vintages[vintages["available_from"]
                             <= (t_prev + pd.Timedelta(days=1)).tz_localize("UTC")]
            if known.empty:
                continue
            reg_rows.append({
                "event_key": event, "series_ticker": series,
                "dm": float(m[t_cur] - m[t_prev]),
                "gap": float(known["value"].iloc[-1] - m[t_prev]),
            })
    reg = pd.DataFrame(reg_rows)
    out["n_regression_obs"] = int(len(reg))
    if len(reg) < 40:
        out["verdict"] = "INSUFFICIENT DATA (<40 regression observations)"
        print(json.dumps(out, indent=2, default=str))
        return 1

    X = np.column_stack([np.ones(len(reg)), reg["gap"].to_numpy()])
    r = cluster_ols(reg["dm"].to_numpy(), X, reg["event_key"].to_numpy())
    beta, t = float(r["beta"][1]), float(r["t"][1])
    out["regression"] = {"alpha": float(r["beta"][0]), "beta": beta,
                         "t_alpha": float(r["t"][0]), "t_beta": t,
                         "n": r["n"], "n_clusters": r["n_clusters"]}
    out["verdict"] = ("Q1 PASS: drift toward nowcast" if beta > 0 and abs(t) >= 1.96
                      else "Q2: market leads nowcast (anti-edge)"
                      if beta < 0 and abs(t) >= 1.96 else "INCONCLUSIVE -> dead")

    per_series = []
    for s, g in reg.groupby("series_ticker"):
        if len(g) < 20:
            continue
        Xs = np.column_stack([np.ones(len(g)), g["gap"].to_numpy()])
        rs = cluster_ols(g["dm"].to_numpy(), Xs, g["event_key"].to_numpy())
        per_series.append({"series": s, "beta": float(rs["beta"][1]),
                           "t": float(rs["t"][1]), "n": rs["n"]})
    out["per_series"] = per_series

    # secondary: terminal accuracy horse race (descriptive)
    finals = (panel.sort_values("obs_date").groupby("event_key").last()
              .dropna(subset=["print"]))
    horse = []
    for event, row in finals.iterrows():
        sid = f"{SERIES_TO_NOWCAST[row['series_ticker']]}:{row['tag']}"
        vint = nc[(nc["series_id"] == sid)
                  & (nc["available_from"]
                     <= row["close_date"].tz_localize("UTC"))]
        if vint.empty:
            continue
        horse.append({"market_err": abs(row["implied_mean"] - row["print"]),
                      "nowcast_err": abs(float(vint.sort_values("available_from")
                                               ["value"].iloc[-1]) - row["print"])})
    if horse:
        h = pd.DataFrame(horse)
        out["terminal_mae"] = {"market": float(h["market_err"].mean()),
                               "nowcast": float(h["nowcast_err"].mean()),
                               "n_events": int(len(h))}

    artifact = REPO_ROOT / "diagnostics" / "kalshi_nowcast_drift.json"
    artifact.write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps(out, indent=2, default=str))
    print(f"\nartifact -> {artifact}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
