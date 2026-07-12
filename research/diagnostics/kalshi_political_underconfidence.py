"""PRE-REGISTERED diagnostic #4 — political-market underconfidence (scan idea #6).

Mechanism (Le, arXiv 2602.19520, 292M trades): political prediction markets show
UNDERCONFIDENCE — the calibration slope of outcome on price exceeds 1, i.e.
favorites win more often than priced. This is the OPPOSITE sign to the
favorite-longshot bias (idea #1) and is domain-conditional: never pool political
with economics markets (our macro sample showed a different structure; Whelan
Table 8 shows category heterogeneity).

=============================== PRE-REGISTRATION ================================
Written 2026-07-11 BEFORE the Kalshi Politics-category backfill was ingested or
read (committed ahead of the data, like iterations 5-6). Universe: settled binary
Kalshi POLITICS-category markets from the `kalshi_hist_politics` ingest — the
series set is exactly the series_tickers recorded in that ingest's audit record,
minus any of the 12 macro series (domain-conditionality guard); life volume
>= 1,000 contracts; entry = last daily-bar traded YES price on a day in
[close-10d, close-1d] (the T-1 convention shared by diagnostics #1-#3); outcome =
settlement value; fees per research/diagnostics/_common.py taker_fee. Tests
clustered by event_key (CR1):

Q1 (calibration sign): Mincer-Zarnowitz y - p = alpha + psi*p has psi > 0 with
    |t| >= 1.96 (two-sided). In this form psi > 0 means high-priced contracts
    win MORE than priced — Le's underconfidence direction.
Q2 (tradeable side): buying YES as taker at entry in [0.70, 0.95] earns mean
    post-fee return > 0 with t >= +1.645 (one-sided). The fee at ~0.90 is
    ~0.6-0.7% of stake — the edge must clear it.
Q3 (recency): Q2's point estimate restricted to settlements in the most recent
    18 months is > 0 (18 not 12: political settlement flow is election-cycle
    lumpy and a 12m window can be cycle-empty).

Decision rule: Q1 & Q2 & Q3 pass -> propose a favorite-tilt events-sleeve signal
design for politics markets (events sleeve return stream; NOT a registry factor;
any later signal addition is a documented spec, n_trials untouched by this
diagnostic). Any spec change after seeing results is a NEW pre-registration.
=================================================================================

Secondary descriptives (no pass/fail): calibration by 10c band with Wilson
intervals, MZ by year, volume-weighted Q2 variant, by-series counts.

Usage (after `uv run python scripts/ingest.py --dataset kalshi_hist_politics
--start 2021-01-01 --full`):
    uv run python research/diagnostics/kalshi_political_underconfidence.py
Artifact: diagnostics/kalshi_political_underconfidence.json
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

from production.core.lake import Lake  # noqa: E402
from _common import cluster_mean_test, cluster_ols, post_fee_return  # noqa: E402
from kalshi_longshot_fade import ENTRY_WINDOW_DAYS, build_entries, load_panel  # noqa: E402

# The 12 macro series (and legacy aliases) measured by diagnostics #1-#3 — the
# domain-conditionality guard excludes them regardless of how they were ingested.
_MACRO = frozenset({
    "KXCPI", "KXCPIYOY", "KXCPICORE", "KXCPICOREYOY", "KXPCECORE", "KXPAYROLLS",
    "KXUSNFP", "KXU3", "KXJOBLESS", "KXFED", "KXFEDDECISION", "KXGDP",
    "CPI", "CPIYOY", "CPICORE", "CPICOREYOY", "PCECORE", "PAYROLLS", "USNFP",
    "U3", "JOBLESS", "FED", "FEDDECISION", "GDP",
})
FAVORITE_BUCKET = (0.70, 0.95)
MIN_LIFE_VOLUME = 1000.0
RECENCY_MONTHS = 18


def main() -> int:
    bars, st = load_panel()
    ent = build_entries(bars, st)          # same T-1 convention as diagnostic #1
    ent = ent[~ent["series_ticker"].isin(_MACRO)].reset_index(drop=True)
    out: dict = {"pre_registration": {
        "Q1": "MZ psi > 0, |t| >= 1.96 (underconfidence sign)",
        "Q2": "YES at [0.70,0.95] post-fee mean > 0, t >= +1.645",
        "Q3": "Q2 point estimate > 0 in last 18 months",
        "filters": {"min_life_volume": MIN_LIFE_VOLUME,
                    "entry_window_days": ENTRY_WINDOW_DAYS,
                    "macro_series_excluded": True},
    }, "n_markets": int(len(ent)),
       "n_events": int(ent["event_key"].nunique()) if len(ent) else 0,
       "settle_range": [str(ent["settle_date"].min()), str(ent["settle_date"].max())]
       if len(ent) else None}

    if len(ent) < 200:
        out["verdict"] = "INSUFFICIENT DATA (<200 qualifying non-macro markets)"
        print(json.dumps(out, indent=2, default=str))
        return 1

    X = np.column_stack([np.ones(len(ent)), ent["entry_yes"].to_numpy()])
    mz = cluster_ols((ent["terminal"] - ent["entry_yes"]).to_numpy(), X,
                     ent["event_key"].to_numpy())
    psi, t_psi = float(mz["beta"][1]), float(mz["t"][1])
    out["Q1"] = {"psi": psi, "t": t_psi, "n": mz["n"], "n_clusters": mz["n_clusters"],
                 "pass": bool(psi > 0 and abs(t_psi) >= 1.96)}

    fav = ent[(ent["entry_yes"] >= FAVORITE_BUCKET[0])
              & (ent["entry_yes"] <= FAVORITE_BUCKET[1])]
    fav_ret = fav.apply(lambda r: post_fee_return(r["entry_yes"], r["terminal"]),
                        axis=1)
    q2 = cluster_mean_test(fav_ret, fav["event_key"]) if len(fav) else None
    out["Q2"] = {**(q2 or {}), "pass": bool(q2 and q2["mean"] > 0 and q2["t"] >= 1.645)}

    cutoff = ent["settle_date"].max() - pd.DateOffset(months=RECENCY_MONTHS)
    fav3 = fav[fav["settle_date"] >= cutoff]
    fav3_ret = fav3.apply(lambda r: post_fee_return(r["entry_yes"], r["terminal"]),
                          axis=1)
    q3 = cluster_mean_test(fav3_ret, fav3["event_key"]) if len(fav3) else None
    out["Q3"] = {**(q3 or {}), "pass": bool(q3 and q3["mean"] > 0)}

    out["verdict"] = ("PASS" if out["Q1"]["pass"] and out["Q2"]["pass"]
                      and out["Q3"]["pass"] else "FAIL")

    # ------------------------------------------------- secondary descriptives
    bands = np.arange(0.0, 1.01, 0.10)
    calib = []
    for lo, hi in zip(bands[:-1], bands[1:]):
        b = ent[(ent["entry_yes"] > lo) & (ent["entry_yes"] <= hi)]
        if len(b) == 0:
            continue
        n, k = len(b), float(b["terminal"].sum())
        p_hat = k / n
        z = 1.96
        denom = 1 + z * z / n
        centre = (p_hat + z * z / (2 * n)) / denom
        half = z * np.sqrt(p_hat * (1 - p_hat) / n + z * z / (4 * n * n)) / denom
        calib.append({"band": f"{lo:.1f}-{hi:.1f}", "n": n,
                      "mean_price": float(b["entry_yes"].mean()),
                      "win_rate": p_hat,
                      "wilson_lo": centre - half, "wilson_hi": centre + half,
                      "mean_postfee_ret": float(b.apply(
                          lambda r: post_fee_return(r["entry_yes"], r["terminal"]),
                          axis=1).mean())})
    out["calibration"] = calib

    by_year = []
    for yr, g in ent.groupby(ent["settle_date"].dt.year):
        if len(g) < 60:
            continue
        Xg = np.column_stack([np.ones(len(g)), g["entry_yes"].to_numpy()])
        r = cluster_ols((g["terminal"] - g["entry_yes"]).to_numpy(), Xg,
                        g["event_key"].to_numpy())
        by_year.append({"year": int(yr), "psi": float(r["beta"][1]),
                        "t_psi": float(r["t"][1]), "n": r["n"]})
    out["mz_by_year"] = by_year
    out["by_series_top"] = {s: int(n) for s, n in
                            ent["series_ticker"].value_counts().head(20).items()}
    out["n_favorite_bucket"] = int(len(fav))

    artifact = REPO_ROOT / "diagnostics" / "kalshi_political_underconfidence.json"
    artifact.write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps(out, indent=2, default=str))
    print(f"\nartifact -> {artifact}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
