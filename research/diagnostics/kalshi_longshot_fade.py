"""PRE-REGISTERED diagnostic #1 — Kalshi economic-ladder longshot-fade calibration.

Mechanism (practitioner scan idea #1; Buergi-Deng-Whelan MPRA 126350): Kalshi
contract prices show a favorite-longshot bias — sub-10c contracts lose >60% of
stake, the pattern is taker-side, and it persisted every year 2021-2025 (their
Table 9: Mincer-Zarnowitz slope 0.021-0.048, though 2025 is the weakest at p<0.1;
their Table 8 Economics column: slope 0.034*** but constant -0.978 NOT significant
— the Economics-only level is noisier than the pooled headline).

=============================== PRE-REGISTRATION ================================
Written 2026-07-10 BEFORE the backfilled data was first read. Universe: settled
binary markets in the 12 backfilled US macro series; entry = last daily-bar traded
YES price on a day in [close-10d, close-1d] (the T-1 close, Whelan's "24 hours
earlier" analogue); outcome = settlement value; per-market life volume >= 1,000
contracts (~$1,000 staked, Whelan's floor). The >=24h-life filter is vacuous here
(macro ladders live weeks). Fees per _common.taker_fee. Tests, all clustered by
event_key (CR1):

P1 (replication): mean POST-fee return of buying YES at entry <= 0.10 is <= -0.40
    and significantly negative (one-sided p < 0.05, i.e. t <= -1.645).
P2 (tradeable side): buying NO as taker against YES entries in [0.05, 0.20]
    (NO entry = 1 - yes price) earns mean post-fee return > 0 with t >= +1.645.
P3 (recency / re-rank trigger): P2's point estimate restricted to settlements in
    the most recent 12 months is > 0; if not, demote idea #1 per protocol.

Decision rule: P1&P2&P3 pass -> propose events-sleeve execution policy candidate
(no registry factor; n_trials untouched by this diagnostic). Any spec change after
seeing results = a NEW pre-registration.
=================================================================================

Secondary descriptives (no pass/fail): calibration by 10c band, Mincer-Zarnowitz
y-p = a + psi*p pooled and by year, volume-weighted variants.

Usage: uv run python research/diagnostics/kalshi_longshot_fade.py
Artifact: diagnostics/kalshi_longshot_fade.json
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
from _common import (cluster_mean_test, cluster_ols, post_fee_return,  # noqa: E402
                     pre_fee_return, taker_fee)

MIN_LIFE_VOLUME = 1000.0          # contracts ~= $1,000 staked
ENTRY_WINDOW_DAYS = 10            # entry bar must fall in [close-10d, close-1d]
LOW_BUCKET = (0.0, 0.10)          # P1
FADE_BUCKET = (0.05, 0.20)        # P2 (YES price; NO entry = 1 - p)


def load_panel() -> tuple[pd.DataFrame, pd.DataFrame]:
    lake = Lake()
    df = lake.read_curated("event_markets_hist", "events")
    bars = df[df["row_type"] == "bar"].copy()
    st = df[df["row_type"] == "settlement"].copy()
    st = st.drop_duplicates(subset=["instrument_id"], keep="last")
    return bars, st


def build_entries(bars: pd.DataFrame, st: pd.DataFrame) -> pd.DataFrame:
    """One row per market: T-1 entry price, outcome, metadata."""
    st = st.set_index("instrument_id")
    close_date = pd.to_datetime(st["close_time"], utc=True).dt.tz_localize(None).dt.normalize()
    life_vol = bars.groupby("instrument_id")["volume"].sum()

    rows = []
    for iid, g in bars.groupby("instrument_id"):
        if iid not in st.index:
            continue
        cd = close_date.get(iid)
        if pd.isna(cd):
            continue
        pre = g[(g["obs_date"] < cd)
                & (g["obs_date"] >= cd - pd.Timedelta(days=ENTRY_WINDOW_DAYS))]
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
    return ent[ent["life_volume"] >= MIN_LIFE_VOLUME].reset_index(drop=True)


def main() -> int:
    bars, st = load_panel()
    ent = build_entries(bars, st)
    out: dict = {"pre_registration": {
        "P1": "mean post-fee YES return, entry<=0.10: <= -0.40 and t <= -1.645",
        "P2": "mean post-fee NO return vs YES in [0.05,0.20]: > 0 and t >= +1.645",
        "P3": "P2 point estimate > 0 in most recent 12m of settlements",
        "filters": {"min_life_volume": MIN_LIFE_VOLUME,
                    "entry_window_days": ENTRY_WINDOW_DAYS},
    }, "n_markets": int(len(ent)),
       "n_events": int(ent["event_key"].nunique()) if len(ent) else 0,
       "settle_range": [str(ent["settle_date"].min()), str(ent["settle_date"].max())]
       if len(ent) else None}

    if len(ent) < 50:
        out["verdict"] = "INSUFFICIENT DATA (<50 qualifying markets)"
        print(json.dumps(out, indent=2, default=str))
        return 1

    # ------------------------------------------------------------------ P1
    p1 = ent[(ent["entry_yes"] > LOW_BUCKET[0]) & (ent["entry_yes"] <= LOW_BUCKET[1])]
    p1_ret = p1.apply(lambda r: post_fee_return(r["entry_yes"], r["terminal"]), axis=1)
    p1_stat = cluster_mean_test(p1_ret, p1["event_key"]) if len(p1) else None
    out["P1"] = {**(p1_stat or {}), "pass": bool(
        p1_stat and p1_stat["mean"] <= -0.40 and p1_stat["t"] <= -1.645)}

    # ------------------------------------------------------------------ P2
    p2 = ent[(ent["entry_yes"] >= FADE_BUCKET[0]) & (ent["entry_yes"] <= FADE_BUCKET[1])]
    p2_ret = p2.apply(lambda r: post_fee_return(1.0 - r["entry_yes"],
                                                1.0 - r["terminal"]), axis=1)
    p2_stat = cluster_mean_test(p2_ret, p2["event_key"]) if len(p2) else None
    out["P2"] = {**(p2_stat or {}), "pass": bool(
        p2_stat and p2_stat["mean"] > 0 and p2_stat["t"] >= 1.645)}

    # ------------------------------------------------------------------ P3
    cutoff = ent["settle_date"].max() - pd.Timedelta(days=365)
    p3 = p2[p2["settle_date"] >= cutoff]
    p3_ret = p3.apply(lambda r: post_fee_return(1.0 - r["entry_yes"],
                                                1.0 - r["terminal"]), axis=1)
    p3_stat = cluster_mean_test(p3_ret, p3["event_key"]) if len(p3) else None
    out["P3"] = {**(p3_stat or {}), "pass": bool(p3_stat and p3_stat["mean"] > 0)}

    out["verdict"] = ("PASS" if out["P1"]["pass"] and out["P2"]["pass"]
                      and out["P3"]["pass"] else "FAIL")

    # ------------------------------------------------- secondary descriptives
    bands = np.arange(0.0, 1.01, 0.10)
    calib = []
    for lo, hi in zip(bands[:-1], bands[1:]):
        b = ent[(ent["entry_yes"] > lo) & (ent["entry_yes"] <= hi)]
        if len(b) == 0:
            continue
        rets = b.apply(lambda r: post_fee_return(r["entry_yes"], r["terminal"]), axis=1)
        calib.append({
            "band": f"{lo:.1f}-{hi:.1f}", "n": int(len(b)),
            "mean_price": float(b["entry_yes"].mean()),
            "win_rate": float(b["terminal"].mean()),
            "mean_prefee_ret": float(b.apply(
                lambda r: pre_fee_return(r["entry_yes"], r["terminal"]), axis=1).mean()),
            "mean_postfee_ret": float(rets.mean()),
        })
    out["calibration"] = calib

    X = np.column_stack([np.ones(len(ent)), ent["entry_yes"].to_numpy()])
    mz = cluster_ols((ent["terminal"] - ent["entry_yes"]).to_numpy(), X,
                     ent["event_key"].to_numpy())
    out["mincer_zarnowitz"] = {"alpha": float(mz["beta"][0]), "psi": float(mz["beta"][1]),
                               "t_alpha": float(mz["t"][0]), "t_psi": float(mz["t"][1]),
                               "n": mz["n"], "n_clusters": mz["n_clusters"]}

    by_year = []
    for yr, g in ent.groupby(ent["settle_date"].dt.year):
        if len(g) < 30:
            continue
        Xg = np.column_stack([np.ones(len(g)), g["entry_yes"].to_numpy()])
        r = cluster_ols((g["terminal"] - g["entry_yes"]).to_numpy(), Xg,
                        g["event_key"].to_numpy())
        by_year.append({"year": int(yr), "psi": float(r["beta"][1]),
                        "t_psi": float(r["t"][1]), "n": r["n"]})
    out["mz_by_year"] = by_year
    out["by_series"] = {s: int(n) for s, n in
                        ent["series_ticker"].value_counts().items()}

    artifact = REPO_ROOT / "diagnostics" / "kalshi_longshot_fade.json"
    artifact.write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps(out, indent=2, default=str))
    print(f"\nartifact -> {artifact}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
