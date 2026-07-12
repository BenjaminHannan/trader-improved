"""Iteration 9 pre-registered diagnostic (wiki commit a621ce2): Kalshi post-move drift.

Does binary-market under-reaction drift (Angelini-De Angelis, sports beta=0.63)
exist in our politics/econ panel? Spec is FIXED by the pre-registration:

- sample: settled kalshi markets, market-days with >= 5 days to close and
  pre-move yes_price in [0.10, 0.90];
- trigger: |1-day delta yes_price| >= 0.05 (consecutive daily bars only);
- response: signed continuation over the next 3 obs days;
- Q1 existence: mean >= +0.01, cluster t >= 2.0 (clusters=event_key), n >= 1000;
- Q2 economics: mean net of the Whelan taker curve charged twice > 0;
- Q3 capacity: above-median-volume half alone has cluster t >= 1.5.

All pass -> fund ONE trial (n_trials 24->25) at exactly these parameters.
Any fail -> negative filed, no trial. Verdict-only output.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from production.core.lake import Lake
from research.diagnostics._common import cluster_mean_test, taker_fee

MOVE = 0.05
BAND = (0.10, 0.90)
HOLD = 3
MIN_DTC_DAYS = 5


def build_triggers() -> pd.DataFrame:
    lake = Lake()
    df = lake.read_curated("event_markets_hist", "events")
    df = (df.sort_values(["obs_date", "instrument_id", "available_from", "ingested_at"])
            .drop_duplicates(["obs_date", "instrument_id"], keep="last"))
    df = df[df["venue"].astype(str).str.lower() == "kalshi"]
    df = df.dropna(subset=["yes_price", "close_time"])
    df["obs_date"] = pd.to_datetime(df["obs_date"])
    close = pd.to_datetime(df["close_time"], utc=True, errors="coerce").dt.tz_localize(None)
    df["days_to_close"] = (close.dt.normalize() - df["obs_date"]).dt.days

    rows = []
    for iid, g in df.groupby("instrument_id", sort=False):
        g = g.sort_values("obs_date").reset_index(drop=True)
        if len(g) < HOLD + 2:
            continue
        p = g["yes_price"].to_numpy(dtype=float)
        d = g["obs_date"].to_numpy()
        gap1 = (d[1:] - d[:-1]).astype("timedelta64[D]").astype(int)
        for t in range(1, len(g) - HOLD):
            if gap1[t - 1] != 1:
                continue  # 1-day move means consecutive daily bars, per spec
            move = p[t] - p[t - 1]
            if abs(move) < MOVE or not (BAND[0] <= p[t - 1] <= BAND[1]):
                continue
            if g["days_to_close"].iat[t] < MIN_DTC_DAYS:
                continue
            # exit HOLD obs rows later; bound calendar gaps so "3 days" stays 3-ish
            gap_exit = int((d[t + HOLD] - d[t]).astype("timedelta64[D]").astype(int))
            if gap_exit > HOLD + 2:
                continue
            sign = 1.0 if move > 0 else -1.0
            cont = (p[t + HOLD] - p[t]) * sign
            fees = taker_fee(p[t]) + taker_fee(p[t + HOLD])
            rows.append({"instrument_id": iid,
                         "event_key": g["event_key"].iat[t],
                         "obs_date": g["obs_date"].iat[t],
                         "year": g["obs_date"].iat[t].year,
                         "cont": cont, "net": cont - fees,
                         "volume": g["volume"].iat[t]})
    return pd.DataFrame(rows)


def main() -> int:
    trig = build_triggers()
    n = len(trig)
    print(f"triggers: n={n}  markets={trig['instrument_id'].nunique() if n else 0}  "
          f"events={trig['event_key'].nunique() if n else 0}")
    if n == 0:
        print("FINAL: FAIL (no triggers)")
        return 1

    by_year = trig.groupby("year").agg(n=("cont", "size"), mean_cont=("cont", "mean"),
                                       mean_net=("net", "mean"))
    print(by_year.round(4).to_string())

    q1 = cluster_mean_test(trig["cont"], trig["event_key"])
    q1_pass = (q1["mean"] >= 0.01) and (q1["t"] >= 2.0) and (n >= 1000)
    print(f"\nQ1 existence: mean={q1['mean']:+.4f} t={q1['t']:.2f} "
          f"(clusters={q1.get('n_clusters', '?')}) n={n} -> {'PASS' if q1_pass else 'FAIL'}")

    q2 = cluster_mean_test(trig["net"], trig["event_key"])
    q2_pass = q2["mean"] > 0
    print(f"Q2 net-of-fee (taker x2): mean={q2['mean']:+.4f} t={q2['t']:.2f} "
          f"-> {'PASS' if q2_pass else 'FAIL'}")

    liq = trig[trig["volume"] > trig["volume"].median()]
    q3 = cluster_mean_test(liq["cont"], liq["event_key"])
    q3_pass = q3["t"] >= 1.5
    print(f"Q3 above-median-volume half: n={len(liq)} mean={q3['mean']:+.4f} "
          f"t={q3['t']:.2f} -> {'PASS' if q3_pass else 'FAIL'}")

    final = q1_pass and q2_pass and q3_pass
    print(f"\nFINAL: {'PASS — fund ONE trial (n_trials 24->25) at these exact '
          'parameters' if final else 'FAIL — negative filed, no trial burned'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
