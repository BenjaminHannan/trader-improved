"""Maker-fill simulation — PRE-REGISTERED at wiki commit c9d47fd ("PRE-REGISTERED
(maker-fill simulation...)"), written BEFORE this script existed or the tick tape
was read for fills. Implements that registration exactly:

- Triggers: iteration 9's exact spec, reused unchanged from
  kalshi_postmove_drift.build_triggers (|1d delta yes_price| >= 5c, prior close in
  [0.10,0.90], >=5 obs days to close, consecutive daily bars, settled kalshi
  markets). MOVE/BAND/HOLD/MIN_DTC_DAYS are IMPORTED from that module, not
  redefined, so this sim cannot silently drift from the registered parameters.
  That function only returns the already sign-adjusted `cont`/`net` columns, so
  its trigger-selection loop is copied VERBATIM below
  (`build_triggers_with_fill_fields`) with three additive fields the fill sim
  needs and the original does not expose: `sign` (the raw move sign), `entry_price`
  (p_t, the resting order's level), and `window_end_date` (the market's own
  3rd-bar-out date, reused unchanged as the fill window's calendar bound). No
  filter, threshold, or parameter differs from the original — see the
  `--fidelity check` printed at runtime, which re-runs the original
  build_triggers() and confirms identical n / cont-sum / net-sum.
- Entry: rest on the fade side at the post-move close p_t. Move UP (sign +1):
  fade SELLS yes at p_t -> resting YES ASK -> FILL at the first later trade with
  taker_side=="yes" AND yes_price STRICTLY > p_t. Move DOWN (sign -1): fade BUYS
  yes at p_t -> resting YES BID -> FILL at the first later trade with
  taker_side=="no" AND yes_price STRICTLY < p_t. At-level prints are ignored (no
  queue-position assumption) — conservative, queue-free. The FILL PRICE is
  ALWAYS p_t (price priority), never the crossing trade's own print price.
  Window: (trigger obs_date + 1 day 00:00 UTC, through the market's own 3rd
  next daily-bar date + 1 day 00:00), i.e. strictly after the close that set
  p_t, through the end of the calendar day of the 3rd next daily bar the
  original trigger definition already anchors on (`window_end_date` above).
  Window also naturally ends at the market's last available tick (there is
  nothing to find past that regardless of the nominal bound).
- Exit: TAKER at the daily close 3 obs days after the FILL's own obs date
  (looked up in the same per-instrument sorted daily-bar series the trigger
  loop builds) — NOT 3 days after the trigger date. If fewer than 3 daily bars
  remain after the fill's bar, the exit uses the market's last available close
  (truncated exit; counted and reported).
- Net per contract, price space: UP-fade net = (p_t - p_exit) -
  0.25*taker_fee(p_t) - taker_fee(p_exit); DOWN-fade net = (p_exit - p_t) -
  same fees. 0.25x is the maker-fee fraction (25% of taker, 2026-07-07
  schedule, matching maker_upper_bound.py's MAKER_FRAC) charged on the maker
  entry; the exit pays the full taker fee.
- Q1 capacity: fill rate >= 15% of triggers.
- Q2 economics: filled-subset mean net > 0 AND event-clustered t >= 2.0.
- Q3 robustness: filled subset restricted to above-median trigger-day volume
  (median computed across ALL triggers, same convention as
  kalshi_postmove_drift's Q3) has event-clustered t >= 1.5.
- FINAL PASS iff Q1 and Q2 and Q3 all pass -> fund ONE trial (n_trials 24->25)
  at exactly these parameters. Any fail -> negative filed, zero trials.

Depends on the one-time tick cache built by kalshi_extract_ticks.py (see
TICK_CACHE_PATH below) — run that first if the cache is missing.

Usage: uv run python research/diagnostics/kalshi_maker_fill_sim.py
Verdict-only output; no artifact is written.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "research" / "diagnostics"))

from production.core.lake import Lake  # noqa: E402
from _common import cluster_mean_test, taker_fee  # noqa: E402
from kalshi_postmove_drift import (  # noqa: E402
    MOVE, BAND, HOLD, MIN_DTC_DAYS, build_triggers as build_triggers_orig,
)

# Written by research/diagnostics/kalshi_extract_ticks.py; deliberately outside
# data/ and outside the repo (scratch derived from the raw lake, not curated
# data and not a report artifact).
TICK_CACHE_PATH = Path(
    r"C:\Users\benja\AppData\Local\Temp\claude\C--Users-benja-Downloads-trader-improved"
    r"\4da31de3-3dbd-44d1-a6d2-f10753b484ce\scratchpad\kalshi_ticks.parquet"
)

MAKER_FRAC = 0.25            # maker fee = 25% of taker (2026-07-07 schedule)
INSTRUMENT_PREFIX = "EV:kalshi:"


# ---------------------------------------------------------------- trigger build
def load_curated_kalshi_bars() -> pd.DataFrame:
    """Identical preprocessing to kalshi_postmove_drift.build_triggers's `df`."""
    lake = Lake()
    df = lake.read_curated("event_markets_hist", "events")
    df = (df.sort_values(["obs_date", "instrument_id", "available_from", "ingested_at"])
            .drop_duplicates(["obs_date", "instrument_id"], keep="last"))
    df = df[df["venue"].astype(str).str.lower() == "kalshi"]
    df = df.dropna(subset=["yes_price", "close_time"])
    df["obs_date"] = pd.to_datetime(df["obs_date"])
    close = pd.to_datetime(df["close_time"], utc=True, errors="coerce").dt.tz_localize(None)
    df["days_to_close"] = (close.dt.normalize() - df["obs_date"]).dt.days
    return df


def build_triggers_with_fill_fields(df: pd.DataFrame):
    """VERBATIM copy of build_triggers's selection loop (MOVE/BAND/HOLD/
    MIN_DTC_DAYS imported, not redefined; same filters, same row construction)
    plus `sign`, `entry_price`, `window_end_date`. Also returns the
    per-instrument sorted daily-bar frames built along the way, unchanged, for
    the exit-price lookup (so exit reads the identical daily-bar series the
    trigger/window logic anchors on).
    """
    rows = []
    bars_by_instrument: dict[str, pd.DataFrame] = {}
    for iid, g in df.groupby("instrument_id", sort=False):
        g = g.sort_values("obs_date").reset_index(drop=True)
        bars_by_instrument[iid] = g[["obs_date", "yes_price"]].reset_index(drop=True)
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
                         "volume": g["volume"].iat[t],
                         "sign": sign,
                         "entry_price": float(p[t]),
                         "window_end_date": g["obs_date"].iat[t + HOLD]})
    return pd.DataFrame(rows), bars_by_instrument


# -------------------------------------------------------------------- tick I/O
def load_ticks() -> pd.DataFrame:
    if not TICK_CACHE_PATH.exists():
        raise FileNotFoundError(
            f"tick cache missing at {TICK_CACHE_PATH} -- run "
            "research/diagnostics/kalshi_extract_ticks.py first")
    ticks = pd.read_parquet(TICK_CACHE_PATH)
    return ticks.sort_values(["market_ticker", "created_time"]).reset_index(drop=True)


def group_ticks_by_market(ticks: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {mt: g.reset_index(drop=True) for mt, g in ticks.groupby("market_ticker", sort=False)}


# ------------------------------------------------------------------- fill/exit
def find_fill(ticks_by_market: dict[str, pd.DataFrame], market_ticker: str,
              sign: float, entry_price: float,
              window_start: pd.Timestamp, window_end: pd.Timestamp):
    """First trade in (window_start, window_end) on the opposite (aggressor)
    side, strictly through entry_price. Returns the matching tick row (a
    pandas Series) or None. Rows arrive pre-sorted by created_time within each
    market, so `.iloc[0]` after the boolean mask is the first qualifying trade."""
    sub = ticks_by_market.get(market_ticker)
    if sub is None or sub.empty:
        return None
    w = sub[(sub["created_time"] > window_start) & (sub["created_time"] < window_end)]
    if w.empty:
        return None
    if sign > 0:
        cand = w[(w["taker_side"] == "yes") & (w["yes_price"] > entry_price)]
    else:
        cand = w[(w["taker_side"] == "no") & (w["yes_price"] < entry_price)]
    if cand.empty:
        return None
    return cand.iloc[0]


def find_exit(bars_by_instrument: dict[str, pd.DataFrame], instrument_id: str,
              fill_created_time: pd.Timestamp):
    """Daily close 3 obs days after the FILL's own obs date, in the market's own
    sorted daily-bar series. Truncates to the last available bar if fewer than
    3 remain. Returns (exit_date, exit_price, truncated, exact_date_match)."""
    bars = bars_by_instrument[instrument_id]
    obs_dates = bars["obs_date"]
    fill_date = pd.Timestamp(fill_created_time).normalize()
    idx = int(obs_dates.searchsorted(fill_date, side="left"))
    exact = idx < len(bars) and obs_dates.iloc[idx] == fill_date
    if not exact:
        # Defensive fallback only: tick cache and curated bars are both built
        # from the same raw trade tapes, so every fill date should have an
        # exact daily-bar match. If this ever fires, anchor at the nearest
        # prior bar and surface a count so it can be investigated rather than
        # silently trusted.
        idx = max(0, idx - 1)
    last_idx = len(bars) - 1
    exit_idx = min(idx + HOLD, last_idx)
    truncated = (idx + HOLD) > last_idx
    row = bars.iloc[exit_idx]
    return row["obs_date"], float(row["yes_price"]), truncated, exact


def simulate(trig: pd.DataFrame, bars_by_instrument: dict[str, pd.DataFrame],
             ticks_by_market: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, int]:
    results = []
    n_exit_mismatch = 0
    for row in trig.itertuples(index=False):
        market_ticker = (row.instrument_id[len(INSTRUMENT_PREFIX):]
                          if row.instrument_id.startswith(INSTRUMENT_PREFIX)
                          else row.instrument_id)
        window_start = pd.Timestamp(row.obs_date).normalize() + pd.Timedelta(days=1)
        window_end = pd.Timestamp(row.window_end_date).normalize() + pd.Timedelta(days=1)
        rec = {"instrument_id": row.instrument_id, "market_ticker": market_ticker,
               "event_key": row.event_key, "obs_date": row.obs_date, "sign": row.sign,
               "entry_price": row.entry_price, "volume": row.volume, "filled": False}

        fill = find_fill(ticks_by_market, market_ticker, row.sign, row.entry_price,
                         window_start, window_end)
        if fill is not None:
            fill_ct = pd.Timestamp(fill["created_time"])
            exit_date, exit_price, truncated, exact = find_exit(
                bars_by_instrument, row.instrument_id, fill_ct)
            if not exact:
                n_exit_mismatch += 1
            pre_fee_edge = ((row.entry_price - exit_price) if row.sign > 0
                            else (exit_price - row.entry_price))
            fee_entry = MAKER_FRAC * taker_fee(row.entry_price)
            fee_exit = taker_fee(exit_price)
            net = pre_fee_edge - fee_entry - fee_exit
            days_to_fill = (fill_ct - window_start).total_seconds() / 86400.0
            rec.update({
                "filled": True,
                "fill_trade_id": fill["trade_id"], "fill_created_time": fill_ct,
                "fill_yes_price": float(fill["yes_price"]), "fill_count": float(fill["count"]),
                "fill_taker_side": str(fill["taker_side"]),
                "fill_is_block_trade": bool(fill["is_block_trade"]),
                "exit_date": exit_date, "exit_price": exit_price, "truncated_exit": truncated,
                "pre_fee_edge": pre_fee_edge, "net": net, "days_to_fill": days_to_fill,
            })
        results.append(rec)
    return pd.DataFrame(results), n_exit_mismatch


def main() -> int:
    df = load_curated_kalshi_bars()
    trig, bars_by_instrument = build_triggers_with_fill_fields(df)
    n = len(trig)
    print(f"triggers: n={n}  markets={trig['instrument_id'].nunique() if n else 0}  "
          f"events={trig['event_key'].nunique() if n else 0}")
    if n == 0:
        print("FINAL: FAIL (no triggers)")
        return 1

    orig = build_triggers_orig()
    fidelity_ok = (len(orig) == n
                  and np.isclose(orig["cont"].sum(), trig["cont"].sum())
                  and np.isclose(orig["net"].sum(), trig["net"].sum()))
    print(f"trigger-copy fidelity check vs kalshi_postmove_drift.build_triggers(): "
          f"n_orig={len(orig)} n_mine={n} sums_match={fidelity_ok}")

    ticks = load_ticks()
    print(f"tick cache: {len(ticks)} rows, {ticks['market_ticker'].nunique()} market_tickers")
    ticks_by_market = group_ticks_by_market(ticks)

    sim, n_exit_mismatch = simulate(trig, bars_by_instrument, ticks_by_market)
    if n_exit_mismatch:
        print(f"NOTE: {n_exit_mismatch} fills had no exact daily-bar match on the fill "
              f"date (anchored at nearest prior bar) -- see report.")

    n_filled = int(sim["filled"].sum())
    fill_rate = n_filled / n
    print(f"\nfilled: n={n_filled}/{n}  fill_rate={fill_rate:.4f}")

    q1_pass = fill_rate >= 0.15
    print(f"Q1 fill-rate capacity: fill_rate={fill_rate:.4f} -> {'PASS' if q1_pass else 'FAIL'} "
          f"(threshold >= 0.15)")

    filled = sim[sim["filled"]]
    if n_filled:
        q2 = cluster_mean_test(filled["net"], filled["event_key"])
        q2_pass = (q2["mean"] > 0) and (q2["t"] >= 2.0)
    else:
        q2 = {"mean": float("nan"), "t": float("nan"), "n_clusters": 0}
        q2_pass = False
    print(f"Q2 filled-subset net edge: mean={q2['mean']:+.4f} t={q2['t']:.2f} "
          f"(clusters={q2.get('n_clusters', '?')}) -> {'PASS' if q2_pass else 'FAIL'} "
          f"(threshold: mean>0 and t>=2.0)")

    median_volume = trig["volume"].median()
    liq_filled = filled[filled["volume"] > median_volume]
    if len(liq_filled):
        q3 = cluster_mean_test(liq_filled["net"], liq_filled["event_key"])
        q3_pass = q3["t"] >= 1.5
    else:
        q3 = {"mean": float("nan"), "t": float("nan"), "n_clusters": 0}
        q3_pass = False
    print(f"Q3 above-median-volume filled subset: n={len(liq_filled)} mean={q3['mean']:+.4f} "
          f"t={q3['t']:.2f} -> {'PASS' if q3_pass else 'FAIL'} (threshold t>=1.5; "
          f"median_volume={median_volume:.1f} across all {n} triggers)")

    final = q1_pass and q2_pass and q3_pass
    print(f"\nFINAL: {'PASS -> fund ONE trial (n_trials 24->25) at these exact '
          'parameters' if final else 'FAIL -> negative filed, no trial burned'}")

    if n_filled:
        print(f"\nfilled subset PRE-fee mean edge: {filled['pre_fee_edge'].mean():+.4f}")
        print(f"filled subset mean days-to-fill: {filled['days_to_fill'].mean():.2f}")
        print(f"n truncated exits (fewer than 3 daily bars remaining after fill): "
              f"{int(filled['truncated_exit'].sum())}")

    # ---------------------------------------------------------- worked examples
    examples = filled.sort_values(["instrument_id", "obs_date"]).head(3)
    print(f"\n=== {len(examples)} worked examples (hand verification) ===")
    for _, r in examples.iterrows():
        direction = ("UP (fade=SELL yes -> resting YES ASK)" if r["sign"] > 0
                     else "DOWN (fade=BUY yes -> resting YES BID)")
        print(f"\nmarket_ticker: {r['market_ticker']}")
        print(f"  trigger obs_date: {r['obs_date']}   direction: {direction}")
        print(f"  resting level (p_t): {r['entry_price']:.4f}")
        print(f"  fill trade: trade_id={r['fill_trade_id']} "
              f"created_time={r['fill_created_time']} "
              f"yes_price={r['fill_yes_price']:.4f} count={r['fill_count']:.2f} "
              f"taker_side={r['fill_taker_side']} is_block_trade={r['fill_is_block_trade']}")
        print(f"  exit: date={r['exit_date']}  close={r['exit_price']:.4f}  "
              f"truncated={r['truncated_exit']}")
        print(f"  net (per contract, price space): {r['net']:+.4f}   "
              f"pre-fee edge: {r['pre_fee_edge']:+.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
