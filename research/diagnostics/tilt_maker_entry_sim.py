"""Tilt maker-entry simulation — PRE-REGISTERED at wiki commit 6a87689
("PRE-REGISTERED (tilt maker-entry, ...)"), written BEFORE this script existed
or the tick tape was read for this purpose. Implements that registration
exactly:

- Eligibility (universe): the production tilt's eligibility VERBATIM, reused by
  DIRECT IMPORT of :func:`production.events.signals.political_favorite_tilt`
  (called at its own defaults: kalshi, category=="politics", yes_price in
  [0.70,0.95] closed, days-to-close <= 10, NaT close_time fails closed) rather
  than a hand-rolled re-derivation — this is the strongest form of "mirror the
  production eligibility exactly" and carries zero drift risk from the real
  signal. Applied to the curated ``event_markets_hist`` panel's ``row_type ==
  "bar"`` rows only (settlement rows carry yes_price in {0.0, 1.0}, outside the
  [0.70,0.95] bucket, so they are inert to this filter regardless — excluded
  explicitly for clarity). Vendor-deduped the same way
  ``kalshi_maker_fill_sim.load_curated_kalshi_bars`` dedupes: sort by
  (obs_date, instrument_id, available_from, ingested_at), keep last. Markets
  must also have settled (a ``row_type=="settlement"`` row with
  ``result in ("yes","no")`` — "present/parseable"; the loader only ever
  emits a settlement row for a parseable binary result, so this is a
  belt-and-braces re-check, not a new filter). One entry per market: if a
  market is eligible on multiple days, take the FIRST such day (mirrors a
  one-shot entry decision).

- Baseline arm: taker entry at that first-eligible-day close p_t, held to
  settlement, full taker fee once at entry —
  ``_common.post_fee_return(p_t, terminal)`` (Whelan eq. 3).

- Maker arm: rest a YES bid AT p_t from the end of the eligible obs_date
  (window opens at ``obs_date + 1 day 00:00 UTC``, the identical boundary
  convention ``kalshi_maker_fill_sim`` uses). Fill = first later trade with
  ``taker_side == "no"`` AND ``yes_price`` STRICTLY less than p_t (same
  conservative, queue-free, price-priority rule as the maker-fill sim's
  DOWN-fade case — fill price is ALWAYS p_t, never the crossing trade's own
  print). Search window closes at ``min(3rd-obs-day-out bar's close + 1 day,
  market close_time)`` — "3 obs days" is a POSITIONAL offset in the market's
  own sorted daily-bar series (mirrors the maker-fill sim's bar-indexed
  window/exit logic), not a calendar offset; "market end" is the market's own
  ``close_time``. If filled: commission ``c = 0.25 * taker_fee(p_t)`` (25%
  maker fraction, 2026-07-07 schedule, matching maker_upper_bound.py's
  MAKER_FRAC and kalshi_maker_fill_sim's), entry price p_t (price priority):
  ``return = (terminal - p_t - c) / (p_t + c)``.

  If unfilled within that window: FALLBACK to a taker entry at the close 3
  obs days after the eligible day (same positional bar offset), or the last
  available close if fewer than 3 remain (counted as a truncation), full
  taker fee at that price — ``post_fee_return(p_fallback, terminal)``.
  Fallback markets may have drifted outside [0.70,0.95]; per the registration
  the arm mirrors a COMMITTED strategy, so no re-filter is applied.

  Degenerate case ("the market has settled before the fallback close
  exists"): if the eligible day's own bar is the LAST bar row recorded for
  that market (zero bar rows after it — trading effectively stopped before
  any next daily print, e.g. settling overnight), there is no distinct
  "close 3 obs days after" and no distinct "last available close" to fall
  back to either (the only candidate is the entry bar itself, which is not a
  fallback). The registration is explicit for exactly this case: the
  position was simply never opened -> maker-arm return = 0.0 for that
  market, counted separately (n_never_opened).

- Paired difference per market: maker_arm_return - baseline_return, over ALL
  eligible+settled markets (never-opened markets contribute their real
  paired cost, 0 - baseline_return, per the registration's "return 0 for
  that arm" instruction — this is a real economic outcome of the maker
  strategy's execution-failure mode, not an exclusion). Stats via
  ``_common.cluster_mean_test``, clusters = event_key (CR1).

- Verdict: Q1 paired mean > 0 with cluster t >= 2.0; Q2 fill rate (n_filled /
  n_markets) >= 0.30; Q3 paired mean point-estimate > 0 restricted to markets
  whose settle_date falls within 12 months (``pd.DateOffset(months=12)``,
  the same helper ``kalshi_political_underconfidence.py`` uses for its own
  "N months" recency cutoff) of the max settle_date in the sample. FINAL =
  Q1 and Q2 and Q3. ALL pass -> execution-layer re-spec of the events sleeve
  (no trial burned); any fail -> negative filed. Zero n_trials impact either
  way (per the registration).

Also reports each arm's unconditional mean net return over all markets, and
the filled-subset's baseline-vs-maker means (the adverse-selection read: does
resting at p_t and getting dipped into predict the market subsequently
losing?), plus a filled-vs-unfilled terminal (settlement) win-rate comparison
for the same read.

Worked examples: printed deterministically, sorted by (instrument_id,
eligible_date) — the first 3 FILLED markets, plus the first FALLBACK market
(if any exist), clearly labeled per case (the registration's parenthetical
"sorted, first 3 with a fill and first 1 fallback if any" names both subsets
explicitly, so both are printed rather than only three total rows).

Depends on the one-time tick cache built by kalshi_extract_ticks.py (see
TICK_CACHE_PATH below) — already built (6.12M rows), reused unchanged.

Usage: uv run python research/diagnostics/tilt_maker_entry_sim.py
Verdict-only output; no artifact is written. Does not touch production/,
configs/, tests/, or data/.
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
from production.events.signals import political_favorite_tilt  # noqa: E402
from _common import cluster_mean_test, post_fee_return, taker_fee  # noqa: E402

# Written by research/diagnostics/kalshi_extract_ticks.py; deliberately outside
# data/ and outside the repo (scratch derived from the raw lake, not curated
# data and not a report artifact). Reused unchanged from kalshi_maker_fill_sim.
TICK_CACHE_PATH = Path(
    r"C:\Users\benja\AppData\Local\Temp\claude\C--Users-benja-Downloads-trader-improved"
    r"\4da31de3-3dbd-44d1-a6d2-f10753b484ce\scratchpad\kalshi_ticks.parquet"
)

INSTRUMENT_PREFIX = "EV:kalshi:"
MAKER_FRAC = 0.25          # maker fee = 25% of taker (2026-07-07 schedule)
FALLBACK_OBS_DAYS = 3      # "3 obs days" per the registration, positional bar offset
RECENCY_MONTHS = 12        # Q3 "most recent 12 months of settlements"


# ---------------------------------------------------------------- panel build
def load_curated() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Vendor-deduped curated Kalshi event-markets panel, split into bar and
    settlement rows. Dedup is identical to
    ``kalshi_maker_fill_sim.load_curated_kalshi_bars``: sort by (obs_date,
    instrument_id, available_from, ingested_at), keep last -- "vendor-deduped
    like the other readers."
    """
    lake = Lake()
    df = lake.read_curated("event_markets_hist", "events")
    df = (df.sort_values(["obs_date", "instrument_id", "available_from", "ingested_at"])
            .drop_duplicates(["obs_date", "instrument_id"], keep="last"))
    df["obs_date"] = pd.to_datetime(df["obs_date"])
    bars = df[df["row_type"] == "bar"].copy()
    st = df[df["row_type"] == "settlement"].copy()
    st = st[st["result"].isin(["yes", "no"])]           # "result present/parseable"
    st = st.drop_duplicates(subset=["instrument_id"], keep="last")
    return bars, st


def build_eligible_markets(bars: pd.DataFrame, st: pd.DataFrame) -> pd.DataFrame:
    """One row per market: first eligible day (production eligibility, exact
    defaults), p_t, and settlement outcome. Only settled markets are kept.
    """
    elig = political_favorite_tilt(bars)     # [obs_date, instrument_id, value]
    if elig.empty:
        return pd.DataFrame()
    elig = elig.sort_values(["instrument_id", "obs_date"], kind="stable")
    first = (elig.groupby("instrument_id", as_index=False).first()
                 [["instrument_id", "obs_date"]]
                 .rename(columns={"obs_date": "eligible_date"}))

    st_idx = st.set_index("instrument_id")
    first = first[first["instrument_id"].isin(st_idx.index)].reset_index(drop=True)

    b = bars.merge(first, left_on=["instrument_id", "obs_date"],
                   right_on=["instrument_id", "eligible_date"], how="inner")
    b = b.rename(columns={"yes_price": "p_t"})
    b["terminal"] = b["instrument_id"].map(st_idx["yes_price"]).astype(float)
    b["settle_date"] = b["instrument_id"].map(st_idx["obs_date"])
    b["market_ticker"] = b["instrument_id"].where(
        ~b["instrument_id"].str.startswith(INSTRUMENT_PREFIX),
        b["instrument_id"].str.slice(len(INSTRUMENT_PREFIX)))
    cols = ["instrument_id", "market_ticker", "event_key", "eligible_date",
            "p_t", "close_time", "terminal", "settle_date"]
    return b[cols].drop_duplicates(subset=["instrument_id"]).reset_index(drop=True)


def group_bars_by_instrument(bars: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {iid: g.sort_values("obs_date").reset_index(drop=True)
            for iid, g in bars.groupby("instrument_id", sort=False)}


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


# ------------------------------------------------------------------- fill/fallback
def find_fallback_bar(bars_g: pd.DataFrame, idx: int, obs_days: int = FALLBACK_OBS_DAYS):
    """The bar `obs_days` positions after `idx` in this market's own sorted
    bar series, or the last available bar if fewer remain (truncated=True).
    Returns None if there is no bar at all after `idx` (degenerate: "the
    market has settled before the fallback close exists" -> never opened).
    """
    last_idx = len(bars_g) - 1
    if idx >= last_idx:
        return None
    fb_idx = min(idx + obs_days, last_idx)
    truncated = (idx + obs_days) > last_idx
    row = bars_g.iloc[fb_idx]
    return row["obs_date"], float(row["yes_price"]), truncated


def find_fill(ticks_by_market: dict[str, pd.DataFrame], market_ticker: str, p_t: float,
              window_start: pd.Timestamp, window_end: pd.Timestamp):
    """First trade in (window_start, window_end) with taker_side=="no" AND
    yes_price STRICTLY < p_t. Fill price is always p_t (price priority)."""
    sub = ticks_by_market.get(market_ticker)
    if sub is None or sub.empty:
        return None
    w = sub[(sub["created_time"] > window_start) & (sub["created_time"] < window_end)]
    if w.empty:
        return None
    cand = w[(w["taker_side"] == "no") & (w["yes_price"] < p_t)]
    if cand.empty:
        return None
    return cand.iloc[0]


# ------------------------------------------------------------------------ sim
def simulate(markets: pd.DataFrame, bars_by_instrument: dict[str, pd.DataFrame],
             ticks_by_market: dict[str, pd.DataFrame]) -> pd.DataFrame:
    results = []
    for row in markets.itertuples(index=False):
        bars_g = bars_by_instrument[row.instrument_id]
        obs_dates = bars_g["obs_date"]
        matches = np.flatnonzero((obs_dates == row.eligible_date).to_numpy())
        idx = int(matches[0])  # exact match guaranteed: eligible_date came from bars_g itself

        rec = {"instrument_id": row.instrument_id, "market_ticker": row.market_ticker,
               "event_key": row.event_key, "eligible_date": row.eligible_date,
               "p_t": row.p_t, "terminal": row.terminal, "settle_date": row.settle_date,
               "baseline_return": post_fee_return(row.p_t, row.terminal),
               "filled": False, "never_opened": False, "truncated_fallback": False}

        fallback = find_fallback_bar(bars_g, idx)
        if fallback is None:
            rec["never_opened"] = True
            rec["maker_return"] = 0.0
        else:
            fb_date, fb_price, truncated = fallback
            window_start = pd.Timestamp(row.eligible_date).normalize() + pd.Timedelta(days=1)
            window_end_by_bars = pd.Timestamp(fb_date).normalize() + pd.Timedelta(days=1)
            close_naive = pd.Timestamp(row.close_time)
            if close_naive.tzinfo is not None:
                close_naive = close_naive.tz_convert("UTC").tz_localize(None)
            window_end = min(window_end_by_bars, close_naive)

            fill = find_fill(ticks_by_market, row.market_ticker, row.p_t,
                             window_start, window_end)
            if fill is not None:
                rec["filled"] = True
                c = MAKER_FRAC * taker_fee(row.p_t)
                rec["maker_return"] = (row.terminal - row.p_t - c) / (row.p_t + c)
                rec["fill_trade_id"] = fill["trade_id"]
                rec["fill_created_time"] = pd.Timestamp(fill["created_time"])
                rec["fill_yes_price"] = float(fill["yes_price"])
                rec["fill_taker_side"] = str(fill["taker_side"])
            else:
                rec["truncated_fallback"] = truncated
                rec["fallback_date"] = fb_date
                rec["fallback_price"] = fb_price
                rec["maker_return"] = post_fee_return(fb_price, row.terminal)

        rec["paired_diff"] = rec["maker_return"] - rec["baseline_return"]
        results.append(rec)
    return pd.DataFrame(results)


def main() -> int:
    bars, st = load_curated()
    markets = build_eligible_markets(bars, st)
    n = len(markets)
    print(f"eligible+settled markets: n={n}  events={markets['event_key'].nunique() if n else 0}")
    if n == 0:
        print("FINAL: FAIL (no eligible+settled markets)")
        return 1

    bars_by_instrument = group_bars_by_instrument(bars)
    ticks = load_ticks()
    print(f"tick cache: {len(ticks)} rows, {ticks['market_ticker'].nunique()} market_tickers")
    ticks_by_market = group_ticks_by_market(ticks)

    sim = simulate(markets, bars_by_instrument, ticks_by_market)

    n_filled = int(sim["filled"].sum())
    n_never_opened = int(sim["never_opened"].sum())
    n_fallback = n - n_filled - n_never_opened
    n_truncated = int(sim["truncated_fallback"].sum())
    fill_rate = n_filled / n

    print(f"\nn markets={n}  n filled={n_filled} (fill_rate={fill_rate:.4f})  "
          f"n fallback={n_fallback}  n never-opened={n_never_opened}  "
          f"n truncated fallback closes={n_truncated}")

    # ------------------------------------------------------------------- Q1
    q1 = cluster_mean_test(sim["paired_diff"], sim["event_key"])
    q1_pass = (q1["mean"] > 0) and (q1["t"] >= 2.0)
    print(f"\nQ1 paired improvement: mean={q1['mean']:+.4f} t={q1['t']:.2f} "
          f"(clusters={q1['n_clusters']}) -> {'PASS' if q1_pass else 'FAIL'} "
          f"(threshold: mean>0 and t>=2.0)")

    # ------------------------------------------------------------------- Q2
    q2_pass = fill_rate >= 0.30
    print(f"Q2 fill rate: {fill_rate:.4f} -> {'PASS' if q2_pass else 'FAIL'} "
          f"(threshold >= 0.30)")

    # ------------------------------------------------------------------- Q3
    cutoff = sim["settle_date"].max() - pd.DateOffset(months=RECENCY_MONTHS)
    recent = sim[sim["settle_date"] >= cutoff]
    if len(recent):
        q3 = cluster_mean_test(recent["paired_diff"], recent["event_key"])
        q3_pass = q3["mean"] > 0
    else:
        q3 = {"mean": float("nan"), "t": float("nan"), "n_clusters": 0}
        q3_pass = False
    print(f"Q3 recent-12m paired improvement: n={len(recent)} mean={q3['mean']:+.4f} "
          f"t={q3['t']:.2f} -> {'PASS' if q3_pass else 'FAIL'} (threshold: point estimate > 0; "
          f"cutoff={cutoff.date()}, max settle_date={sim['settle_date'].max().date()})")

    final = q1_pass and q2_pass and q3_pass
    print(f"\nFINAL: {'PASS -> execution-layer re-spec of the events sleeve (no trial)' if final else 'FAIL -> negative filed'}"
          f"  (zero n_trials impact either way)")

    # -------------------------------------------------------- descriptives
    baseline_mean = sim["baseline_return"].mean()
    maker_mean = sim["maker_return"].mean()
    print(f"\nunconditional mean net return -- baseline arm: {baseline_mean:+.4f}   "
          f"maker arm: {maker_mean:+.4f}")

    filled = sim[sim["filled"]]
    if len(filled):
        fb_base = filled["baseline_return"].mean()
        fb_maker = filled["maker_return"].mean()
        print(f"filled-subset (n={len(filled)}) means -- baseline: {fb_base:+.4f}   "
              f"maker: {fb_maker:+.4f}   (adverse-selection read: maker-baseline = "
              f"{fb_maker - fb_base:+.4f})")

    unfilled = sim[~sim["filled"]]
    filled_winrate = float(filled["terminal"].mean()) if len(filled) else float("nan")
    unfilled_winrate = float(unfilled["terminal"].mean()) if len(unfilled) else float("nan")
    print(f"terminal (settlement) win-rate -- filled subset: {filled_winrate:.4f}   "
          f"unfilled subset (fallback+never-opened): {unfilled_winrate:.4f}   "
          f"delta (filled-unfilled): {filled_winrate - unfilled_winrate:+.4f}")

    # ---------------------------------------------------------- worked examples
    sorted_sim = sim.sort_values(["instrument_id", "eligible_date"], kind="stable")
    fill_examples = sorted_sim[sorted_sim["filled"]].head(3)
    fallback_examples = sorted_sim[(~sorted_sim["filled"]) & (~sorted_sim["never_opened"])].head(1)

    print(f"\n=== worked examples: {len(fill_examples)} filled + "
          f"{len(fallback_examples)} fallback (hand verification) ===")
    for _, r in fill_examples.iterrows():
        print(f"\n[FILLED] market_ticker: {r['market_ticker']}")
        print(f"  eligible_date: {r['eligible_date']}   p_t: {r['p_t']:.4f}")
        print(f"  fill trade: trade_id={r['fill_trade_id']} "
              f"created_time={r['fill_created_time']} "
              f"yes_price={r['fill_yes_price']:.4f} taker_side={r['fill_taker_side']}")
        print(f"  terminal: {r['terminal']:.1f}")
        print(f"  baseline_return: {r['baseline_return']:+.4f}   "
              f"maker_return: {r['maker_return']:+.4f}   "
              f"paired_diff: {r['paired_diff']:+.4f}")
    for _, r in fallback_examples.iterrows():
        print(f"\n[FALLBACK] market_ticker: {r['market_ticker']}")
        print(f"  eligible_date: {r['eligible_date']}   p_t: {r['p_t']:.4f}")
        print(f"  fallback close: date={r['fallback_date']} price={r['fallback_price']:.4f} "
              f"truncated={r['truncated_fallback']}")
        print(f"  terminal: {r['terminal']:.1f}")
        print(f"  baseline_return: {r['baseline_return']:+.4f}   "
              f"maker_return: {r['maker_return']:+.4f}   "
              f"paired_diff: {r['paired_diff']:+.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
