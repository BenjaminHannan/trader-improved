"""Diagnostic: the political favorite-tilt events sleeve, measured in isolation —
"does the 8th sleeve move the Sharpe?"

Runs :func:`production.events.signals.political_favorite_tilt` (edge = 0.03 on Kalshi
politics markets priced in ``[0.70, 0.95]``) through the REAL production walk-forward
P&L engine, :func:`production.events.backtest.event_sleeve_returns`, and reports the
resulting daily return stream's honest performance and portfolio impact.

Data: curated ``event_markets_hist`` (2021-07 .. 2026-07, 511 Kalshi series: 12 measured
macro series + ~499 political series, ``row_type in {"bar", "settlement"}``). The panel
carries no ``category``/``status`` columns (that is a live-loader ``event_markets``
concept) — :func:`_build_panel` derives them here, in the research script, never in
production: ``category = "politics"`` for every series NOT in the 12-series macro set
(the exact frozenset :mod:`kalshi_political_underconfidence` pre-registered and measured
against), ``"macro"`` otherwise; ``status`` is set from ``row_type`` for schema
completeness (no current production code path actually reads it).

Reuse technique — why a monkeypatch, not a rewrite: :func:`event_sleeve_returns`'s
PIT walk-forward loop (rebalance-date book construction, liquid-universe filtering,
event de-dup, fractional-Kelly sizing, entry-price bookkeeping, fee charge, holding-
period mark-to-market, resolution snapping) is exactly the machinery this measurement
needs and must not silently reimplement (drift risk). But its per-rebalance alpha input
is the module-private ``_combined_signals``, hard-wired to
``longshot_bias + resolution_convergence`` — ``political_favorite_tilt`` is not wired
into it at all. Since editing production/events/backtest.py is out of scope for this
diagnostic, this script instead REBINDS the two module-level names
``production.events.backtest.longshot_bias`` / ``...resolution_convergence`` in-process
(standard Python monkeypatch — the production file on disk is never touched) so
``_combined_signals`` sums ``political_favorite_tilt`` (aliased in as "longshot_bias")
plus an empty stub (aliased in as "resolution_convergence"). The result is the REAL
engine, unmodified, fed the tilt signal alone. This is undone before the script exits.

PIT discipline preserved end to end: entries read only ``yes_price`` from rows knowable
by the rebalance date (``available_from <= t``); outcomes resolve only through
``row_type == "settlement"`` rows, whose ``available_from`` is the settlement timestamp
— a position can never mark to an outcome before it was knowable.

Evidence-matched variant (orchestrator follow-up, pre-registered BEFORE running — the
verbatim block is recorded in the artifact under ``follow_up_pre_registration``): the
as-wired run entered any market whose price ever sat in [0.70, 0.95] (weeks before
close, where the diagnostic never claimed edge) and paid the sleeve's flat 200bp fee at
every weekly re-entry (~2x the actual hold-to-settlement taker fee for a favorite). The
variant matches the VALIDATED trade instead:

  * **nearness** — universe restricted to <= 10 days to close via ``max_days_to_close=10``,
    passed through ``event_sleeve_returns``'s ``sizing_kw`` into the real
    ``liquid_universe`` filter (the monkeypatch path does NOT bypass that filter — no
    panel-level workaround needed);
  * **fees** — the flat ``FEE_BPS`` is neutralized in-process at BOTH module globals
    that read it (``production.events.sizing.FEE_BPS``, the sizing-time edge haircut,
    and ``production.events.backtest.FEE_BPS``, the imported copy behind the
    entry-notional charge — restored on exit, file on disk untouched); the engine then
    emits GROSS returns, and the actual Whelan-imputed per-contract taker fee
    (``_common.taker_fee``, the same 100-lot imputation every prior diagnostic used) is
    applied at script level: ``taker_fee(entry)/entry`` per unit weight, charged ONCE on
    NEW weight increments per instrument at each rebalance's first hold day. No exit fee
    (hold-to-settlement; settlement is free on Kalshi). Documented approximation: a
    position dropped from the book before settlement is also uncharged on exit, and with
    zero sizing haircut the full 0.03 edge reaches Kelly, so variant positions typically
    pin the per-group cap (larger books than as-wired).

Usage:
    uv run python research/diagnostics/events_tilt_backtest.py
Artifact: diagnostics/events_tilt_backtest.json
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

from production.backtest.metrics import (  # noqa: E402
    ann_return, ann_vol, hit_rate, max_drawdown, sharpe,
)
from production.core.calendar import rebalance_grid  # noqa: E402
from production.core.config import backtest_config, universe_config  # noqa: E402
from production.core.lake import Lake  # noqa: E402
import production.events.backtest as ev_backtest  # noqa: E402
import production.events.sizing as ev_sizing  # noqa: E402
from production.events.markets import dedupe_related, liquid_universe  # noqa: E402
from production.events.signals import political_favorite_tilt  # noqa: E402
from production.events.sizing import FEE_BPS, size_event_book  # noqa: E402
from production.portfolio.allocation import apply_risk_caps, erc_weights  # noqa: E402

from _common import taker_fee  # noqa: E402
from kalshi_political_underconfidence import _MACRO  # noqa: E402

RECENCY_MONTHS = 18
ELECTION_MONTH = pd.Period("2024-11", freq="M")
ARTIFACT = REPO_ROOT / "diagnostics" / "events_tilt_backtest.json"
VARIANT_MAX_DAYS_TO_CLOSE = 10

_SIG_COLS = ["obs_date", "instrument_id", "value"]

# Recorded verbatim per the orchestrator's instruction ("record this block verbatim in
# the artifact"). Do not edit.
FOLLOW_UP_PRE_REGISTRATION = """\
Follow-up, pre-registered by the orchestrator before running (record this block verbatim in the artifact):

Your -0.67 Sharpe measured the tilt AS WIRED — entries any time a price sits in [0.70,0.95] (weeks before close, where the diagnostic never claimed edge) and the sleeve's flat 200bp fee haircut (~2x the actual hold-to-settlement taker fee for a favorite). The VALIDATED trade was T-1-entry at real Kalshi fees. Run an evidence-matched variant, same script, new artifact section "variant_evidence_matched":

1. NEARNESS: restrict tradeable markets to <= 10 days to close (liquid_universe has max_days_to_close — pass it; if your monkeypatch path bypasses that filter, apply it on the panel: close_time minus obs_date <= 10 days).
2. FEES: replace the flat 200bp haircut with the actual per-contract taker fee — research/diagnostics/_common.py::taker_fee (Whelan 100-lot imputation), charged ONCE at entry (hold-to-settlement pays no exit fee; settlement is free on Kalshi). If the fee lives inside sizing where you can't cleanly swap it, compute the return stream with the sleeve's gross returns and apply the fee at your script level, documenting exactly where the haircut was neutralized.
3. Report the same headline block (full span, last-18mo, no-2024-11, lumpiness, correlations, ERC combined-Sharpe scenario) for the variant alongside the as-wired numbers.
4. DECISION RULE (pre-registered): variant full-span Sharpe > 0 AND last-18mo Sharpe > 0 -> the mismatch is confirmed as the cause and a production re-spec of the events sleeve's entry-nearness + fee model becomes the proposal (execution layer, no n_trials). Either Sharpe <= 0 -> the tilt fails at portfolio level even evidence-matched; record it as a dead end at sizing scale despite the per-stake edge.
"""


# --------------------------------------------------------------------------- panel build
def _build_panel() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Curated ``event_markets_hist`` -> the ``event_markets``-shaped panel the production
    events machinery expects. Returns ``(panel, raw)`` — ``raw`` (with ``series_ticker``,
    ``row_type`` intact) is kept around for the settlement-lumpiness cross-check."""
    lake = Lake()
    raw = lake.read_curated("event_markets_hist", "events")
    raw = raw.copy()
    raw["obs_date"] = pd.to_datetime(raw["obs_date"])
    raw["category"] = np.where(raw["series_ticker"].isin(_MACRO), "macro", "politics")
    raw["status"] = np.where(raw["row_type"] == "settlement", "settled", "active")

    panel = raw[["obs_date", "instrument_id", "yes_price", "volume", "close_time",
                 "available_from", "venue", "category", "event_key", "question",
                 "status"]].copy()
    panel["open_interest"] = 0.0  # event_markets_hist carries no OI column; unused by
    # liquid_universe's filter conditions (volume + days-to-close only) — only stored on
    # EventMarket for informational purposes, so defaulting it is inert to the P&L.
    return panel, raw


# ------------------------------------------------------------------- monkeypatch machinery
def _empty_signal(_avail: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(columns=_SIG_COLS)


class _TiltOnlySignals:
    """Context manager: rebind production.events.backtest's signal inputs to
    (political_favorite_tilt, empty) for the duration of the `with` block, restoring the
    real functions on exit. In-process only — the file on disk is never written."""

    def __enter__(self):
        self._orig_longshot = ev_backtest.longshot_bias
        self._orig_convergence = ev_backtest.resolution_convergence
        ev_backtest.longshot_bias = political_favorite_tilt
        ev_backtest.resolution_convergence = _empty_signal
        return self

    def __exit__(self, *exc):
        ev_backtest.longshot_bias = self._orig_longshot
        ev_backtest.resolution_convergence = self._orig_convergence
        return False


def _tilt_returns(panel: pd.DataFrame, rebal_dates: pd.DatetimeIndex,
                  **sizing_kw) -> pd.Series:
    with _TiltOnlySignals():
        return ev_backtest.event_sleeve_returns(panel, rebal_dates, **sizing_kw)


class _ZeroFlatFee:
    """Context manager: neutralize the flat FEE_BPS at BOTH module globals that read it
    — ``production.events.sizing.FEE_BPS`` (size_event_book's sizing-time edge haircut)
    and ``production.events.backtest.FEE_BPS`` (its imported copy, behind the
    entry-notional fee charge). The engine then emits GROSS returns; the evidence-matched
    variant re-applies the ACTUAL taker fee at script level (see _taker_fee_stream).
    In-process only, restored on exit — the files on disk are never written."""

    def __enter__(self):
        self._sizing_fee = ev_sizing.FEE_BPS
        self._engine_fee = ev_backtest.FEE_BPS
        ev_sizing.FEE_BPS = 0.0
        ev_backtest.FEE_BPS = 0.0
        return self

    def __exit__(self, *exc):
        ev_sizing.FEE_BPS = self._sizing_fee
        ev_backtest.FEE_BPS = self._engine_fee
        return False


# --------------------------------------------------------------- book side channel
def _books(panel: pd.DataFrame, rebal_dates: pd.DatetimeIndex, per_group_cap: float,
           universe_kw: dict | None = None) -> dict:
    """PIT book (instrument_id, side, weight, entry) at each rebalance date — the same
    book-construction logic event_sleeve_returns runs internally, replicated here
    (read-only, using the real production primitives) as a diagnostic side-channel for
    n_positions and the variant's script-level taker-fee stream; it does not touch P&L.
    Must run under the SAME fee context as the engine call it mirrors (the sizing
    haircut changes book composition), so the caller wraps both in one `with` block."""
    universe_kw = universe_kw or {}
    df = panel.copy()
    df["obs_date"] = pd.to_datetime(df["obs_date"])
    avail_from = pd.to_datetime(df["available_from"], utc=True, errors="coerce")
    empty = pd.DataFrame(columns=["instrument_id", "side", "weight", "entry"])
    books = {}
    for t in rebal_dates:
        mask = (avail_from <= pd.Timestamp(t).tz_localize("UTC")).to_numpy()
        avail = df[mask]
        if avail.empty:
            books[t] = empty
            continue
        universe = liquid_universe(avail, **universe_kw)
        if not universe:
            books[t] = empty
            continue
        groups = dedupe_related(universe)
        universe_ids = {m.id for m in universe}
        sig = political_favorite_tilt(avail)
        sig = sig[sig["instrument_id"].isin(universe_ids)]
        if sig.empty:
            books[t] = empty
            continue
        book = size_event_book(sig, avail, capital_frac=0.10, groups=groups,
                               per_group_cap=per_group_cap)
        if book.empty:
            books[t] = empty
            continue
        # entry price = the latest PIT-knowable YES price at t (exactly what the engine
        # uses as the entered level)
        entry_px = (avail.sort_values("obs_date", kind="stable")
                    .drop_duplicates("instrument_id", keep="last")
                    .set_index("instrument_id")["yes_price"])
        book = book.copy()
        book["entry"] = book["instrument_id"].map(entry_px).astype(float)
        books[t] = book
    return books


def _sizes_from_books(books: dict) -> pd.Series:
    return pd.Series({t: int(len(b)) for t, b in books.items()})


def _taker_fee_stream(books: dict, grid: pd.DatetimeIndex,
                      rebal_dates: pd.DatetimeIndex) -> pd.Series:
    """Actual Kalshi taker-fee drag for the evidence-matched variant, mirroring the
    engine's fee-day placement (charged on the first grid date of each rebalance's hold
    window, exactly where event_sleeve_returns books its flat fee).

    Per unit weight entered at YES price ``entry``, the fee fraction of stake is
    ``taker_fee(entry) / entry`` (the tilt is always YES-side: its edge is +0.03, never
    negative). Charged ONCE per entry: only the INCREMENT of an instrument's weight over
    the previous rebalance's book is charged, so a position carried across consecutive
    weekly books (economically a continued hold) is not re-charged the way the as-wired
    flat fee re-charges it. No exit fee (hold-to-settlement; settlement is free)."""
    rebals = pd.DatetimeIndex(sorted(pd.to_datetime(pd.Index(rebal_dates))))
    rebals = rebals[(rebals >= grid[0]) & (rebals <= grid[-1])]
    fees = pd.Series(0.0, index=grid, dtype=float)
    prev_w: dict[str, float] = {}
    for ri, t in enumerate(rebals):
        t_next = rebals[ri + 1] if ri + 1 < len(rebals) else grid[-1]
        book = books.get(t)
        if book is None or book.empty:
            prev_w = {}   # engine holds nothing in (t, t_next] -> next entry is fresh
            continue
        hold = grid[(grid > t) & (grid <= t_next)]
        if len(hold) == 0:
            continue      # engine skips this partition entirely (book never traded)
        fee = 0.0
        cur_w: dict[str, float] = {}
        for _, row in book.iterrows():
            iid = row["instrument_id"]
            w = float(row["weight"])
            entry = float(row["entry"])
            cur_w[iid] = w
            if not np.isfinite(entry) or entry <= 0.0:
                continue  # engine also earns no P&L on such a position
            added = max(0.0, w - prev_w.get(iid, 0.0))
            if added > 0.0:
                fee += added * taker_fee(entry) / entry
        fees.loc[hold[0]] += fee
        prev_w = cur_w
    return fees


# ------------------------------------------------------------------------------ metrics
def _perf_block(r: pd.Series) -> dict:
    r = r.dropna()
    return {
        "n_days": int(len(r)),
        "start": str(r.index.min().date()) if len(r) else None,
        "end": str(r.index.max().date()) if len(r) else None,
        "ann_return": ann_return(r),
        "ann_vol": ann_vol(r),
        "sharpe": sharpe(r),
        "max_drawdown": max_drawdown(r),
        "hit_rate": hit_rate(r),
        "total_additive_pnl": float(r.sum()),
    }


def _positions_block(sizes: pd.Series) -> dict:
    active = sizes[sizes > 0]
    return {
        "n_rebalances": int(len(sizes)),
        "n_rebalances_with_positions": int(len(active)),
        "mean_when_active": float(active.mean()) if len(active) else 0.0,
        "median_when_active": float(active.median()) if len(active) else 0.0,
        "max": int(sizes.max()) if len(sizes) else 0,
    }


def _stream_blocks(returns: pd.Series, sizes: pd.Series, raw: pd.DataFrame,
                   end: pd.Timestamp) -> dict:
    """The four honesty blocks (full span, last-18mo, no-2024-11, lumpiness) for one
    return stream — shared by the as-wired run and the evidence-matched variant."""
    blocks: dict = {}
    blocks["full_span"] = _perf_block(returns)
    blocks["full_span"]["n_positions"] = _positions_block(sizes)

    cutoff_18m = end - pd.DateOffset(months=RECENCY_MONTHS)
    r18 = returns[returns.index >= cutoff_18m]
    sizes18 = sizes[sizes.index >= cutoff_18m]
    blocks["last_18_months"] = _perf_block(r18)
    blocks["last_18_months"]["n_positions"] = _positions_block(sizes18)
    blocks["last_18_months"]["window_start"] = str(cutoff_18m.date())

    no_nov = returns[returns.index.to_period("M") != ELECTION_MONTH]
    blocks["no_2024_11_variant"] = _perf_block(no_nov)
    blocks["no_2024_11_variant"]["note"] = ("2024-11 daily returns dropped entirely (not "
                                            "resampled) before recomputing metrics on the "
                                            "remaining span — a robustness/lumpiness cut, "
                                            "not a smooth counterfactual.")
    full_ar = blocks["full_span"]["ann_return"]
    nov_ar = blocks["no_2024_11_variant"]["ann_return"]
    edge_concentrated = bool(
        np.isfinite(full_ar) and np.isfinite(nov_ar)
        and ((full_ar > 0 and nov_ar <= 0) or (full_ar != 0 and nov_ar / full_ar < 0.3)))
    blocks["no_2024_11_variant"]["edge_concentrated_in_election_month"] = edge_concentrated

    r_nonzero = returns[returns != 0.0]
    total_pnl = float(returns.sum())
    if len(r_nonzero) >= 5:
        top5 = r_nonzero.abs().sort_values(ascending=False).head(5)
        top5_dates = top5.index
        top5_signed_sum = float(returns.loc[top5_dates].sum())
        pol_settle_dates = set(
            raw.loc[(raw["row_type"] == "settlement") & (raw["category"] == "politics"),
                   "obs_date"].dt.normalize())
        n_coincide = sum(1 for d in top5_dates
                         if pd.Timestamp(d).normalize() in pol_settle_dates)
        blocks["settlement_lumpiness"] = {
            "total_additive_pnl": total_pnl,
            "top5_dates": [str(pd.Timestamp(d).date()) for d in top5_dates],
            "top5_daily_returns": [float(returns.loc[d]) for d in top5_dates],
            "top5_signed_sum": top5_signed_sum,
            "top5_share_of_total_pnl": (top5_signed_sum / total_pnl
                                        if total_pnl != 0 else float("nan")),
            "top5_days_with_same_day_political_settlement": int(n_coincide),
            "note": ("top-5 by |daily return|; share-of-total-P&L is additive (sum of "
                     "simple daily returns), not compounded."),
        }
    else:
        blocks["settlement_lumpiness"] = {"note": "fewer than 5 nonzero-return days — skipped"}
    return blocks


# --------------------------------------------------------------------- lake price proxies
def _proxy_returns(lake: Lake, asset_class: str, symbols: list[str]) -> pd.Series:
    """Equal-weighted daily simple-return proxy for a sleeve, built straight from curated
    lake prices (NOT the optimizer's book weights) — the fallback the task specifies when
    the stored backtest report carries only aggregate stats, no daily return series."""
    df = lake.read_curated("prices", asset_class)
    if df.empty:
        return pd.Series(dtype=float)
    sym = df["instrument_id"].astype(str).str.split(":").str[1]
    df = df[sym.isin(symbols)]
    wide = (df.pivot_table(index="obs_date", columns="instrument_id", values="close",
                           aggfunc="last").sort_index())
    rets = wide.pct_change()
    return rets.mean(axis=1, skipna=True).dropna()


def _corr(a: pd.Series, b: pd.Series) -> tuple[float, int]:
    joined = pd.concat([a.rename("a"), b.rename("b")], axis=1, sort=True).dropna()
    if len(joined) < 10:
        return float("nan"), int(len(joined))
    return float(joined["a"].corr(joined["b"])), int(len(joined))


# ------------------------------------------------------------- mean-variance combination
def _mv_combo(mu_book: float, sigma_book: float, mu_events: float, sigma_events: float,
             rho: float, risk_cap_events: float) -> dict:
    """Two-asset (book, events) risk-contribution allocation using the SAME machinery
    production.portfolio.allocation uses for the real N-sleeve ERC + risk-cap blend
    (erc_weights + apply_risk_caps), applied here to a synthetic 2x2 covariance so the
    events sleeve's risk-contribution share is capped at exactly the configured 10%
    (configs/backtest.yaml: allocation.risk_caps.events)."""
    if not (np.isfinite(sigma_book) and np.isfinite(sigma_events) and np.isfinite(rho)):
        return {"error": "non-finite inputs"}
    Sigma = np.array([
        [sigma_book ** 2, rho * sigma_book * sigma_events],
        [rho * sigma_book * sigma_events, sigma_events ** 2],
    ])
    cov_df = pd.DataFrame(Sigma, index=["book", "events"], columns=["book", "events"])
    w_erc = erc_weights(cov_df)
    w_capped = apply_risk_caps(w_erc, cov_df,
                               caps={"events": risk_cap_events, "max_any_sleeve": 1.0})
    mu = pd.Series({"book": mu_book, "events": mu_events})
    w = w_capped.reindex(["book", "events"]).to_numpy()
    combined_mu = float((w_capped.reindex(["book", "events"]) * mu).sum())
    combined_var = float(w @ Sigma @ w)
    combined_sigma = float(np.sqrt(max(combined_var, 0.0)))
    combined_sharpe = combined_mu / combined_sigma if combined_sigma > 0 else float("nan")
    return {
        "inputs": {"mu_book": mu_book, "sigma_book": sigma_book, "mu_events": mu_events,
                   "sigma_events": sigma_events, "rho": rho},
        "erc_weights_uncapped": {"book": float(w_erc["book"]), "events": float(w_erc["events"])},
        "weights_capped_at_events_risk_cap": {"book": float(w_capped["book"]),
                                              "events": float(w_capped["events"])},
        "combined_ann_return": combined_mu,
        "combined_ann_vol": combined_sigma,
        "combined_sharpe": combined_sharpe,
        "book_alone_sharpe": mu_book / sigma_book if sigma_book > 0 else float("nan"),
    }


# ---------------------------------------------------------------------------------- main
def main() -> int:
    panel, raw = _build_panel()
    bt_cfg = backtest_config()
    uni_cfg = universe_config()
    per_group_cap = float(bt_cfg["constraints"]["position_cap"].get("events", 0.02))
    risk_cap_events = float(bt_cfg["allocation"]["risk_caps"].get("events", 0.10))

    start, end = panel["obs_date"].min(), panel["obs_date"].max()
    rebal_dates = rebalance_grid(start, end, "weekly")
    grid = pd.DatetimeIndex(sorted(panel["obs_date"].unique()))

    # ---- as-wired stream (flat 200bp fee, no nearness restriction) ----
    tilt_returns = _tilt_returns(panel, rebal_dates, per_group_cap=per_group_cap)
    book_sizes = _sizes_from_books(_books(panel, rebal_dates, per_group_cap))

    # ---- evidence-matched variant (pre-registered follow-up) ----
    # nearness: max_days_to_close=10 through the REAL liquid_universe filter;
    # fees: flat FEE_BPS neutralized at both module globals (engine emits gross),
    # actual taker fee re-applied at script level on new entries only.
    variant_universe_kw = {"max_days_to_close": VARIANT_MAX_DAYS_TO_CLOSE}
    with _ZeroFlatFee():
        variant_gross = _tilt_returns(panel, rebal_dates, per_group_cap=per_group_cap,
                                      **variant_universe_kw)
        books_variant = _books(panel, rebal_dates, per_group_cap,
                               universe_kw=variant_universe_kw)
    variant_fees = _taker_fee_stream(books_variant, grid, rebal_dates)
    variant_returns = variant_gross.sub(
        variant_fees.reindex(variant_gross.index).fillna(0.0))
    variant_sizes = _sizes_from_books(books_variant)

    out: dict = {
        "generated_at": pd.Timestamp.now("UTC").isoformat(),
        "signal": "production.events.signals.political_favorite_tilt (edge=0.03, "
                  "yes_price in [0.70, 0.95], category=='politics', venue=='kalshi')",
        "method_note": (
            "event_sleeve_returns's real per-rebalance loop was reused unmodified; its "
            "hard-wired (longshot_bias, resolution_convergence) signal inputs were "
            "in-process monkeypatched to (political_favorite_tilt, empty) for the "
            "duration of this run only — production/events/backtest.py on disk is "
            "untouched. See module docstring."
        ),
        "panel": {
            "n_bar_rows": int((raw["row_type"] == "bar").sum()),
            "n_settlement_rows": int((raw["row_type"] == "settlement").sum()),
            "n_series": int(raw["series_ticker"].nunique()),
            "n_political_series": int(raw.loc[raw["category"] == "politics",
                                              "series_ticker"].nunique()),
            "n_macro_series": int(raw.loc[raw["category"] == "macro",
                                          "series_ticker"].nunique()),
            "obs_date_range": [str(start.date()), str(end.date())],
        },
        "rebalance": {"cadence": "weekly (production default_freq grid)",
                      "n_rebalance_dates": int(len(rebal_dates)),
                      "per_group_cap": per_group_cap, "capital_frac": 0.10},
        "fees": {
            "round_trip_fee_bps": float(FEE_BPS),
            "note": ("event_sleeve_returns charges FEE_BPS (200bp = 2% round-trip) on "
                     "entered notional as a return drag on the first hold day of each "
                     "rebalance. Separately, size_event_book subtracts the SAME "
                     "FEE_BPS/1e4 (2%) from the raw signal edge as a trade-worthiness "
                     "floor before Kelly-sizing (a position is only opened if its edge "
                     "clears 2%) — since political_favorite_tilt's own edge is only "
                     "~0.03 (haircut=0.03 by signal design, already the deliberately "
                     "conservative slice of the ~3.4%% measured edge), this sizing-time "
                     "floor consumes roughly two-thirds of the signal's own edge before "
                     "any position is sized, and shrinks further wherever the 0.99 cap "
                     "binds. Both charges key off the same FEE_BPS constant but play "
                     "different roles: one is the ex-ante sizing filter, the other the "
                     "realized cost."),
        },
        "follow_up_pre_registration": FOLLOW_UP_PRE_REGISTRATION,
    }

    # ---------------------------------------- 1/2/4: as-wired performance + honesty blocks
    out.update(_stream_blocks(tilt_returns, book_sizes, raw, end))
    edge_concentrated = out["no_2024_11_variant"]["edge_concentrated_in_election_month"]

    # ------------------------------------------------------- 3: portfolio impact
    lake = Lake()
    crypto_proxy = _proxy_returns(lake, "crypto", uni_cfg["sleeves"]["crypto"]["symbols"])
    fx_proxy = _proxy_returns(lake, "fx", list(uni_cfg["sleeves"]["fx_etf"]["symbols"].keys()))
    corr_crypto, n_crypto = _corr(tilt_returns, crypto_proxy)
    corr_fx, n_fx = _corr(tilt_returns, fx_proxy)

    report_path = REPO_ROOT / "reports" / "backtest_20260711T195019Z.json"
    report = json.loads(report_path.read_text())
    headline = report.get("headline", {})
    per_sleeve = report.get("per_sleeve", {})

    blended_book = pd.concat([crypto_proxy.rename("c"), fx_proxy.rename("f")],
                             axis=1, sort=True).mean(axis=1, skipna=True).dropna()
    corr_blend, n_blend = _corr(tilt_returns, blended_book)
    mu_blend = ann_return(blended_book)
    sigma_blend = ann_vol(blended_book)

    # The raw equal-weighted buy-and-hold proxy is dominated by crypto's historical bull
    # run (~78%+ annualized vol) -- nothing like the risk-managed book, which targets
    # total_vol_target (configs/backtest.yaml) via the vol-target overlay + ERC alloc.
    # Rescale (a pure linear scalar -- Sharpe- and correlation-invariant) so sigma_book
    # matches the book's actual target vol, making the mean-variance comparison scale-
    # honest without re-simulating the overlay's causal EWMA/clip/deadband machinery.
    total_vol_target = float(bt_cfg.get("total_vol_target", 0.10))
    vt_scale = (total_vol_target / sigma_blend) if sigma_blend > 0 else 1.0
    mu_blend_vt = mu_blend * vt_scale
    sigma_blend_vt = sigma_blend * vt_scale

    mu_events_full = out["full_span"]["ann_return"]
    sigma_events_full = out["full_span"]["ann_vol"]

    out["portfolio_impact"] = {
        "report_used": str(report_path.name),
        "report_schema_note": ("headline/per_sleeve carry AGGREGATE stats only — no "
                               "daily total_returns series is persisted in this report "
                               "-> falling back to lake-price sleeve proxies + the "
                               "aggregate headline stats, per task instructions."),
        "current_book_headline": {
            "ann_return_net": headline.get("ann_return_net"),
            "ann_vol_net": headline.get("ann_vol_net"),
            "net_sharpe": headline.get("net_sharpe"),
            "caveat": ("this report's total-book aggregates are numerically ~0 "
                      "(ann_vol_net ~1.6e-9) — both re-specs in the prior commit "
                      "failed the promotion gate, leaving the current book with "
                      "essentially no live factor exposure. Any mean-variance "
                      "combination against this 'book' is therefore degenerate "
                      "(near-zero-vol asset) and is reported below only as a "
                      "transparency check, not as the headline estimate."),
            "per_sleeve_present": list(per_sleeve.keys()),
        },
        "correlation_vs_lake_proxies": {
            "crypto": {"corr": corr_crypto, "n_overlap_days": n_crypto,
                      "symbols": uni_cfg["sleeves"]["crypto"]["symbols"]},
            "fx_etf": {"corr": corr_fx, "n_overlap_days": n_fx,
                      "symbols": list(uni_cfg["sleeves"]["fx_etf"]["symbols"].keys())},
            "note": ("equal-weighted daily simple-return proxies built directly from "
                     "curated data/curated/dataset=prices (NOT the optimizer's book "
                     "weights, since the current book carries ~no live crypto/fx "
                     "signal right now per the caveat above)."),
        },
        "mean_variance_combined": {
            "risk_cap_events": risk_cap_events,
            "scenario_degenerate_current_book": _mv_combo(
                headline.get("ann_return_net", 0.0) or 0.0,
                headline.get("ann_vol_net", 0.0) or 0.0,
                mu_events_full, sigma_events_full, 0.0, risk_cap_events),
            "scenario_fx_crypto_lake_proxy_blend_raw": {
                **_mv_combo(mu_blend, sigma_blend, mu_events_full, sigma_events_full,
                           corr_blend if np.isfinite(corr_blend) else 0.0, risk_cap_events),
                "caveat": (f"RAW unmanaged buy-and-hold proxy: sigma_book={sigma_blend:.2%} "
                          "annualized -- dominated by crypto's historical run, nothing "
                          "like the real risk-managed book (no vol targeting, no "
                          "hedging, no cost model). Useful mainly for the correlation "
                          "estimate, not for combined-Sharpe magnitude -- see the "
                          "vol_targeted scenario for a scale-honest comparison."),
            },
            "scenario_fx_crypto_lake_proxy_blend_vol_targeted": {
                **_mv_combo(mu_blend_vt, sigma_blend_vt, mu_events_full, sigma_events_full,
                           corr_blend if np.isfinite(corr_blend) else 0.0, risk_cap_events),
                "caveat": (f"the raw blend rescaled by a single linear factor so its "
                          f"ann_vol matches total_vol_target={total_vol_target:.0%} "
                          "(configs/backtest.yaml) -- a scale-honest approximation of "
                          "the risk-managed book's actual operating vol, NOT a re-"
                          "simulation of the causal EWMA vol-target overlay (which "
                          "clips/deadbands and reacts with a lag). Sharpe and "
                          "correlation are invariant to this rescale by construction; "
                          "only the absolute ann_return/ann_vol/weights shift. This is "
                          "the headline mean-variance scenario."),
            },
            "note": ("erc_weights + apply_risk_caps reused unmodified from "
                     "production.portfolio.allocation on a synthetic 2-asset "
                     "(book, events) covariance; the events risk-contribution share is "
                     "capped at allocation.risk_caps.events (0.10) exactly as the real "
                     "N-sleeve ERC does."),
        },
    }

    # ------------------------------- evidence-matched variant (pre-registered follow-up)
    v_blocks = _stream_blocks(variant_returns, variant_sizes, raw, end)
    v_corr_crypto, v_n_crypto = _corr(variant_returns, crypto_proxy)
    v_corr_fx, v_n_fx = _corr(variant_returns, fx_proxy)
    v_corr_blend, _ = _corr(variant_returns, blended_book)
    v_mu = v_blocks["full_span"]["ann_return"]
    v_sigma = v_blocks["full_span"]["ann_vol"]

    v_full_sharpe = v_blocks["full_span"]["sharpe"]
    v_18_sharpe = v_blocks["last_18_months"]["sharpe"]
    dr_pass = bool(np.isfinite(v_full_sharpe) and np.isfinite(v_18_sharpe)
                   and v_full_sharpe > 0 and v_18_sharpe > 0)

    out["variant_evidence_matched"] = {
        "spec": {
            "nearness": (f"liquid_universe max_days_to_close="
                         f"{VARIANT_MAX_DAYS_TO_CLOSE} passed through "
                         "event_sleeve_returns's sizing_kw — the monkeypatch path does "
                         "NOT bypass the filter, so no panel-level workaround was "
                         "needed. min_days_to_close stays at its default 1 (the T-1 "
                         "convention's [close-10d, close-1d] entry window)."),
            "fees": ("flat FEE_BPS neutralized IN-PROCESS at both module globals that "
                     "read it: production.events.sizing.FEE_BPS (size_event_book's "
                     "sizing-time edge haircut) and production.events.backtest.FEE_BPS "
                     "(its imported copy, behind the entry-notional charge); both "
                     "restored after the run, files on disk untouched. The engine "
                     "therefore emitted GROSS returns; the actual Whelan-imputed "
                     "per-contract taker fee (research/diagnostics/_common.py::"
                     "taker_fee, 100-lot imputation) was applied at script level as "
                     "taker_fee(entry)/entry per unit weight, charged ONCE on new "
                     "weight increments per instrument at each rebalance's first hold "
                     "day (same fee-day placement as the engine's flat charge). No "
                     "exit fee: hold-to-settlement, settlement free on Kalshi."),
            "documented_approximations": [
                "a position dropped from the book before settlement pays no exit fee "
                "either (with <=10d-to-close markets on a weekly grid, nearly all "
                "positions run to settlement, so the understatement is small)",
                "with a zero sizing haircut the full 0.03 edge reaches Kelly, so "
                "variant positions typically pin the per-group cap — books are "
                "systematically larger than as-wired",
                "engine wiring quirk shared by both runs: size_event_book takes the "
                "LATEST signal per instrument, so a market whose price was in-bucket "
                "on an earlier bar but has since left [0.70,0.95] can still be entered "
                "at its current (out-of-bucket) price",
            ],
        },
        **v_blocks,
        "fees_applied": {
            "model": "_common.taker_fee (Whelan 100-lot imputation), entry-only",
            "total_fee_drag_additive": float(variant_fees.sum()),
            "n_fee_charge_days": int((variant_fees != 0).sum()),
            "gross_total_additive_pnl": float(variant_gross.sum()),
        },
        "correlation_vs_lake_proxies": {
            "crypto": {"corr": v_corr_crypto, "n_overlap_days": v_n_crypto},
            "fx_etf": {"corr": v_corr_fx, "n_overlap_days": v_n_fx},
        },
        "mean_variance_combined": {
            "risk_cap_events": risk_cap_events,
            "scenario_fx_crypto_lake_proxy_blend_vol_targeted": {
                **_mv_combo(mu_blend_vt, sigma_blend_vt, v_mu, v_sigma,
                           v_corr_blend if np.isfinite(v_corr_blend) else 0.0,
                           risk_cap_events),
                "caveat": ("same vol-targeted proxy-book scenario as the as-wired "
                          "section (see its caveat); only the events leg differs."),
            },
        },
        "decision_rule": {
            "rule": ("PRE-REGISTERED: variant full-span Sharpe > 0 AND last-18mo "
                     "Sharpe > 0 -> mismatch confirmed as the cause; proposal = "
                     "production re-spec of the events sleeve's entry-nearness + fee "
                     "model (execution layer, no n_trials). Either Sharpe <= 0 -> the "
                     "tilt fails at portfolio level even evidence-matched; record as a "
                     "dead end at sizing scale despite the per-stake edge."),
            "full_span_sharpe": v_full_sharpe,
            "last_18mo_sharpe": v_18_sharpe,
            "pass": dr_pass,
            "conclusion": (
                "MISMATCH CONFIRMED: the as-wired negative Sharpe was caused by the "
                "entry-nearness + fee-model mismatch; propose a production re-spec of "
                "the events sleeve's entry-nearness + fee model (execution layer, no "
                "n_trials change)." if dr_pass else
                "DEAD END AT SIZING SCALE: the tilt fails at portfolio level even "
                "evidence-matched (T-1-nearness entries, actual taker fees), despite "
                "the measured per-stake edge."),
        },
    }

    # -------------------------------------------------------------------------- verdict
    fs = out["full_span"]
    l18 = out["last_18_months"]
    mv = out["portfolio_impact"]["mean_variance_combined"][
        "scenario_fx_crypto_lake_proxy_blend_vol_targeted"]
    verdict_lines = [
        f"TILT-ONLY EVENTS SLEEVE {fs['start']}..{fs['end']} (n={fs['n_days']}d): "
        f"ann_return={fs['ann_return']:.4%} ann_vol={fs['ann_vol']:.4%} "
        f"sharpe={fs['sharpe']:.3f} maxDD={fs['max_drawdown']:.4%}",
        f"last {RECENCY_MONTHS}mo (n={l18['n_days']}d): ann_return={l18['ann_return']:.4%} "
        f"sharpe={l18['sharpe']:.3f}",
        f"positions: active in {fs['n_positions']['n_rebalances_with_positions']}/"
        f"{fs['n_positions']['n_rebalances']} rebalances, "
        f"median {fs['n_positions']['median_when_active']:.1f} when active",
        f"no-2024-11 variant sharpe={out['no_2024_11_variant']['sharpe']:.3f} "
        f"ann_return={out['no_2024_11_variant']['ann_return']:.4%} "
        f"[EDGE CONCENTRATED IN ELECTION MONTH]" if edge_concentrated else
        f"no-2024-11 variant sharpe={out['no_2024_11_variant']['sharpe']:.3f} "
        f"ann_return={out['no_2024_11_variant']['ann_return']:.4%} (edge survives)",
        f"corr vs crypto proxy={corr_crypto:.3f}, vs fx_etf proxy={corr_fx:.3f}",
        f"mean-variance @ {risk_cap_events:.0%} events risk cap "
        f"(vol-targeted fx/crypto-proxy book): "
        f"combined_sharpe={mv.get('combined_sharpe', float('nan')):.3f} vs "
        f"book-alone={mv.get('book_alone_sharpe', float('nan')):.3f}",
    ]
    vfs = v_blocks["full_span"]
    v18 = v_blocks["last_18_months"]
    v_mv = out["variant_evidence_matched"]["mean_variance_combined"][
        "scenario_fx_crypto_lake_proxy_blend_vol_targeted"]
    verdict_lines += [
        f"EVIDENCE-MATCHED VARIANT (<= {VARIANT_MAX_DAYS_TO_CLOSE}d to close, actual "
        f"taker fees, entry-only): ann_return={vfs['ann_return']:.4%} "
        f"ann_vol={vfs['ann_vol']:.4%} sharpe={vfs['sharpe']:.3f} "
        f"maxDD={vfs['max_drawdown']:.4%}",
        f"variant last {RECENCY_MONTHS}mo: ann_return={v18['ann_return']:.4%} "
        f"sharpe={v18['sharpe']:.3f}",
        f"variant no-2024-11: sharpe={v_blocks['no_2024_11_variant']['sharpe']:.3f}"
        + (" [EDGE CONCENTRATED IN ELECTION MONTH]"
           if v_blocks["no_2024_11_variant"]["edge_concentrated_in_election_month"]
           else " (edge survives)"),
        f"variant mean-variance @ {risk_cap_events:.0%} cap: "
        f"combined_sharpe={v_mv.get('combined_sharpe', float('nan')):.3f} vs "
        f"book-alone={v_mv.get('book_alone_sharpe', float('nan')):.3f}",
        f"DECISION RULE: {'PASS -> mismatch confirmed, propose entry-nearness + '
        'fee-model re-spec (execution layer, no n_trials)' if dr_pass else
        'FAIL -> dead end at sizing scale despite the per-stake edge'}",
    ]
    out["verdict"] = " | ".join(verdict_lines)

    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT.write_text(json.dumps(out, indent=2, default=str))
    print(out["verdict"])
    print(f"\nartifact -> {ARTIFACT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
