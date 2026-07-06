"""Walk-forward backtest engine — the layer that ties the whole system together.

Data flow at a glance (everything below is strictly point-in-time; a decision dated ``t``
reads only inputs knowable by end of day ``t`` and earns returns from ``t+1``):

  signals (once, on the full bundle — proven PIT by the corruption harness)
    -> z-scores per (date, sleeve)
    -> per-(factor, sleeve) rolling shrunk IC*  (embargoes the unresolved forward tail)
  then, walking the weekly rebalance grid:
    monthly:  RiskModel.build(as_of)  and  sleeve_allocation refresh
    weekly:   z(t) -> refine_alpha(IC*(t), resid_vol) -> combine -> optimize_sleeve
              -> new sleeve weights. With >=2 live factors in a sleeve the combine is the
              correlation-aware Grinold-Kahn blend (w = C^-1 ic) using a score-correlation
              matrix C re-estimated PIT at the monthly points (cached like the risk model);
              single-factor sleeves keep the plain refine path.
    daily:    positions (lagged one day) earn returns; costs charged on the first
              effective day of each rebalance; sleeves blended by the monthly allocation;
              total exposure scaled by the (PIT) overlay multiplier.

Why the precompute-once step is still PIT: rolling_shrunk_ic at decision date ``t`` uses
only IC observations whose forward window resolved by ``t`` (positional horizon embargo),
and every IC observation is a same-date cross-sectional statistic. Corrupting any input
row dated after ``t`` therefore cannot move a single decision at or before ``t`` — the
integration corruption test in tests/test_backtest_engine.py exercises exactly this.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from production.alpha.combine import (combine_alphas, combine_alphas_grinold,
                                      score_correlation)
from production.alpha.ic import forward_returns, rank_ic, rolling_shrunk_ic
from production.alpha.refine import refine_alpha
from production.alpha.registry import FactorRegistry
from production.alpha.zscore import zscore_scores
from production.backtest.cost_model import CostModel
from production.backtest.report import build_report
from production.core.calendar import month_starts, rebalance_grid
from production.core.config import backtest_config, costs_config, risk_config
from production.portfolio.optimizer import optimize_sleeve
from production.portfolio.overlays import overlay_multiplier
from production.portfolio.allocation import sleeve_allocation
from production.risk.covariance import ewma_cov
from production.risk.exposures import build_exposures
from production.risk.factor_returns import estimate_factor_returns
from production.risk.model import RiskModel
from production.risk.specific import specific_vol
from production.signals.base import sleeve_from_id

# Nominal book size used only to turn fractional weight changes into a dollar trade for
# the cost model's square-root impact term. It cancels out of the return-space bookkeeping
# except through that (small) impact term; the per-sleeve cost floor is what actually binds.
_NOMINAL_AUM = 1.0e8
_WARMUP_YEARS = 3
# Risk-model estimation window. RiskModel.build re-estimates factor returns over the whole
# price history it is handed; feeding it only a trailing slice bounds each monthly rebuild
# to a fixed cost (otherwise the growing window makes the walk O(months^2)). Two years of
# calendar history keeps > 252 factor-return observations (the covariance min_obs) while
# staying PIT — the slice still contains only obs_date <= as_of rows.
_RISK_WINDOW_DAYS = 560


@dataclass
class BacktestResult:
    """Everything the report needs, plus the raw series for downstream inspection."""
    total_returns: pd.Series                       # net, daily
    total_gross_returns: pd.Series                 # gross of costs, daily
    sleeve_returns: pd.DataFrame                    # net, daily, one column per sleeve
    weights_history: dict                           # sleeve -> DataFrame (rebalance dates x ids)
    costs: pd.Series                                # daily cost drag on the total book
    overlay: pd.Series                              # daily overlay multiplier
    report: dict = field(default_factory=dict)
    risk_models: dict = field(default_factory=dict)  # sleeve -> last built RiskModel (diagnostics)
    # ---- internal carriers for the report layer (not part of the public headline) ----
    _factor_returns: pd.DataFrame | None = None     # risk-model factor returns (attribution basis)
    _weights_by_factor: dict | None = None          # factor -> weights DataFrame (per-alpha attrib)
    _inst_returns: pd.DataFrame | None = None       # daily instrument returns (per-alpha attrib)
    _ic_realized_vs_training: dict | None = None    # factor -> {realized, training}
    _factors_used: list | None = None
    _gate_applied: bool = True
    _warnings: list = field(default_factory=list)
    _cost_sensitivity: dict | None = None           # {multiplier: annual-drag-scaling ratio}


def _prices_wide(prices: pd.DataFrame, ids, value: str = "close") -> pd.DataFrame:
    """Wide daily pivot of ``value`` for ``ids`` (all obs_dates present)."""
    df = prices[prices["instrument_id"].isin(list(ids))]
    if df.empty:
        return pd.DataFrame()
    return (df.pivot_table(index="obs_date", columns="instrument_id",
                           values=value, aggfunc="last").sort_index())


def _asof_row(panel: pd.DataFrame, t) -> pd.Series:
    """Latest row of a date-indexed wide panel with index <= t (empty Series if none)."""
    sub = panel[panel.index <= t]
    if sub.empty:
        return pd.Series(dtype=float)
    return sub.iloc[-1]


def _precompute_structural_returns(sprices: pd.DataFrame, sleeve: str, cfg_risk: dict,
                                   sectors: pd.Series | None = None):
    """Estimate the full-span factor returns / residuals for a structural sleeve, ONCE.

    ``RiskModel.build`` internally re-runs ``estimate_factor_returns`` from the sleeve
    start on every monthly rebuild — O(months x history). But a factor return on day ``d``
    depends only on that day's cross-section and the exposures frozen before its month, all
    dated ``<= d``; nothing after ``d`` can change it. So estimating once over the full span
    and slicing ``<= as_of`` per month is IDENTICAL to (and far cheaper than) rebuilding, and
    stays point-in-time. Returns ``(factor_returns, residuals)`` or ``(None, None)`` on any
    failure (the caller then falls back to ``RiskModel.build``).
    """
    try:
        start = pd.to_datetime(sprices["obs_date"]).min()
        end = pd.to_datetime(sprices["obs_date"]).max()
        fr, resid = estimate_factor_returns(sprices, sleeve, start, end, cfg_risk, sectors)
        if fr is None or fr.empty:
            return None, None
        fr.index = pd.DatetimeIndex(fr.index)
        resid.index = pd.DatetimeIndex(resid.index)
        return fr, resid
    except Exception:  # noqa: BLE001
        return None, None


def _assemble_structural(window_prices: pd.DataFrame, sleeve: str, as_of, ids,
                         cfg_risk: dict, fr_full: pd.DataFrame,
                         resid_full: pd.DataFrame,
                         sectors: pd.Series | None = None) -> RiskModel | None:
    """Assemble a structural RiskModel from precomputed factor returns (mirrors
    ``RiskModel._build_structural`` exactly, but reuses the one-shot factor returns).

    Returns None if too few resolved factor-return observations exist as of ``as_of`` —
    the caller then defers to ``RiskModel.build`` (full-covariance fallback).
    """
    fac_cfg = cfg_risk.get("factor_covariance", {})
    spec_cfg = cfg_risk.get("specific_risk", {})
    min_obs = int(fac_cfg.get("min_obs", 252))

    fr = fr_full[fr_full.index <= as_of]
    if len(fr) < min_obs:
        return None
    ids = list(ids)
    B = build_exposures(window_prices, sleeve, as_of, ids, cfg_risk, sectors)
    F = ewma_cov(
        fr, halflife=float(fac_cfg.get("ewma_halflife_days", 90)),
        shrink=float(fac_cfg.get("shrinkage_to_diagonal", 0.3)),
        min_obs=min_obs).reindex(index=B.columns, columns=B.columns)
    resid = resid_full[resid_full.index <= as_of]
    sv = specific_vol(
        resid.reindex(columns=ids),
        halflife=float(spec_cfg.get("ewma_halflife_days", 42)),
        shrink_weight=float(spec_cfg.get("cross_sectional_shrink_weight", 0.25)),
        min_obs=int(spec_cfg.get("min_obs", 63))).reindex(ids)
    sv = sv.fillna(sv.median() if sv.notna().any() else 0.0)
    D = (sv ** 2).rename("D")
    return RiskModel(sleeve, ids, B=B, F=F, D=D, resid_vol=sv.rename("resid_vol"))


def _empty_macro() -> pd.DataFrame:
    return pd.DataFrame({"obs_date": pd.Series([], dtype="datetime64[ns]"),
                         "series_id": pd.Series([], dtype=object),
                         "value": pd.Series([], dtype=float),
                         "available_from": pd.Series([], dtype="datetime64[ns, UTC]")})


def _select_factors(registry: FactorRegistry) -> tuple[dict, bool, list]:
    """Accepted factors, or ALL candidates with a loud warning if the gate has not run."""
    accepted = registry.factors(status="accepted")
    warnings_out: list = []
    if accepted:
        return accepted, True, warnings_out
    allf = registry.factors()
    msg = ("gate not yet applied — trading candidates: no factor has status 'accepted' "
           f"in the registry, falling back to all {len(allf)} candidate factors")
    warnings.warn(msg, stacklevel=2)
    warnings_out.append(msg)
    return allf, False, warnings_out


def _rolling_ic_se(ic_by_date: pd.Series, window: int = 252, horizon_days: int = 1) -> pd.Series:
    """Trailing standard error of the IC series over the SAME embargoed window as
    ``rolling_shrunk_ic``: ``std(window) / sqrt(max(n_window, 1))``.

    This is the estimation-error scale of the shrunk IC that drives the alpha-uncertainty
    ellipsoid. The window selection mirrors ``rolling_shrunk_ic`` exactly (positional
    horizon embargo on the IC index), so the SE at ``t`` is a pure function of IC dates
    whose forward window has resolved by ``t`` — PIT-safe by the same argument.
    """
    s = pd.Series(ic_by_date).dropna().sort_index()
    out: dict = {}
    for i, t in enumerate(s.index):
        j_max = i - horizon_days
        if j_max < 0:
            out[t] = float("nan")
            continue
        w = s.iloc[:j_max + 1].iloc[-window:]
        n = len(w)
        out[t] = float(w.std(ddof=1) / np.sqrt(max(n, 1))) if n >= 2 else float("nan")
    return pd.Series(out)


def _precompute_factor(name: str, spec: dict, registry: FactorRegistry,
                       data: dict, sleeve_map: pd.Series, ic_floor=None,
                       compute_se: bool = False):
    """Signal -> z-scores -> per-sleeve rolling shrunk IC* for one factor.

    Returns ``(z_panel, ic_table, icstar_by_sleeve, icse_by_sleeve)`` or ``None`` if the
    signal produced nothing usable. ``icstar_by_sleeve[sleeve]`` is a date-indexed Series
    of IC*; ``icse_by_sleeve[sleeve]`` the matching IC standard-error series (empty dict
    unless ``compute_se`` — the robust term is opt-in and this is the only extra cost). The
    IC series fed to ``rolling_shrunk_ic`` is trimmed to ``obs_date >= ic_floor`` (a tail
    long enough to cover every grid-date rolling window) purely to bound the Python-loop cost.
    """
    cls = registry.signal_class(name)
    panel = cls().compute(data)
    if panel is None or panel.empty:
        return None
    z = zscore_scores(panel, sleeve_map)
    if z.empty:
        return None
    horizon = int(spec["horizon_days"])
    fwd = forward_returns(data["prices"], horizon)
    ic = rank_ic(z, fwd, sleeve_map)
    icstar_by_sleeve: dict = {}
    icse_by_sleeve: dict = {}
    for sleeve, g in ic.groupby("sleeve"):
        ic_by_date = g.set_index("obs_date")["rank_ic"].sort_index()
        if ic_floor is not None:
            ic_by_date = ic_by_date[ic_by_date.index >= ic_floor]
        icstar_by_sleeve[sleeve] = rolling_shrunk_ic(
            ic_by_date, window=252, n0=126, horizon_days=horizon)
        if compute_se:
            icse_by_sleeve[sleeve] = _rolling_ic_se(
                ic_by_date, window=252, horizon_days=horizon)
    return z, ic, icstar_by_sleeve, icse_by_sleeve


def _latest_z_at(z_panel: pd.DataFrame, t, ids) -> pd.DataFrame:
    """Latest z-score per instrument with obs_date <= t, re-stamped to date ``t``.

    Using the most recent knowable score (never a future one) keeps the decision PIT even
    when a signal does not emit exactly on the grid date.
    """
    sub = z_panel[(z_panel["instrument_id"].isin(list(ids)))
                  & (z_panel["obs_date"] <= t)]
    if sub.empty:
        return sub
    last = (sub.sort_values("obs_date").groupby("instrument_id", as_index=False).last())
    last["obs_date"] = t
    return last[["obs_date", "instrument_id", "value"]]


def _sufficient_ids(prices: pd.DataFrame, ids, as_of, min_days: int) -> list:
    """Ids with at least ``min_days`` observations on/before ``as_of`` (trustable risk est.)."""
    df = prices[(prices["instrument_id"].isin(list(ids))) & (prices["obs_date"] <= as_of)]
    counts = df.groupby("instrument_id")["obs_date"].count()
    return [i for i in ids if counts.get(i, 0) >= min_days]


def run_backtest(data: dict, instruments: pd.Series, cfg: dict | None = None,
                 factors_cfg=None, lake=None,
                 sectors: pd.Series | None = None) -> BacktestResult:
    """Run the walk-forward backtest. See module docstring for the full data flow.

    Parameters
    ----------
    data         : bundle with ``prices`` (required) and optional ``funding``/``macro``/``cot``.
    instruments  : Series mapping instrument_id -> sleeve; only these are traded.
    cfg          : backtest config (defaults to ``backtest_config()``).
    factors_cfg  : a ``FactorRegistry`` or a path to factors.yaml (defaults to the repo file).
    lake         : unused by the engine itself; accepted for signature symmetry with the CLI.
    sectors      : optional Series mapping instrument_id -> sector label. When given, it is
                   threaded into the equity sleeve's risk model (sector dummies in ``B``) and
                   optimizer (the ±sector_band constraint). Non-equity sleeves never see it.
    """
    sectors = pd.Series(sectors) if sectors is not None else None
    if cfg is None:
        cfg = backtest_config()
    if isinstance(factors_cfg, FactorRegistry):
        registry = factors_cfg
    else:
        registry = FactorRegistry(factors_cfg) if factors_cfg is not None else FactorRegistry()

    prices = data["prices"]
    macro = data.get("macro")
    macro_panel = macro if (macro is not None and not macro.empty) else _empty_macro()
    costs_cfg = costs_config()
    cost_model = CostModel(costs_cfg)
    adv_window = int(costs_cfg.get("adv_window_days", 20))

    instruments = pd.Series(instruments)
    sleeve_map = instruments  # id -> sleeve, for zscore / rank_ic

    # Trade only sleeves that are both requested and present in the price bundle.
    present_ids = set(prices["instrument_id"].unique())
    traded = instruments[instruments.index.isin(present_ids)]
    sleeves = [s for s in pd.unique(traded.to_numpy()) if s in ("equity", "crypto",
                                                                 "fx_etf", "commodity_etf")]
    sleeve_ids = {s: sorted(traded[traded == s].index.tolist()) for s in sleeves}

    factors, gate_applied, warn_list = _select_factors(registry)
    # Alpha-uncertainty robustness is opt-in; when off (default) we skip the per-factor IC
    # standard-error precompute and the per-rebalance alpha_se assembly entirely.
    robust_kappa = float(cfg["optimizer"].get("robust_kappa", 0.0))

    # ---- timeline ---------------------------------------------------------------------
    start = pd.Timestamp(prices["obs_date"].min())
    end = pd.Timestamp(prices["obs_date"].max())
    warmup_start = start + pd.DateOffset(years=_WARMUP_YEARS)
    grid = rebalance_grid(warmup_start, end, freq=cfg["walk_forward"].get("rebalance", "weekly"))
    grid = pd.DatetimeIndex([g for g in grid if g <= end])
    month_pts = set(month_starts(grid))

    if len(grid) == 0:
        raise ValueError(
            "empty rebalance grid — need > 3y of history before the backtest window")

    # Only IC observations within ~1.75y before the first rebalance can enter any grid-date
    # rolling window (252 trading days + horizon embargo); slicing the IC series to that tail
    # before the (Python-loop) rolling_shrunk_ic keeps the precompute cost bounded. PIT-safe:
    # a shorter head cannot change any value the grid actually reads.
    ic_floor = grid[0] - pd.Timedelta(days=640)

    # ---- precompute signals -> z -> IC*  (once, on the full bundle) -------------------
    z_panels: dict = {}
    ic_tables: dict = {}
    icstar: dict = {}          # (factor, sleeve) -> date-indexed IC* series
    icse: dict = {}            # (factor, sleeve) -> date-indexed IC-SE series (robust only)
    factor_sleeves: dict = {}
    for name, spec in factors.items():
        try:
            pre = _precompute_factor(name, spec, registry, data, sleeve_map, ic_floor,
                                     compute_se=robust_kappa > 0.0)
        except Exception as exc:  # noqa: BLE001 - one bad signal must not sink the run
            warn_list.append(f"factor {name}: precompute failed ({exc}) — skipped")
            continue
        if pre is None:
            continue
        z, ic, icstar_by_sleeve, icse_by_sleeve = pre
        z_panels[name] = z
        ic_tables[name] = ic
        factor_sleeves[name] = set(spec["sleeves"])
        for sleeve, series in icstar_by_sleeve.items():
            icstar[(name, sleeve)] = series
        for sleeve, series in icse_by_sleeve.items():
            icse[(name, sleeve)] = series

    min_hist = 252  # trailing observations required before an id enters the risk model
    sleeve_vol_target = cfg.get("sleeve_vol_target")
    cfg_risk = risk_config()
    structural_sleeves = set(cfg_risk.get("structural_sleeves", ("equity", "crypto")))

    # ---- per-sleeve walk-forward ------------------------------------------------------
    sleeve_net: dict = {}       # sleeve -> daily net return Series
    sleeve_gross: dict = {}     # sleeve -> daily gross return Series
    weights_history: dict = {}  # sleeve -> DataFrame (rebalance dates x ids)
    weights_by_factor: dict = {name: {} for name in z_panels}  # factor -> {date: Series}
    structural_fr: dict = {}    # sleeve -> full-span factor returns (reused for attribution)

    risk_models: dict = {}      # sleeve -> last built RiskModel (diagnostics carrier)

    for sleeve in sleeves:
        ids_all = sleeve_ids[sleeve]
        # Sector labels are an equity-only concept here; restrict to this sleeve's ids so
        # every downstream reindex (exposures, factor returns, optimizer) sees only names it
        # actually trades. Non-equity sleeves always get None.
        sleeve_sectors = None
        if sectors is not None and sleeve == "equity":
            ss = sectors.reindex(ids_all).dropna()
            sleeve_sectors = ss if not ss.empty else None
        sprices = prices[prices["instrument_id"].isin(ids_all)]
        close_wide = _prices_wide(sprices, ids_all)
        if close_wide.empty:
            continue
        rets_wide = close_wide.pct_change()

        # Precompute the trailing ADV / sigma panels ONCE (matches
        # cost_model.trailing_adv_sigma: shift(1) then rolling over each instrument's own
        # series). Slicing these per rebalance is far cheaper than a groupby-per-date and is
        # PIT-identical — row t uses only data through t-1.
        dv_wide = _prices_wide(sprices, ids_all, value="dollar_volume")
        adv_panel = dv_wide.shift(1).rolling(adv_window).mean()
        sigma_panel = rets_wide.shift(1).rolling(adv_window).std()

        active_factors = [f for f in z_panels if sleeve in factor_sleeves[f]]

        # One-shot factor-return estimation for structural sleeves (see helper docstring).
        fr_full = resid_full = None
        if sleeve in structural_sleeves:
            fr_full, resid_full = _precompute_structural_returns(sprices, sleeve, cfg_risk,
                                                                 sleeve_sectors)
            if fr_full is not None:
                structural_fr[sleeve] = fr_full

        rm: RiskModel | None = None
        rm_ids: list = []
        # Score-correlation matrix for the Grinold combine, re-estimated PIT at the monthly
        # points (window 252) and cached across the intervening weekly rebalances — the same
        # monthly-cadence trick used for the risk model. Only built when >=2 factors are live.
        score_corr = None
        w_prev = pd.Series(dtype=float)   # over union ids, carried across rebalances
        rebal_weights: dict = {}          # grid date -> Series (rm_ids)
        rebal_costs: dict = {}            # grid date -> cost_return (fraction of NAV)

        for t in grid:
            # --- monthly risk-model rebuild ---
            if t in month_pts or rm is None:
                ids_ok = _sufficient_ids(sprices, ids_all, t, min_hist)
                if len(ids_ok) >= 2:
                    try:
                        window_prices = sprices[
                            sprices["obs_date"] > t - pd.Timedelta(days=_RISK_WINDOW_DAYS)]
                        built = None
                        if fr_full is not None:
                            built = _assemble_structural(window_prices, sleeve, t, ids_ok,
                                                         cfg_risk, fr_full, resid_full,
                                                         sleeve_sectors)
                        if built is None:  # non-structural, or too few factor returns yet
                            built = RiskModel.build(window_prices, sleeve, t, ids_ok,
                                                    sectors=sleeve_sectors)
                        rm = built
                        rm_ids = list(rm.ids)
                    except Exception as exc:  # noqa: BLE001
                        warn_list.append(f"risk build {sleeve}@{t.date()} failed ({exc})")
            if rm is None or not rm_ids:
                continue

            # --- monthly score-correlation re-estimate (PIT, cached) ---
            if len(active_factors) >= 2 and (t in month_pts or score_corr is None):
                score_corr = score_correlation(
                    {f: z_panels[f] for f in active_factors}, t, window=252)

            resid_vol = rm.resid_vol.reindex(rm_ids)
            resid_panel = pd.DataFrame({"obs_date": t, "instrument_id": rm_ids,
                                        "value": resid_vol.to_numpy()})

            # --- per-factor alpha at t ---
            alpha_panels: dict = {}
            z_by_factor: dict = {}   # factor -> z(t) panel, for the Grinold combine
            ic_by_factor: dict = {}  # factor -> IC*(t)
            # Quadrature accumulator for the per-name alpha standard error (robust term only):
            # se_i^2 = sum_f (resid_vol_i * se_IC_f * |z_i,f|)^2, aligned like the alpha combine.
            se_sq = pd.Series(0.0, index=rm_ids) if robust_kappa > 0.0 else None
            for f in active_factors:
                ic_star = icstar.get((f, sleeve))
                if ic_star is None or ic_star.empty:
                    continue
                val = ic_star.asof(t)
                if not np.isfinite(val):
                    continue
                z_t = _latest_z_at(z_panels[f], t, rm_ids)
                if z_t.empty:
                    continue
                a_f = refine_alpha(z_t, float(val), resid_panel)
                if not a_f.empty:
                    alpha_panels[f] = a_f
                    z_by_factor[f] = z_t
                    ic_by_factor[f] = float(val)
                    # stand-alone (single-factor) target weights for per-alpha attribution
                    _record_factor_weights(weights_by_factor[f], t, a_f, rm_ids)
                    if se_sq is not None:
                        se_series = icse.get((f, sleeve))
                        se_ic = float(se_series.asof(t)) if (
                            se_series is not None and not se_series.empty) else float("nan")
                        if np.isfinite(se_ic):
                            z_abs = (z_t.set_index("instrument_id")["value"]
                                     .reindex(rm_ids).abs().fillna(0.0))
                            contrib = resid_vol.reindex(rm_ids).fillna(0.0) * se_ic * z_abs
                            se_sq = se_sq.add(contrib ** 2, fill_value=0.0)

            if not alpha_panels:
                continue
            if len(alpha_panels) >= 2 and score_corr is not None:
                # Correlation-aware blend: down-weights redundant factors (no double count).
                # PIT: z_by_factor is already knowable at t; score_corr was estimated <= t.
                combined = combine_alphas_grinold(z_by_factor, ic_by_factor, resid_panel,
                                                  score_corr, ridge=0.10)
            else:
                combined = combine_alphas(alpha_panels)
            alpha_series = (combined.set_index("instrument_id")["value"]
                            .reindex(rm_ids).fillna(0.0))

            # --- costs for the optimizer penalty (representative per-name trade) ---
            adv = _asof_row(adv_panel, t).reindex(rm_ids)
            sigma = _asof_row(sigma_panel, t).reindex(rm_ids)
            pos_cap = cfg["constraints"]["position_cap"][sleeve]
            repr_trade = pd.Series(pos_cap * _NOMINAL_AUM, index=rm_ids)
            cost_opt = cost_model.cost_bps(repr_trade, adv, sigma, sleeve,
                                           instrument_ids=rm_ids)

            alpha_se = None
            if se_sq is not None:
                alpha_se = np.sqrt(se_sq).reindex(rm_ids).fillna(0.0)

            wp = w_prev.reindex(rm_ids).fillna(0.0)
            res = optimize_sleeve(alpha_series, rm, wp, cost_opt, sleeve, cfg,
                                  vol_target=sleeve_vol_target, sectors=sleeve_sectors,
                                  alpha_se=alpha_se)
            w_new = res.w.reindex(rm_ids).fillna(0.0)

            # --- realized cost on the actual trade ---
            dw = (w_new - wp).abs()
            trade_usd = dw * _NOMINAL_AUM
            cost_real = cost_model.cost_bps(trade_usd, adv, sigma, sleeve,
                                            instrument_ids=rm_ids, record=True)
            cost_return = float((dw * cost_real / 1e4).sum())

            rebal_weights[t] = w_new
            rebal_costs[t] = cost_return
            w_prev = w_new

        if rm is not None:
            risk_models[sleeve] = rm  # last-built model for this sleeve (diagnostics)
        if not rebal_weights:
            continue
        Wg = pd.DataFrame(rebal_weights).T.sort_index()
        Wg.index = pd.DatetimeIndex(Wg.index)
        weights_history[sleeve] = Wg

        # --- daily sleeve returns: positions take effect t+1 ---
        daily_idx = rets_wide.index[rets_wide.index >= grid[0]]
        w_daily = (Wg.reindex(columns=[c for c in Wg.columns])
                   .reindex(daily_idx, method="ffill").shift(1).fillna(0.0))
        aligned_rets = rets_wide.reindex(index=daily_idx, columns=Wg.columns).fillna(0.0)
        gross_daily = (w_daily * aligned_rets).sum(axis=1)

        # --- costs charged on the first effective day of each rebalance ---
        cost_daily = pd.Series(0.0, index=daily_idx)
        for t, c in rebal_costs.items():
            future = daily_idx[daily_idx > t]
            if len(future):
                cost_daily.loc[future[0]] += c
        net_daily = (gross_daily - cost_daily).astype(float)

        sleeve_gross[sleeve] = gross_daily
        sleeve_net[sleeve] = net_daily

    if not sleeve_net:
        raise ValueError("no sleeve produced any weights — check data coverage / warmup")

    # ---- blend sleeves into the total book --------------------------------------------
    union_idx = sorted(set().union(*[s.index for s in sleeve_net.values()]))
    union_idx = pd.DatetimeIndex(union_idx)
    net_df = pd.DataFrame({s: sleeve_net[s] for s in sleeve_net}).reindex(union_idx).fillna(0.0)
    gross_df = pd.DataFrame({s: sleeve_gross[s] for s in sleeve_gross}).reindex(union_idx).fillna(0.0)

    # monthly sleeve allocation, forward-filled to daily
    alloc_dates = pd.DatetimeIndex(sorted(month_pts))
    alloc_rows = {}
    for t in alloc_dates:
        alloc_rows[t] = sleeve_allocation(net_df, t, cfg).reindex(net_df.columns).fillna(0.0)
    alloc_df = (pd.DataFrame(alloc_rows).T.sort_index()
                .reindex(union_idx, method="ffill").fillna(0.0))

    total_pre_net = (alloc_df * net_df).sum(axis=1)
    total_pre_gross = (alloc_df * gross_df).sum(axis=1)

    # ---- overlays: scale the day's return by the PIT overlay multiplier ---------------
    realized_dates: list = []
    realized_net: list = []
    equity_vals: list = []
    net_out = np.empty(len(union_idx), dtype=float)
    gross_out = np.empty(len(union_idx), dtype=float)
    mult_out = np.empty(len(union_idx), dtype=float)
    eq = 1.0
    prev_mult: float | None = None   # carried across dates so the vol-target deadband binds
    for k, day in enumerate(union_idx):
        rn = pd.Series(realized_net, index=pd.DatetimeIndex(realized_dates))
        eqs = pd.Series(equity_vals, index=pd.DatetimeIndex(realized_dates))
        mult = overlay_multiplier(rn, eqs, macro_panel, day, cfg,
                                  prev_multiplier=prev_mult)
        prev_mult = mult
        net = float(total_pre_net.iloc[k] * mult)
        gross = float(total_pre_gross.iloc[k] * mult)
        net_out[k] = net
        gross_out[k] = gross
        mult_out[k] = mult
        eq *= (1.0 + net)
        realized_dates.append(day)
        realized_net.append(net)
        equity_vals.append(eq)

    total_returns = pd.Series(net_out, index=union_idx).fillna(0.0)
    total_gross_returns = pd.Series(gross_out, index=union_idx).fillna(0.0)
    overlay = pd.Series(mult_out, index=union_idx)
    costs = (total_gross_returns - total_returns).rename("cost_drag")

    # ---- attribution inputs -----------------------------------------------------------
    all_ids = sorted(traded.index.tolist())
    inst_returns = _prices_wide(prices, all_ids).pct_change().reindex(union_idx).fillna(0.0)
    factor_returns = _attribution_factor_returns(prices, sleeve_ids, union_idx, structural_fr)
    wbf = {f: (pd.DataFrame(d).T.sort_index() if d else pd.DataFrame())
           for f, d in weights_by_factor.items()}
    ic_rvt = _ic_realized_vs_training(ic_tables, icstar, factor_sleeves, grid[0], end)

    # Counterfactual cost sensitivity from the realized-trade ledger (alpha=0.15 unchanged —
    # a CLAUDE.md hard rule; this is reporting-only, see research-cost-model-calibration.md).
    cost_sensitivity = cost_model.cost_sensitivity() if cost_model.ledger else None

    result = BacktestResult(
        total_returns=total_returns,
        total_gross_returns=total_gross_returns,
        sleeve_returns=net_df,
        weights_history=weights_history,
        costs=costs,
        overlay=overlay,
        risk_models=risk_models,
        _factor_returns=factor_returns,
        _weights_by_factor=wbf,
        _inst_returns=inst_returns,
        _ic_realized_vs_training=ic_rvt,
        _factors_used=list(z_panels.keys()),
        _gate_applied=gate_applied,
        _warnings=warn_list,
        _cost_sensitivity=cost_sensitivity,
    )
    result.report = build_report(result, cfg, registry)
    return result


def latest_target_weights(data: dict, instruments: pd.Series, cfg: dict | None = None,
                          factors_cfg=None, result: "BacktestResult | None" = None,
                          sectors: pd.Series | None = None) -> pd.Series:
    """Book-level target weights on the most recent rebalance grid date.

    Thin wrapper over :func:`run_backtest`: it runs the identical walk-forward (so the
    weights come out of exactly the PIT machinery the backtest already validates — no
    parallel, un-tested decision path lives in the live runner), then blends each
    sleeve's *final* rebalance weights by the latest monthly sleeve allocation and the
    latest overlay multiplier into a single fraction-of-book weight per instrument.

    Returns a Series indexed by instrument_id (gross exposure <= 1), non-zero entries
    only. Pass a precomputed ``result`` to avoid re-running the backtest.
    """
    if cfg is None:
        cfg = backtest_config()
    if result is None:
        result = run_backtest(data, instruments, cfg=cfg, factors_cfg=factors_cfg,
                              sectors=sectors)

    weights_history = result.weights_history
    if not weights_history or result.sleeve_returns.empty:
        return pd.Series(dtype=float)

    last_day = result.sleeve_returns.index.max()
    alloc = sleeve_allocation(result.sleeve_returns, last_day, cfg)
    overlay_mult = float(result.overlay.iloc[-1]) if len(result.overlay) else 1.0

    parts: list = []
    for sleeve, Wg in weights_history.items():
        if Wg is None or Wg.empty:
            continue
        w_last = Wg.iloc[-1].dropna()
        a = float(alloc.get(sleeve, 0.0))
        if a == 0.0:
            continue
        parts.append(w_last * a * overlay_mult)
    if not parts:
        return pd.Series(dtype=float)
    book = pd.concat(parts).groupby(level=0).sum()
    return book[book != 0.0]


def _record_factor_weights(store: dict, t, a_f: pd.DataFrame, rm_ids) -> None:
    """Stash a factor's stand-alone target weights at ``t`` for per-alpha attribution.

    Approximation (documented in attribution.py): rather than a full single-factor
    optimization each rebalance, we take the factor's refined alpha and normalize it to
    unit gross exposure per rebalance. It captures the factor's directional tilt cheaply;
    it is not an exact P&L split of the jointly-optimized book.
    """
    a = a_f.set_index("instrument_id")["value"].reindex(rm_ids).fillna(0.0)
    gross = a.abs().sum()
    store[t] = (a / gross) if gross > 0 else a


def _attribution_factor_returns(prices, sleeve_ids, union_idx, structural_fr) -> pd.DataFrame:
    """Risk-factor return basis for attribution (equity sleeve's factors, v1).

    Reuses the equity sleeve's already-estimated full-span factor returns (no second
    estimation pass); the total book is regressed on market/size/momentum/vol. Falls back
    to any available structural sleeve, else an empty frame.
    """
    fr = structural_fr.get("equity")
    if fr is None:
        for s in ("crypto",):
            if s in structural_fr:
                fr = structural_fr[s]
                break
    if fr is None or fr.empty:
        return pd.DataFrame()
    return fr.reindex(union_idx).fillna(0.0)


def _ic_realized_vs_training(ic_tables, icstar, factor_sleeves, test_start, test_end) -> dict:
    """Per factor: realized test-window rank IC vs the training IC* actually applied."""
    out: dict = {}
    for f, ic in ic_tables.items():
        window = ic[(ic["obs_date"] >= test_start) & (ic["obs_date"] <= test_end)]
        realized = float(window["rank_ic"].mean()) if not window.empty else float("nan")
        training_vals = []
        for sleeve in factor_sleeves.get(f, ()):  # applied IC* over the window
            s = icstar.get((f, sleeve))
            if s is None:
                continue
            seg = s[(s.index >= test_start) & (s.index <= test_end)].dropna()
            if len(seg):
                training_vals.append(seg.mean())
        training = float(np.mean(training_vals)) if training_vals else float("nan")
        out[f] = {"realized": realized, "training": training}
    return out
