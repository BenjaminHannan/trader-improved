"""Exposure overlays — pure PIT, multiplicative, composed by product.

Each overlay returns a scalar in [0, cap] that scales *gross* exposure at decision date
`t`. All three are strictly trailing: they may look only at history dated before `t`
(returns/equity indexed strictly `< t`; macro rows with `available_from <= t`), and any
rolling statistic that would otherwise include the current observation is `shift(1)`-ed so
a value at `t` never enters its own baseline. Insufficient history -> neutral 1.0.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from production.core.pit import asof_panel


def vol_target_multiplier(daily_returns: pd.Series, t, target: float = 0.10,
                          lookback: int = 21, clip: tuple = (0.0, 1.5),
                          ewma_halflife: float | None = None,
                          prev_multiplier: float | None = None,
                          deadband: float = 0.0) -> float:
    """target / realized_vol over the trailing returns strictly before t.

    Scales up when recent realized vol is below target and down when above, clipped.

    Realized-vol estimator (strictly trailing, rows dated `< t` only):
      - `ewma_halflife is None`  -> sample std over the last `lookback` returns
        (the original raw rolling-window behavior);
      - `ewma_halflife` set      -> EWMA std (exponentially-weighted, `min_periods=lookback`)
        over ALL trailing returns, centred at the same responsiveness but smoothed.
        Smoothing reduces multiplier whipsaw (arXiv 2212.07288).

    Hysteresis deadband: when `prev_multiplier` is given and the newly computed
    multiplier is within `deadband` fractional distance of it (|new/prev - 1| <= deadband),
    the previous multiplier is returned unchanged — leverage only moves when it matters,
    which attacks the turnover-cost erosion channel (Barroso & Detzel 2021). `deadband=0.0`
    (or no `prev_multiplier`) reproduces the un-hysteretic path bit-for-bit.
    """
    r = daily_returns[daily_returns.index < t].dropna()
    if len(r) < lookback:
        return 1.0
    if ewma_halflife is None:
        window = r.iloc[-lookback:]
        realized = float(window.std(ddof=1)) * np.sqrt(252.0)
    else:
        ew = r.ewm(halflife=float(ewma_halflife), min_periods=lookback).std(bias=False)
        realized = float(ew.iloc[-1]) * np.sqrt(252.0)
    if not np.isfinite(realized) or realized <= 0.0:
        return 1.0
    new = float(np.clip(target / realized, clip[0], clip[1]))
    if prev_multiplier is not None and deadband > 0.0:
        p = float(prev_multiplier)
        if p != 0.0 and abs(new / p - 1.0) <= deadband:
            return p
    return new


def drawdown_multiplier(equity: pd.Series, t, threshold: float = 0.08,
                        scale: float = 0.5, mode: str = "step",
                        recovery_frac: float = 0.75,
                        prev_multiplier: float | None = None) -> float:
    """Scale gross exposure as running drawdown (equity strictly before t) deepens.

    Drawdown `dd = 1 - cur/peak` is measured on equity dated strictly `< t` (PIT:
    corruption of equity at/after `t` cannot move the answer).

    `mode="step"` (default, back-compat): binary rule — return `scale` when `dd > threshold`,
    else 1.0. Bit-identical to the original overlay.

    `mode="ramp"`: graduated de-risking (risk-control-index style, less whipsaw):
      - `dd < threshold`            -> 1.0;
      - `threshold <= dd <= 2*threshold` -> linear interpolation from 1.0 at `threshold`
        down to `scale` at `2*threshold`;
      - `dd > 2*threshold`          -> `scale`.

    Hysteresis (ramp mode only, when `prev_multiplier` is not None and `prev_multiplier < 1.0`,
    i.e. we were already de-risked): hold `min(prev_multiplier, ramp_value)` — never re-risk
    at the same boundary, but still allow further de-risking if the drawdown deepens — until
    the drawdown recovers below `threshold * recovery_frac`, at which point release to 1.0.
    This re-entry asymmetry attacks the whipsaw-cost channel when equity oscillates around
    the trigger. Absent `prev_multiplier` (or `>= 1.0`) reproduces the memoryless ramp.
    """
    e = equity[equity.index < t].dropna()
    if len(e) < 1:
        return 1.0
    peak = float(e.cummax().iloc[-1])
    cur = float(e.iloc[-1])
    if peak <= 0.0:
        return 1.0
    dd = 1.0 - cur / peak
    if mode == "step":
        return float(scale) if dd > threshold else 1.0

    # ramp
    if dd <= threshold:
        ramp = 1.0
    elif dd >= 2.0 * threshold:
        ramp = float(scale)
    else:
        ramp = 1.0 + (float(scale) - 1.0) * (dd - threshold) / threshold

    if prev_multiplier is not None and float(prev_multiplier) < 1.0:
        if dd < threshold * recovery_frac:
            return 1.0
        return float(min(float(prev_multiplier), ramp))
    return float(ramp)


def _causal_z(vals: pd.Series, z_window: int) -> float | None:
    """Trailing, shift(1) z-score of the latest value in `vals`.

    The rolling mean/std end at the PREVIOUS observation, so the latest value never
    enters its own baseline. Returns None when there is insufficient history or the
    trailing std is non-finite / non-positive.
    """
    vals = vals.reset_index(drop=True)
    if len(vals) < z_window + 1:
        return None
    mean = vals.rolling(z_window).mean().shift(1)
    std = vals.rolling(z_window).std().shift(1)
    m, s = mean.iloc[-1], std.iloc[-1]
    if not np.isfinite(m) or not np.isfinite(s) or s <= 0.0:
        return None
    return (float(vals.iloc[-1]) - float(m)) / float(s)


def macro_derisk_multiplier(macro_panel: pd.DataFrame, t,
                            series=("BAMLH0A0HYM2", "VIXCLS"),
                            z_window: int = 252, z_trigger: float = 1.5,
                            scale: float = 0.6,
                            ratio_pairs: list[list[str]] | None = None) -> float:
    """De-risk when a macro-stress z-score breaches its trigger.

    Causality: the visible frame is reduced through the sanctioned PIT join
    `core.pit.asof_panel(macro_panel, t)`, which does two things at once: (1) keeps only
    rows with `available_from <= t` (future vintages, even if their obs_date <= t, are
    excluded and corruption of them cannot move the answer), and (2) collapses each
    `(obs_date, series_id)` to its latest *visible* vintage. Step (2) matters for
    multi-vintage ALFRED macro series (an original release plus later revisions of the
    same obs_date): without it, superseded vintages would appear as duplicate obs_dates
    and distort the causal-z window. On single-vintage data the dedup is a no-op, so the
    result is bit-identical to the plain `available_from <= t` filter.

    The z-score of the latest value is measured against a rolling mean/std ENDING at the
    previous observation (`shift(1)`), so the latest value never inflates its own baseline.
    If the mean z across the requested series exceeds `z_trigger`, return `scale`.

    `ratio_pairs` (optional, list of `[num, den]` series-id pairs): for each pair, the PIT
    ratio series is built by aligning the two legs' visible histories on `obs_date`
    (`ratio = num / den`) and its causal z is included in the composite z alongside the
    level series above. This captures term-structure / spread signals (e.g. VIX/VIX3M)
    whose *level* legs individually carry a weaker or contrarian signal. A pair whose
    aligned history is too short (or whose trailing std degenerates) is skipped, exactly
    as an insufficient level series is. Absent `ratio_pairs` -> level-only behavior,
    bit-for-bit unchanged.
    """
    # Sanctioned PIT join: visibility filter (available_from <= t) AND latest-visible-
    # vintage dedup per (obs_date, series_id) in one call.
    visible = asof_panel(macro_panel, t)

    zs: list[float] = []
    for sid in series:
        sub = visible[visible["series_id"] == sid].sort_values("obs_date")
        z = _causal_z(sub["value"], z_window)
        if z is not None:
            zs.append(z)

    for pair in (ratio_pairs or []):
        num_id, den_id = pair
        num = (visible[visible["series_id"] == num_id]
               .sort_values("obs_date")[["obs_date", "value"]]
               .rename(columns={"value": "num"}))
        den = (visible[visible["series_id"] == den_id]
               .sort_values("obs_date")[["obs_date", "value"]]
               .rename(columns={"value": "den"}))
        aligned = num.merge(den, on="obs_date", how="inner").sort_values("obs_date")
        aligned = aligned[aligned["den"] != 0.0]
        ratio = (aligned["num"] / aligned["den"])
        z = _causal_z(ratio, z_window)
        if z is not None:
            zs.append(z)

    if not zs:
        return 1.0
    return float(scale) if float(np.mean(zs)) > z_trigger else 1.0


def overlay_components(daily_returns: pd.Series, equity: pd.Series,
                       macro_panel: pd.DataFrame, t, cfg: dict,
                       prev_multiplier: float | None = None,
                       prev_components: dict | None = None) -> dict:
    """Compute the three overlays and their product, driven by cfg['overlays'].

    Returns `{"vol_target": m1, "drawdown": m2, "macro": m3, "product": m1*m2*m3}` so a
    caller can carry the per-overlay previous state (needed to run both the vol-target
    deadband and the drawdown hysteresis independently — a single scalar product cannot be
    decomposed back into its factors).

    Previous-state resolution:
      - `prev_components` (preferred): dict with optional keys `"vol_target"` and
        `"drawdown"`, each the corresponding multiplier from the previous decision date;
      - `prev_multiplier` (deprecated back-compat): a single scalar honored ONLY as the
        previous *vol-target* multiplier, and ONLY when `prev_components` is None. This
        matches the engine's legacy call, which passed the previous overlay product as the
        vol-target hysteresis seed. In that legacy path the drawdown overlay gets no prev
        state (memoryless ramp / step).

    `overlays.drawdown_control.mode` (absent -> "step", back-compat) and `recovery_frac`
    (absent -> 0.75) drive the drawdown overlay shape.
    """
    ov = cfg["overlays"]
    vt = ov["vol_target"]
    dc = ov["drawdown_control"]
    md = ov["macro_derisk"]

    prev_vt: float | None = None
    prev_dd: float | None = None
    if prev_components is not None:
        prev_vt = prev_components.get("vol_target")
        prev_dd = prev_components.get("drawdown")
    elif prev_multiplier is not None:
        prev_vt = prev_multiplier  # legacy: scalar seed is the vol-target prev only

    m_vol = vol_target_multiplier(daily_returns, t, vt["target"],
                                  vt["lookback_days"], tuple(vt["clip"]),
                                  ewma_halflife=vt.get("ewma_halflife"),
                                  prev_multiplier=prev_vt,
                                  deadband=vt.get("deadband", 0.0))
    m_dd = drawdown_multiplier(equity, t, dc["threshold"], dc["scale"],
                               mode=dc.get("mode", "step"),
                               recovery_frac=dc.get("recovery_frac", 0.75),
                               prev_multiplier=prev_dd)
    m_macro = 1.0
    if md.get("enabled", True):
        m_macro = macro_derisk_multiplier(macro_panel, t, tuple(md["series"]),
                                          md["z_window_days"], md["z_trigger"], md["scale"],
                                          ratio_pairs=md.get("ratio_pairs"))
    product = float(m_vol * m_dd * m_macro)
    return {"vol_target": float(m_vol), "drawdown": float(m_dd),
            "macro": float(m_macro), "product": product}


def overlay_multiplier(daily_returns: pd.Series, equity: pd.Series,
                       macro_panel: pd.DataFrame, t, cfg: dict,
                       prev_multiplier: float | None = None,
                       prev_components: dict | None = None) -> float:
    """Product of the three overlays, driven by cfg['overlays'].

    Thin scalar wrapper over `overlay_components` (see there for the full contract). The
    signature is call-compatible with the engine's existing usage: `prev_multiplier`
    (optional) is honored as the previous vol-target multiplier for the deadband hysteresis
    when `prev_components` is absent. Pass `prev_components` to additionally thread the
    drawdown hysteresis state.
    """
    return overlay_components(daily_returns, equity, macro_panel, t, cfg,
                              prev_multiplier=prev_multiplier,
                              prev_components=prev_components)["product"]
