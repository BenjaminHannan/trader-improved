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


def vol_target_multiplier(daily_returns: pd.Series, t, target: float = 0.10,
                          lookback: int = 21, clip: tuple = (0.0, 1.5)) -> float:
    """target / realized_vol over the last `lookback` returns strictly before t.

    Scales up when recent realized vol is below target and down when above, clipped.
    """
    r = daily_returns[daily_returns.index < t].dropna()
    if len(r) < lookback:
        return 1.0
    window = r.iloc[-lookback:]
    realized = float(window.std(ddof=1)) * np.sqrt(252.0)
    if not np.isfinite(realized) or realized <= 0.0:
        return 1.0
    return float(np.clip(target / realized, clip[0], clip[1]))


def drawdown_multiplier(equity: pd.Series, t, threshold: float = 0.08,
                        scale: float = 0.5) -> float:
    """Scale gross by `scale` once running drawdown (using equity strictly before t) exceeds threshold."""
    e = equity[equity.index < t].dropna()
    if len(e) < 1:
        return 1.0
    peak = float(e.cummax().iloc[-1])
    cur = float(e.iloc[-1])
    if peak <= 0.0:
        return 1.0
    dd = 1.0 - cur / peak
    return float(scale) if dd > threshold else 1.0


def macro_derisk_multiplier(macro_panel: pd.DataFrame, t,
                            series=("BAMLH0A0HYM2", "VIXCLS"),
                            z_window: int = 252, z_trigger: float = 1.5,
                            scale: float = 0.6) -> float:
    """De-risk when a macro-stress z-score breaches its trigger.

    Causality: only rows with `available_from <= t` are visible (future vintages, even if
    their obs_date <= t, are excluded and corruption of them cannot move the answer). The
    z-score of the latest value is measured against a rolling mean/std ENDING at the
    previous observation (`shift(1)`), so the latest value never inflates its own baseline.
    If the mean z across the requested series exceeds `z_trigger`, return `scale`.
    """
    tt = pd.Timestamp(t)
    if tt.tzinfo is None:
        tt = tt.tz_localize("UTC")
    af = pd.to_datetime(macro_panel["available_from"])
    if af.dt.tz is None:
        af = af.dt.tz_localize("UTC")
    visible = macro_panel[af.to_numpy() <= tt]

    zs: list[float] = []
    for sid in series:
        sub = visible[visible["series_id"] == sid].sort_values("obs_date")
        vals = sub["value"].reset_index(drop=True)
        if len(vals) < z_window + 1:
            continue
        mean = vals.rolling(z_window).mean().shift(1)
        std = vals.rolling(z_window).std().shift(1)
        m, s = mean.iloc[-1], std.iloc[-1]
        if not np.isfinite(m) or not np.isfinite(s) or s <= 0.0:
            continue
        zs.append((float(vals.iloc[-1]) - float(m)) / float(s))

    if not zs:
        return 1.0
    return float(scale) if float(np.mean(zs)) > z_trigger else 1.0


def overlay_multiplier(daily_returns: pd.Series, equity: pd.Series,
                       macro_panel: pd.DataFrame, t, cfg: dict) -> float:
    """Product of the three overlays, driven by cfg['overlays']."""
    ov = cfg["overlays"]
    vt = ov["vol_target"]
    dc = ov["drawdown_control"]
    md = ov["macro_derisk"]

    m_vol = vol_target_multiplier(daily_returns, t, vt["target"],
                                  vt["lookback_days"], tuple(vt["clip"]))
    m_dd = drawdown_multiplier(equity, t, dc["threshold"], dc["scale"])
    m_macro = 1.0
    if md.get("enabled", True):
        m_macro = macro_derisk_multiplier(macro_panel, t, tuple(md["series"]),
                                          md["z_window_days"], md["z_trigger"], md["scale"])
    return float(m_vol * m_dd * m_macro)
