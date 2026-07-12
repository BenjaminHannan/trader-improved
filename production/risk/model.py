"""The per-sleeve risk model — assembles exposures, factor covariance and specific
risk into a covariance matrix, and validates estimated factor returns against Ken
French benchmarks.

Two structures (per configs/risk.yaml):
- Structural sleeves (equity, crypto): ``Sigma = B F B' + diag(D)`` where ``B`` is
  the exposure matrix as of ``as_of``, ``F`` is the EWMA covariance of estimated
  factor returns, and ``D`` is specific (idiosyncratic) daily variance. If there
  are too few factor-return observations to estimate ``F`` (< min_obs), we fall
  back to the small-sleeve full-covariance mode.
- Small sleeves (fx_etf, commodity_etf): a shrunk EWMA covariance of instrument
  returns directly — N is tiny and a factor structure would just fit noise.

PIT: every estimator reads ``obs_date <= as_of`` data only (delegated to the
exposures / factor-return modules and to a strict trailing pivot here).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from production.core.config import risk_config
from production.risk.covariance import RiskError, ewma_cov, ledoit_wolf_shrinkage
from production.risk.exposures import build_exposures
from production.risk.factor_returns import estimate_factor_returns
from production.risk.specific import specific_vol

_STRUCTURAL = ("equity", "crypto")
_TRADING_DAYS = 252


def _shrunk_cov(returns: pd.DataFrame, cfg: dict, min_obs: int) -> pd.DataFrame:
    """Dispatch a covariance estimate by ``cfg['method']``.

    ``fixed`` (default) — the legacy diagonally-shrunk ``ewma_cov`` path, byte-for-
    byte unchanged for reproducibility. ``lw`` / ``lw_cc`` — Ledoit-Wolf analytic
    shrinkage with EWMA-weighted moments toward the diagonal / constant-correlation
    target respectively. ``min_obs`` is enforced identically across methods.
    """
    method = str(cfg.get("method", "fixed")).lower()
    halflife = float(cfg.get("ewma_halflife_days", 90))
    if method == "fixed":
        Sigma = ewma_cov(returns, halflife=halflife,
                         shrink=float(cfg.get("shrinkage_to_diagonal", 0.3)),
                         min_obs=min_obs)
    elif method in ("lw", "lw_cc"):
        target = "diagonal" if method == "lw" else "constant_correlation"
        Sigma, _delta = ledoit_wolf_shrinkage(
            returns, target=target, ewma_halflife=halflife, min_obs=min_obs)
    else:
        raise RiskError(f"_shrunk_cov: unknown covariance method {method!r}")

    blend = cfg.get("anchor_blend")
    if blend:
        # An EWMA projects its current level flat across the horizon — no
        # long-run anchor, which understates 21d risk after calm stretches and
        # overstates it after spikes. Blend toward the trailing equal-weight
        # covariance: a convex combination of PSD estimates stays PSD. The
        # anchor leg reuses ewma_cov with an effectively-infinite halflife so
        # its NaN-masking/symmetrize/jitter hygiene is identical to the fast
        # leg's. Shorter histories than the window use whatever is available
        # past min_obs.
        w = float(blend["weight"])
        long_n = int(blend.get("long_window_days", 756))
        anchor = ewma_cov(returns.iloc[-long_n:], halflife=1e12, shrink=0.0,
                          min_obs=min_obs)
        Sigma = w * Sigma + (1.0 - w) * anchor.reindex(index=Sigma.index,
                                                       columns=Sigma.columns)
    return Sigma


def _instrument_returns(prices: pd.DataFrame, as_of, ids) -> pd.DataFrame:
    """Strictly-PIT wide daily returns (obs_date <= as_of) for ``ids``."""
    as_of = pd.Timestamp(as_of)
    df = prices[prices["instrument_id"].isin(list(ids))]
    df = df[pd.to_datetime(df["obs_date"]) <= as_of]
    if df.empty:
        return pd.DataFrame(columns=list(ids))
    close = df.pivot_table(index="obs_date", columns="instrument_id",
                           values="close", aggfunc="last").sort_index()
    return close.reindex(columns=list(ids)).pct_change()


class RiskModel:
    """Assembled covariance model for one sleeve. See module docstring."""

    def __init__(self, sleeve: str, ids: pd.Index,
                 B: pd.DataFrame | None = None, F: pd.DataFrame | None = None,
                 D: pd.Series | None = None, _cov: pd.DataFrame | None = None,
                 resid_vol: pd.Series | None = None):
        self.sleeve = sleeve
        self.ids = pd.Index(ids)
        self.B = B
        self.F = F
        self.D = D  # specific daily VARIANCE
        self._cov = _cov
        self.resid_vol = resid_vol  # daily vol

    # ------------------------------------------------------------------ build
    @classmethod
    def build(cls, prices: pd.DataFrame, sleeve: str, as_of, ids,
              cfg: dict | None = None, sectors: pd.Series | None = None) -> "RiskModel":
        if cfg is None:
            cfg = risk_config()
        ids = pd.Index(list(ids), name="instrument_id")
        as_of = pd.Timestamp(as_of)

        fac_cfg = cfg.get("factor_covariance", {})
        inst_cfg = cfg.get("instrument_covariance", {})
        spec_cfg = cfg.get("specific_risk", {})

        if sleeve in _STRUCTURAL:
            model = cls._build_structural(prices, sleeve, as_of, ids, cfg,
                                          sectors, fac_cfg, spec_cfg, inst_cfg)
            if model is not None:
                return model
            # fall through to full-cov fallback (too few factor-return obs)

        _cov = _shrunk_cov(
            _instrument_returns(prices, as_of, ids),
            inst_cfg,
            min_obs=int(inst_cfg.get("min_obs", 60)),
        ).reindex(index=ids, columns=ids)
        resid_vol = pd.Series(np.sqrt(np.diag(_cov.to_numpy())), index=ids,
                              name="resid_vol")
        return cls(sleeve, ids, _cov=_cov, resid_vol=resid_vol)

    @classmethod
    def _build_structural(cls, prices, sleeve, as_of, ids, cfg, sectors,
                          fac_cfg, spec_cfg, inst_cfg) -> "RiskModel | None":
        """Return a structural model, or None if factor returns are too short."""
        B = build_exposures(prices, sleeve, as_of, ids, cfg, sectors)

        # Restrict to this sleeve's names — prices carries no sleeve column, so
        # factor-return estimation must see only the sleeve's cross-section.
        sleeve_prices = prices[prices["instrument_id"].isin(list(ids))]
        start = pd.to_datetime(sleeve_prices["obs_date"]).min()
        factor_returns, residuals = estimate_factor_returns(
            sleeve_prices, sleeve, start, as_of, cfg, sectors)

        min_obs = int(fac_cfg.get("min_obs", 252))
        if len(factor_returns) < min_obs:
            return None

        F = _shrunk_cov(
            factor_returns, fac_cfg, min_obs=min_obs,
        ).reindex(index=B.columns, columns=B.columns)

        sv = specific_vol(
            residuals.reindex(columns=ids),
            halflife=float(spec_cfg.get("ewma_halflife_days", 42)),
            shrink_weight=float(spec_cfg.get("cross_sectional_shrink_weight", 0.25)),
            min_obs=int(spec_cfg.get("min_obs", 63)),
        ).reindex(ids)
        sv = sv.fillna(sv.median() if sv.notna().any() else 0.0)
        D = (sv ** 2).rename("D")
        return cls(sleeve, ids, B=B, F=F, D=D, resid_vol=sv.rename("resid_vol"))

    # ------------------------------------------------------------- covariance
    def covariance(self) -> pd.DataFrame:
        """Daily N x N covariance: ``B F B' + diag(D)`` (structural) or ``_cov``."""
        if self._cov is not None:
            return self._cov
        B = self.B.reindex(index=self.ids)
        F = self.F.reindex(index=B.columns, columns=B.columns)
        common = B @ F @ B.T
        cov = np.array(common.to_numpy(dtype=float), copy=True)
        cov[np.diag_indices_from(cov)] += self.D.reindex(self.ids).to_numpy(dtype=float)
        cov = 0.5 * (cov + cov.T)
        return pd.DataFrame(cov, index=self.ids, columns=self.ids)

    def portfolio_vol(self, w: pd.Series) -> float:
        """Annualized portfolio vol ``sqrt(252 * w' Σ w)``; ids missing from ``w``
        get zero weight (aligned on the model's ids)."""
        cov = self.covariance()
        wv = pd.Series(w).reindex(cov.index).fillna(0.0).to_numpy(dtype=float)
        var = float(wv @ cov.to_numpy(dtype=float) @ wv)
        var = max(var, 0.0)
        return float(np.sqrt(_TRADING_DAYS * var))

    def factor_form(self) -> tuple | None:
        """``(B, F, D)`` for structural sleeves, ``None`` for small sleeves."""
        if self.B is None:
            return None
        return (self.B, self.F, self.D)


def validate_against_french(factor_returns: pd.DataFrame,
                            french_monthly: pd.DataFrame) -> dict[str, float]:
    """Correlate estimated factor returns with Ken French monthly benchmarks.

    Compounds the daily estimated factor returns to monthly, aligns on calendar
    month, and correlates ``market`` with ``Mkt-RF`` and ``momentum`` with
    ``Mom``. Returns ``{factor: correlation}`` for whichever pairs are present.
    Pure function — real French data is fetched/loaded elsewhere.
    """
    mapping = {"market": "Mkt-RF", "momentum": "Mom"}

    fr = factor_returns.copy()
    fr.index = pd.to_datetime(fr.index)
    monthly = (1.0 + fr).groupby(fr.index.to_period("M")).prod() - 1.0

    fm = french_monthly.copy()
    fm.index = pd.PeriodIndex(pd.to_datetime(fm.index), freq="M") \
        if not isinstance(fm.index, pd.PeriodIndex) else fm.index

    out: dict[str, float] = {}
    for factor, bench in mapping.items():
        if factor not in monthly.columns or bench not in fm.columns:
            continue
        joined = pd.concat([monthly[factor], fm[bench]], axis=1, join="inner").dropna()
        if len(joined) < 3:
            continue
        out[factor] = float(joined.iloc[:, 0].corr(joined.iloc[:, 1]))
    return out
