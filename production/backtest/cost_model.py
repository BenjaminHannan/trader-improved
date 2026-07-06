"""Transaction cost model — Almgren-Chriss square-root impact with per-sleeve floors.

    cost_bps = max(floor_bps, half_spread_bps + 1e4 * alpha * sigma * sqrt(|trade_$| / ADV_$))

Everything trailing that feeds the cost of a *same-day* trade is `shift(1)`-ed: ADV
and daily sigma are computed from rows strictly before the decision date. This mirrors
the hard rule in CLAUDE.md — costs are never zero (a per-sleeve floor binds), the impact
term uses trailing/lagged liquidity, and a hard cap at `cap_bps` is recorded as an audit
warning rather than silently swallowed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from production.core.config import costs_config


class CostModel:
    """Per-sleeve square-root cost model with floors, cap, and per-instrument overrides."""

    def __init__(self, cfg: dict | None = None) -> None:
        self.cfg = cfg if cfg is not None else costs_config()
        self.alpha = float(self.cfg["impact_alpha"])
        self.cap_bps = float(self.cfg["cap_bps"])
        self.sleeves = self.cfg["sleeves"]
        self.overrides = self.cfg.get("instrument_overrides") or {}
        # Every capped / bad-ADV name lands here; the audit layer surfaces it. Never silent.
        self.cap_warnings: list[dict] = []

    def _floor_hs(self, instrument_id, sleeve: str) -> tuple[float, float]:
        """Resolve (floor_bps, half_spread_bps): per-instrument override wins over sleeve default."""
        if instrument_id is not None and instrument_id in self.overrides:
            o = self.overrides[instrument_id]
            return float(o["floor_bps"]), float(o["half_spread_bps"])
        spec = self.sleeves[sleeve]
        return float(spec["floor_bps"]), float(spec["half_spread_bps"])

    def cost_bps(self, trade_usd, adv_usd, sigma_daily, sleeve,
                 instrument_ids=None) -> pd.Series:
        """Cost in bps per name. Accepts scalars or aligned Series; returns a Series.

        - zero (or NaN) trade -> 0 cost (not trading is free);
        - missing / non-positive ADV -> cap_bps (unknown liquidity is treated as worst case);
        - otherwise max(floor, half_spread + sqrt-impact), hard-capped at cap_bps.
        """
        # Establish the output index from the first Series-like argument, else instrument_ids.
        idx = None
        for cand in (instrument_ids, trade_usd, adv_usd, sigma_daily):
            if isinstance(cand, pd.Series):
                idx = cand.index
                break
        if idx is None and instrument_ids is not None:
            idx = pd.Index(instrument_ids)
        if idx is None:
            idx = pd.RangeIndex(1)

        def arr(x) -> np.ndarray:
            if isinstance(x, pd.Series):
                return x.reindex(idx).to_numpy(dtype=float)
            return np.broadcast_to(np.asarray(x, dtype=float), (len(idx),)).astype(float)

        trade = np.abs(arr(trade_usd))
        adv = arr(adv_usd)
        sigma = arr(sigma_daily)

        if isinstance(sleeve, pd.Series):
            sl = sleeve.reindex(idx)
        else:
            sl = pd.Series([sleeve] * len(idx), index=idx)

        out = np.empty(len(idx), dtype=float)
        for i, iid in enumerate(idx):
            s = sl.iloc[i]
            floor, hs = self._floor_hs(iid, s)
            tr, a, sg = trade[i], adv[i], sigma[i]
            if not np.isfinite(tr) or tr == 0.0:
                out[i] = 0.0
                continue
            if not np.isfinite(a) or a <= 0.0:
                out[i] = self.cap_bps
                self.cap_warnings.append(
                    {"instrument_id": iid, "sleeve": s, "reason": "missing_adv",
                     "cost_bps": self.cap_bps})
                continue
            sg = 0.0 if not np.isfinite(sg) else sg
            impact = 1e4 * self.alpha * sg * np.sqrt(tr / a)
            cost = max(floor, hs + impact)
            if cost > self.cap_bps:
                self.cap_warnings.append(
                    {"instrument_id": iid, "sleeve": s, "reason": "cap", "raw_bps": cost})
                cost = self.cap_bps
            out[i] = cost
        return pd.Series(out, index=idx)


def trailing_adv_sigma(prices: pd.DataFrame, as_of, window: int = 20
                       ) -> tuple[pd.Series, pd.Series]:
    """Trailing dollar-volume ADV and daily return sigma, EXCLUDING the as_of day.

    Why the exclusion: the cost charged for a trade decided at date D may only use
    liquidity/vol knowable *before* D. We keep rows with obs_date <= as_of (never peek at
    the future), then `shift(1)` so the as_of row itself — its dollar_volume and the return
    that ends on its close — never enters the trailing statistic. Corruption of the as_of
    close/volume (or any later day, already filtered out) therefore cannot move the answer.

    Returns (adv, sigma) as Series indexed by instrument_id.
    """
    as_of = pd.Timestamp(as_of)
    df = prices[prices["obs_date"] <= as_of].sort_values(["instrument_id", "obs_date"])
    advs: dict = {}
    sigs: dict = {}
    for iid, g in df.groupby("instrument_id", sort=False):
        dv = g["dollar_volume"].reset_index(drop=True)
        ret = g["close"].reset_index(drop=True).pct_change()
        # shift(1) drops the as_of (last) row from both statistics.
        adv = dv.shift(1).rolling(window).mean()
        sig = ret.shift(1).rolling(window).std()
        advs[iid] = adv.iloc[-1] if len(adv) else np.nan
        sigs[iid] = sig.iloc[-1] if len(sig) else np.nan
    return pd.Series(advs, dtype=float), pd.Series(sigs, dtype=float)
