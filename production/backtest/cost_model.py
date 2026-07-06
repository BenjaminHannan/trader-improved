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

    def __init__(self, cfg: dict | None = None, overrides_table: dict | None = None) -> None:
        self.cfg = cfg if cfg is not None else costs_config()
        self.alpha = float(self.cfg["impact_alpha"])
        self.cap_bps = float(self.cfg["cap_bps"])
        self.sleeves = self.cfg["sleeves"]
        # yaml instrument_overrides are the base; lake-calibrated overrides (from TCA
        # feedback) merge OVER them per instrument key — the calibrated entry wins. When
        # overrides_table is None behavior is bit-identical to the yaml-only path.
        self.overrides = dict(self.cfg.get("instrument_overrides") or {})
        if overrides_table:
            self.overrides = {**self.overrides, **overrides_table}
        # Every capped / bad-ADV name lands here; the audit layer surfaces it. Never silent.
        self.cap_warnings: list[dict] = []
        # Per-charge decomposition ledger — one entry per `cost_bps(..., record=True)` call
        # (i.e. one rebalance's realized trades). Feeds `cost_sensitivity`; off by default so
        # the optimizer's representative-trade probes never pollute the realized-cost tally.
        self.ledger: list[dict] = []

    def _floor_hs(self, instrument_id, sleeve: str) -> tuple[float, float]:
        """Resolve (floor_bps, half_spread_bps): per-instrument override wins over sleeve default.

        Overrides may be partial — a TCA-calibrated entry carries only ``half_spread_bps``
        (floors are floors, never recalibrated), so each field falls back to the sleeve
        default when the override omits it. Extra metadata keys (n_fills, …) are ignored.
        """
        spec = self.sleeves[sleeve]
        floor = float(spec["floor_bps"])
        hs = float(spec["half_spread_bps"])
        if instrument_id is not None and instrument_id in self.overrides:
            o = self.overrides[instrument_id]
            if "floor_bps" in o:
                floor = float(o["floor_bps"])
            if "half_spread_bps" in o:
                hs = float(o["half_spread_bps"])
        return floor, hs

    def cost_bps(self, trade_usd, adv_usd, sigma_daily, sleeve,
                 instrument_ids=None, record: bool = False) -> pd.Series:
        """Cost in bps per name. Accepts scalars or aligned Series; returns a Series.

        - zero (or NaN) trade -> 0 cost (not trading is free);
        - missing / non-positive ADV -> cap_bps (unknown liquidity is treated as worst case);
        - otherwise max(floor, half_spread + sqrt-impact), hard-capped at cap_bps.

        When ``record`` is true, the per-name floor / half-spread / impact / notional / charged
        vectors and their notional-weighted aggregates are appended to ``self.ledger`` (one
        entry per call). The returned Series is identical whether or not recording is on.
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
        # Per-name decomposition, stored so a counterfactual charge max(floor, spread + m*impact)
        # capped at cap_bps reproduces the realized charge exactly at m=1 (see cost_sensitivity).
        floors_v = np.zeros(len(idx), dtype=float)
        spreads_v = np.zeros(len(idx), dtype=float)
        impacts_v = np.zeros(len(idx), dtype=float)
        notional_v = np.zeros(len(idx), dtype=float)
        for i, iid in enumerate(idx):
            s = sl.iloc[i]
            floor, hs = self._floor_hs(iid, s)
            tr, a, sg = trade[i], adv[i], sigma[i]
            if not np.isfinite(tr) or tr == 0.0:
                out[i] = 0.0
                continue
            notional_v[i] = tr
            if not np.isfinite(a) or a <= 0.0:
                out[i] = self.cap_bps
                # Unknown liquidity: charged at the cap and invariant to the impact prefactor.
                floors_v[i] = self.cap_bps
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
            floors_v[i] = floor
            spreads_v[i] = hs
            impacts_v[i] = impact
        if record:
            floor_bound = np.maximum(floors_v, spreads_v + impacts_v) <= floors_v
            self.ledger.append({
                "sleeve": sleeve if isinstance(sleeve, str) else list(sl),
                "floor": floors_v.tolist(),
                "spread": spreads_v.tolist(),
                "impact": impacts_v.tolist(),
                "notional": notional_v.tolist(),
                "charged": out.tolist(),
                "floor_bound_notional": float((notional_v * floor_bound).sum()),
                "spread_bps_x_notional": float((spreads_v * notional_v).sum()),
                "impact_bps_x_notional": float((impacts_v * notional_v).sum()),
                "floor_bps_x_notional": float((floors_v * notional_v).sum()),
                "total_notional": float(notional_v.sum()),
                "charged_bps_x_notional": float((out * notional_v).sum()),
            })
        return pd.Series(out, index=idx)

    def cost_sensitivity(self, multipliers=(1.0, 10.0 / 3.0, 20.0 / 3.0)) -> dict:
        """Counterfactual annualized-drag-scaling ratios by impact-prefactor multiplier.

        For each multiplier ``m`` the realized per-name impact is scaled by ``m`` (holding
        floors and half-spreads fixed) and re-charged as ``min(max(floor, spread + m*impact),
        cap)``; the notional-weighted total is divided by the realized notional-weighted total.
        ``m = 1.0`` reproduces the realized charge and therefore reconciles to exactly 1.0;
        floor-bound trades are invariant to ``m``; the ratio is monotone nondecreasing in ``m``.

        Returns ``{m: counterfactual_total_bps_x_notional / realized_total_bps_x_notional}``.
        The traded costs (alpha=0.15, a CLAUDE.md hard rule) are never altered — this is a
        reporting-only sensitivity; alpha=0.15 sits below the 0.5-1.0 literature band
        (research/wiki/questions/research-cost-model-calibration.md).
        """
        realized = sum(e["charged_bps_x_notional"] for e in self.ledger)
        out: dict = {}
        for m in multipliers:
            cf_total = 0.0
            for e in self.ledger:
                floors = np.asarray(e["floor"], dtype=float)
                spreads = np.asarray(e["spread"], dtype=float)
                impacts = np.asarray(e["impact"], dtype=float)
                notional = np.asarray(e["notional"], dtype=float)
                cf = np.minimum(np.maximum(floors, spreads + m * impacts), self.cap_bps)
                cf_total += float((cf * notional).sum())
            out[m] = (cf_total / realized) if realized else float("nan")
        return out


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
