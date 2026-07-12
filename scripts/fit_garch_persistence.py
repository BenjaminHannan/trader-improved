"""Iteration-10 registration step: derive the anchor-blend weight w.

Fits a variance-targeted GARCH(1,1) by MLE (scipy, no new dependencies) on
daily returns of the equal-weight crypto coverage-core index — the same panel
construction the harness scores (score_risk_model._wide_returns, 98% coverage,
2016+). Outputs alpha, beta, persistence p = alpha+beta, and the registered
21d-aggregated weight w = (1/21) * sum_{h=1..21} p^h. w is DERIVED, not tuned:
whatever this prints is the candidate (wiki registration 06d7bc7).
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from production.core.lake import Lake
from scripts.score_risk_model import _instrument_map, _load_prices, _wide_returns


def _garch_nll(params: np.ndarray, r: np.ndarray, v_long: float) -> float:
    a, b = params
    if a < 0 or b < 0 or a + b >= 0.9995:
        return 1e12
    omega = v_long * (1.0 - a - b)
    h = np.empty_like(r)
    h[0] = v_long
    for t in range(1, len(r)):
        h[t] = omega + a * r[t - 1] ** 2 + b * h[t - 1]
    h = np.maximum(h, 1e-12)
    return float(np.sum(np.log(h) + r ** 2 / h))


def main() -> int:
    lake = Lake()
    prices = _load_prices(lake, "2016-01-01", None)
    instruments = _instrument_map(lake, prices)
    ids = sorted(instruments[instruments == "crypto"].index)
    panel = _wide_returns(prices, ids)
    idx = panel.mean(axis=1)
    r = (idx - idx.mean()).to_numpy(dtype=float)
    v_long = float(np.var(r, ddof=0))
    print(f"crypto core index: {len(r)} days, {panel.shape[1]} ids, "
          f"ann vol {np.sqrt(v_long * 365):.3f}")

    best = None
    for a0, b0 in [(0.05, 0.90), (0.10, 0.85), (0.02, 0.95)]:
        res = minimize(_garch_nll, x0=np.array([a0, b0]), args=(r, v_long),
                       method="Nelder-Mead",
                       options={"xatol": 1e-6, "fatol": 1e-8, "maxiter": 5000})
        if best is None or res.fun < best.fun:
            best = res
    a, b = float(best.x[0]), float(best.x[1])
    p = a + b
    w = float(np.mean([p ** h for h in range(1, 22)]))
    print(f"alpha={a:.4f}  beta={b:.4f}  persistence={p:.4f}")
    print(f"w (21d-aggregated weight on current vol) = {w:.4f}")

    out = REPO_ROOT / "diagnostics" / (
        f"garch_persistence_crypto_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.json")
    out.write_text(json.dumps(
        {"alpha": a, "beta": b, "persistence": p, "w_21d": w,
         "n_days": int(len(r)), "n_ids": int(panel.shape[1]),
         "nll": float(best.fun), "registration": "06d7bc7"}, indent=2))
    print(f"artifact -> {out.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
