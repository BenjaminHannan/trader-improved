"""Sleeve-level capital allocation — equal risk contribution with risk caps + warm-up.

`erc_weights` solves the equal-risk-contribution portfolio by a sqrt-damped fixed point;
`apply_risk_caps` clips individual sleeves' risk-contribution shares; `sleeve_allocation`
wires them together over a trailing EWMA covariance, with an inverse-vol warm-up rule for
sleeves that lack enough history to trust their covariance. Everything is trailing-only:
`sleeve_allocation` uses returns strictly before the decision date.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

# Daily-return variance floor below which a sleeve is treated as "degenerate" (dead) rather
# than genuinely low-risk, for purposes of the ERC solve in `erc_weights`.
#
# Why 1e-8 (= (1bp daily vol)^2): the smallest per-sleeve cost floor in configs/costs.yaml is
# 5bp (equities/fx_etf/rates_etf/intl_etf/sector_etf; crypto is 30bp). A sleeve whose optimizer
# is actually clearing that floor and trading carries daily return vol at LEAST comparable to
# it -- in practice real sleeves run tens to hundreds of bp/day of vol, i.e. variance >= 1e-6,
# two-plus orders of magnitude above this floor. A sleeve whose alpha never clears its cost
# floor is correctly left untraded by the optimizer, and its return series is then pure
# solver residual (~1e-9-scale weights on ~1e-2-scale price moves -> variance around 1e-20 to
# 1e-11) -- two-plus orders of magnitude BELOW this floor. 1e-8 sits in the wide gap between
# the two regimes, so it cleanly separates "dead" sleeves from any sleeve actually trading.
_MIN_SLEEVE_VARIANCE = 1e-8


def _ewma_cov(returns: pd.DataFrame, halflife: float) -> pd.DataFrame:
    """Exponentially-weighted covariance over complete-case rows (newest weighted most)."""
    df = returns.dropna(how="any")
    T = len(df)
    if T == 0:
        return pd.DataFrame(np.zeros((returns.shape[1], returns.shape[1])),
                            index=returns.columns, columns=returns.columns)
    lam = 0.5 ** (1.0 / halflife)
    w = lam ** np.arange(T - 1, -1, -1)  # oldest -> smallest weight, newest -> 1
    w = w / w.sum()
    X = df.to_numpy(dtype=float)
    mu = np.average(X, axis=0, weights=w)
    Xc = X - mu
    cov = (Xc * w[:, None]).T @ Xc
    cov = 0.5 * (cov + cov.T)
    return pd.DataFrame(cov, index=df.columns, columns=df.columns)


def _risk_contrib_shares(w: np.ndarray, Sigma: np.ndarray) -> np.ndarray:
    m = Sigma @ w
    rc = w * m
    tot = rc.sum()
    if tot <= 0:
        return np.full_like(w, 1.0 / len(w))
    return rc / tot


def erc_weights(cov: pd.DataFrame, tol: float = 1e-8, max_iter: int = 200) -> pd.Series:
    """Equal-risk-contribution weights via the sqrt-damped fixed point.

    Iteration x_i <- sqrt(x_i / (Sigma x)_i), renormalized. Its fixed point satisfies
    x_i (Sigma x)_i = const (equal risk contributions); the sqrt damping avoids the
    oscillation of the raw x_i ∝ 1/(Sigma x)_i map. For a diagonal covariance this
    collapses to inverse-vol weights. Converges when the dispersion of normalized risk
    contributions (max - min) falls below `tol`.

    Sleeves whose covariance diagonal is below `_MIN_SLEEVE_VARIANCE` are "degenerate" --
    typically a sleeve whose optimizer correctly refused to trade (alpha below its cost
    floor), leaving a return series that is pure solver residual rather than real risk. The
    raw fixed point above treats near-zero variance as near-zero risk and explosively levers
    such a sleeve toward weight 1 (the `1e-16` floor on `Sigma @ x` exists only to avoid a
    divide-by-zero, not to express a risk view). Degenerate sleeves are therefore excluded
    from the solve entirely and assigned weight 0 -- a dead sleeve has no positions to fund
    anyway -- with the live sleeves' ERC weights renormalized over just themselves. If every
    sleeve is degenerate there is nothing to solve; fall back to equal weight across all of
    them and warn loudly (this should never happen with real, differentiated sleeves).
    """
    idx = cov.index
    Sigma_full = cov.to_numpy(dtype=float)
    n = Sigma_full.shape[0]
    diag = np.diag(Sigma_full)
    live = diag >= _MIN_SLEEVE_VARIANCE

    if not live.any():
        warnings.warn(
            f"erc_weights: all {n} sleeve(s) {list(idx)} have variance below the degenerate "
            f"floor ({_MIN_SLEEVE_VARIANCE:g}) -- diag={np.array2string(diag, precision=3)}. "
            "Falling back to equal weight across all sleeves.",
            stacklevel=2,
        )
        return pd.Series(np.full(n, 1.0 / n), index=idx)

    if not live.all():
        dead = [s for s, ok in zip(idx, live) if not ok]
        warnings.warn(
            f"erc_weights: excluding degenerate sleeve(s) {dead} (variance below "
            f"{_MIN_SLEEVE_VARIANCE:g} -- solver residual, not real risk) from the ERC solve; "
            "assigning them weight 0 and renormalizing over the remaining live sleeves.",
            stacklevel=2,
        )

    live_idx = [s for s, ok in zip(idx, live) if ok]
    Sigma = cov.loc[live_idx, live_idx].to_numpy(dtype=float)
    n_live = Sigma.shape[0]
    x = np.full(n_live, 1.0 / n_live)
    for _ in range(max_iter):
        m = np.maximum(Sigma @ x, 1e-16)
        x = np.sqrt(x / m)
        x = x / x.sum()
        shares = _risk_contrib_shares(x, Sigma)
        if shares.max() - shares.min() < tol:
            break

    out = pd.Series(0.0, index=idx)
    out.loc[live_idx] = x
    return out


def apply_risk_caps(w, cov, crypto_cap: float | None = None,
                    max_sleeve: float | None = None, *,
                    caps: dict[str, float] | None = None,
                    max_iter: int = 500, tol: float = 1e-9) -> pd.Series:
    """Cap each sleeve's *risk-contribution* share, then renormalize weights to sum 1.

    Per-sleeve caps are supplied as a ``caps`` dict mapping sleeve -> maximum risk-contribution
    share, with a reserved key ``"max_any_sleeve"`` giving the *generic* bound applied to every
    sleeve that has no explicit key. A sleeve absent from the dict and with no ``max_any_sleeve``
    present is left uncapped (``+inf``). This is exactly the ``cfg["allocation"]["risk_caps"]``
    dict the config already carries.

    Back-compat: the legacy ``crypto_cap`` / ``max_sleeve`` args are honored when ``caps`` is not
    given — they map to ``{"crypto": crypto_cap, "max_any_sleeve": max_sleeve}`` with the historical
    defaults (0.20 / 0.40), reproducing the original two-tier behavior bit-for-bit.

    Convergence: each pass finds the single worst violator j and shrinks w_j by the
    damped factor sqrt(cap_j / share_j) (< 1), which strictly lowers share_j; renormalizing
    lifts the others. Because the total over-cap risk share is monotonically drained toward
    the feasible simplex, the iteration converges; `max_iter` is a safety bound.
    """
    w = pd.Series(w, dtype=float).copy()
    idx = w.index
    Sigma = pd.DataFrame(cov).reindex(index=idx, columns=idx).to_numpy(dtype=float)

    if caps is None:
        cc = 0.20 if crypto_cap is None else float(crypto_cap)
        ms = 0.40 if max_sleeve is None else float(max_sleeve)
        caps = {"crypto": cc, "max_any_sleeve": ms}
    generic = float(caps.get("max_any_sleeve", np.inf))
    cap_vec = np.array([float(caps.get(s, generic)) for s in idx], dtype=float)

    wv = np.clip(w.to_numpy(dtype=float), 0.0, None)
    if wv.sum() <= 0:
        return w
    wv = wv / wv.sum()

    for _ in range(max_iter):
        shares = _risk_contrib_shares(wv, Sigma)
        viol = shares - cap_vec
        if np.all(viol <= tol):
            break
        j = int(np.argmax(viol))
        wv[j] *= np.sqrt(cap_vec[j] / max(shares[j], 1e-12))
        wv = wv / wv.sum()

    return pd.Series(wv, index=idx)


def sleeve_allocation(sleeve_returns: pd.DataFrame, t, cfg: dict) -> pd.Series:
    """Allocate across sleeves at date t from trailing sleeve returns (strictly before t).

    Warm sleeves (>= warmup_days observations) are allocated by ERC on an EWMA covariance
    (halflife from cfg). Cold sleeves (fewer obs) fall back to an inverse-vol weight with a
    0.5 haircut — half of their inverse-vol fair share — because their covariance is not yet
    trustworthy. Warm ERC weights and haircut cold weights are pooled and normalized to 1.
    """
    alloc = cfg["allocation"]
    halflife = float(alloc["sleeve_cov_halflife_days"])
    warmup = int(alloc["warmup_days"])
    risk_caps = alloc.get("risk_caps") or None

    r = sleeve_returns[sleeve_returns.index < t]
    cols = list(sleeve_returns.columns)
    counts = r.notna().sum()

    cov = _ewma_cov(r, halflife)
    vol = pd.Series(np.sqrt(np.clip(np.diag(cov.to_numpy(dtype=float)), 0.0, None)),
                    index=cov.index)

    warm = [s for s in cols if counts.get(s, 0) >= warmup]
    cold = [s for s in cols if 0 < counts.get(s, 0) < warmup]

    raw = pd.Series(0.0, index=cols)

    # Cold sleeves: inverse-vol fair share (over all sleeves), halved.
    inv = pd.Series(0.0, index=cols)
    for s in cols:
        v = vol.get(s, np.nan)
        if np.isfinite(v) and v > 0:
            inv[s] = 1.0 / v
    inv_frac = inv / inv.sum() if inv.sum() > 0 else inv
    for s in cold:
        raw[s] = 0.5 * inv_frac[s]

    # Warm sleeves: ERC (single warm sleeve trivially gets full weight within its block),
    # then per-sleeve risk-contribution caps bind within the warm block (crypto/events/... from
    # cfg["allocation"]["risk_caps"]). Caps are scoped to the warm ERC block because a single
    # warm sleeve or a cold-only mix has no meaningful multi-sleeve risk decomposition to clip.
    if len(warm) >= 2:
        warm_cov = cov.reindex(index=warm, columns=warm)
        w_erc = erc_weights(warm_cov)
        if risk_caps:
            w_erc = apply_risk_caps(w_erc, warm_cov, caps=risk_caps)
        for s in warm:
            raw[s] = w_erc[s]
    elif len(warm) == 1:
        raw[warm[0]] = 1.0

    total = raw.sum()
    if total > 0:
        raw = raw / total
    return raw
