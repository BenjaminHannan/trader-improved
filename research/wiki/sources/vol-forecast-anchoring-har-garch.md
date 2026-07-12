---
type: source
source_type: literature scan
title: "Horizon-h vol forecasting: EWMA's missing anchor, GARCH term structure, HAR cascade"
date_published: 2004-2026 (Corsi HAR; MF2-GARCH 2025; crypto HAR studies 2024-26)
url: https://vlab.stern.nyu.edu/docs/volatility/GARCH
confidence: high — textbook results, multiply sourced
key_claims:
  - "EWMA's multi-day forecast stays FLAT at the current level — no unconditional variance, no long-run anchor (the exact mechanism our halflife-30 NO-ADOPT exposed at h=21)"
  - "GARCH(1,1): h-step forecast converges to V_L = omega/(1-alpha-beta) at rate (alpha+beta)^h — a principled horizon-dependent blend weight"
  - "HAR (Corsi): additive daily/weekly/monthly RV cascade; beats GARCH-type for Bitcoin RV in multiple comparisons; ML beats HAR across cryptos EXCEPT Bitcoin"
---

# Vol forecasting with a long-run anchor

Why our halflife candidates failed by construction: any single EWMA projects
its current level flat across the horizon. At h=21 vol mean-reverts, so the
fast EWMA (hl-30) amplified transients and the slow one (hl-90) merely lagged
less — neither anchors to the long-run mean.

- GARCH term structure: sigma²(t+h) = V_L + (alpha+beta)^h (sigma²(t) − V_L).
  Aggregated over a 21d horizon this is a DECLINING weight on current vol —
  a two-component blend with the weight set by measured persistence, not by
  hand.
- HAR: RV_forecast = c + b_d RV_1d + b_w RV_5d + b_m RV_21d (trailing-fit,
  shift(1) compatible). Empirical front-runner for BTC at our horizon.
- Matrix analogue for our harness (structural factor cov + small-sleeve
  fallback): Sigma_fc = w·Sigma_EWMA(hl=90) + (1−w)·Sigma_long (trailing ~3y
  equal-weight), w fixed A PRIORI from GARCH persistence fitted on the crypto
  sleeve index in the registration itself — no post-hoc w tuning.

Sources: [V-Lab GARCH docs](https://vlab.stern.nyu.edu/docs/volatility/GARCH),
[MF2-GARCH (Conrad, JAE 2025)](https://onlinelibrary.wiley.com/doi/full/10.1002/jae.3118),
[GARCH vs HAR crypto comparison (MDPI 2026)](https://www.mdpi.com/2227-7072/14/4/90),
[crypto HAR/ML horserace (Springer 2024)](https://link.springer.com/article/10.1007/s10690-024-09510-6),
[Zivot practical GARCH](https://faculty.washington.edu/ezivot/research/practicalgarchfinal.pdf),
[HAR overview](https://portfoliooptimizer.io/blog/volatility-forecasting-har-model/)
