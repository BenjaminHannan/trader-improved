---
type: source
source_type: papers (construction reference)
author: Asness, Moskowitz & Pedersen (JF 2013); Moskowitz, Ooi & Pedersen (JFE 2012); Kang, Rouwenhorst & Tang (JF 2020); Fernandez-Perez et al. (hedging pressure); Asness & Frazzini (2013); Novy-Marx
date_published: 2012-2020
url: https://pages.stern.nyu.edu/~lpederse/papers/ValMomEverywhere.pdf
confidence: high (canonical constructions are standard and stable across secondary sources)
key_claims:
  - "AMP 2013 momentum: rank WITHIN each of 8 asset classes separately (top/bottom tercile per class, equal-weighted, monthly), then combine per-class portfolios — never pooled ranks across classes"
  - "MOP 2012 TSMOM: per-instrument time-series bet sign(12m excess return), position scaled to constant vol (40%/ex-ante vol) — a time-series strategy, not a cross-sectional rank"
  - "KRT 2020: COT positioning carries TWO premia with OPPOSITE signs — long-horizon commercial hedging-pressure LEVELS (risk premium) vs short-horizon NONcommercial position CHANGES (liquidity provision)"
  - "Practitioner value canon: industry/sector-relative value (Asness-Frazzini devil-in-details; Novy-Marx) has materially better IC than raw cross-sectional E/P, which is dominated by sector bets"
---

# Canonical constructions for the rejected factor families

Reference page: what the anchor papers actually do, vs what our signals do.

## Momentum (mom_12_1)
- Canonical (Jegadeesh-Titman; AMP 2013 "Value and Momentum Everywhere"):
  12-1 return, ranked **within** each asset class; per-class L/S portfolios are
  built first and only then combined at the portfolio level. AMP explicitly keep
  per-class construction "simple and uniform" to avoid pooling artifacts.
- Ours: same 12-1 window (`close.shift(21)/close.shift(252)-1`,
  `production/signals/momentum.py`), registered on 4 sleeves at once. Per-date IC
  IS computed within-sleeve (`production/alpha/ic.py:rank_ic` groups by sleeve) —
  good — but the gate t-stat pools the per-date-per-sleeve IC observations across
  sleeves unweighted. Sleeve IC variances differ ~60x (Var ≈ 1/(N-1): N=8 fx vs
  N≈500 equity), so an unweighted pooled t is dominated by the noisiest sleeves.
  Canonical aggregation would precision-weight (per-date IC weighted by N-1, or
  Stouffer-combine per-sleeve t with weights ∝ √obs).

## TSMOM (tsmom)
- Canonical (Moskowitz-Ooi-Pedersen 2012): a TIME-SERIES strategy — each
  instrument bets sign(own 12m excess return), sized to constant ex-ante vol;
  performance is the average of per-instrument bets. Not a cross-sectional rank.
- Ours: `sign(12m return)` fed into a per-date cross-sectional rank IC across the
  fx/commodity ETF sleeves (N≈8-16). Two divergences: (a) sign() makes the score
  two-valued, so per-date Spearman against returns is extremely coarse at small N;
  (b) rank-IC evaluates *relative* trend across instruments — a different claim
  from MOP's time-series one. A cross-sectional gate can reject a factor whose
  canonical (time-series) form is fine, and 2024-2026 was hostile to both forms
  ([[sources/trend-cot-value-2024-2026-regime]]).

## COT positioning (cot_positioning)
- Canonical hedging pressure (Basu-Miffre; Fernandez-Perez et al. "Hedging
  pressure everywhere"): commercial net-SHORT level = hedging demand; long
  high-hedging-pressure instruments earns the insurance premium. KRT (JF 2020)
  sharpen this: the LEVEL of commercial positions (≈ mirror of noncommercial)
  prices a long-horizon premium, while short-horizon CHANGES in noncommercial
  positions predict returns with the OPPOSITE sign (liquidity provision).
- Ours: `-(156w rolling z of noncomm_net/OI)` (`production/signals/positioning.py`).
  The 3y z-score of a LEVEL is a hybrid: over a 3y window, z variation is driven
  by medium-term position *changes*, which per KRT carry the opposite-signed
  premium to the level effect the contrarian sign assumes. The construction
  plausibly mixes the two premia with conflicting signs — consistent with a large
  train t that doesn't survive OOS.

## Equity value (earnings_yield)
- Canonical academic HML is not sector-neutral, but the practitioner canon
  (Asness-Frazzini "The Devil in HML's Details"; Novy-Marx industry-relative
  value; Asness-Porter-Stevens) documents that within-industry value has better
  IC and less regime beta, because raw E/P ranks are dominated by static sector
  tilts (long financials/energy, short tech).
- Ours: TTM-EPS/price ranked across the whole equity sleeve with no sector
  demean (`production/signals/value.py:EarningsYield`) — so in 2016-2026 it is
  substantially a short-tech sector bet, which explains both the negative train t
  (value winter) and the OOS flip (partial 2026 broadening).

Feeds [[questions/research-rejected-factor-forensics]].
