---
type: synthesis
title: "Research: cost model calibration (iteration 4)"
created: 2026-07-06
status: implemented (reporting hardening; prefactor unchanged per repo constraint)
---

# Research: is our Almgren-Chriss sqrt model honestly calibrated?

## Key Findings
- **Exponent**: sqrt (0.5) is well inside the empirical band — Almgren et al. 2005:
  0.60 (US eq); Moro 2009: 0.64-0.72 (UK); Toth 2011: 0.50-0.60 (futures); Bacry
  2015: 0.40-0.45 (EU eq); Zarinelli 2015: 0.47. Keep sqrt.
- **Prefactor**: literature centers ~0.5-1.0 x daily sigma at full-ADV participation
  (Grinold rule of thumb: trading a day's volume costs about a day's vol). Our
  CLAUDE.md pins `alpha = 0.15` — toward the optimistic end for the impact term;
  the per-sleeve floors are what keep small trades honest.
- **Crypto**: 2025 spot-data study finds the sqrt law does NOT transfer (size
  exponent ~0.1) — impact modeling there is unreliable either way; our 30bp floor
  dominates at swing-trade sizes.

## Verdict
`alpha = 0.15` is a hard rule in CLAUDE.md ("use the constraints already given") —
NOT changed. Adopted instead: **counterfactual cost-sensitivity reporting**. The
CostModel now keeps a per-charge ledger (floor/spread/impact decomposition); every
backtest report shows net Sharpe and annual cost drag under impact-prefactor
multipliers x1 (traded), x3.3 (alpha=0.5), x6.7 (alpha=1.0). If the strategy only
survives at x1, the report says so — optimism made visible, not silently traded.

## Verification (in-repo)
Ledger sums reconcile with charged costs bit-exactly; counterfactual with
multiplier 1.0 equals realized; floor-bound trades are invariant to the multiplier
(floor binds); sensitivity monotone in the multiplier.

## Open Questions
- Real calibration of the crypto impact curve (exponent ~0.1 claim) once live fills
  exist — implementation-shortfall data from paper trading is the natural source.
- Per-instrument half-spread overrides from observed quotes (Stage-2 of costs).

## Sources
- Almgren, Thum, Hauptmann, Li (2005), "Direct Estimation of Equity Market Impact"
- Toth et al. (2011); Bacry et al. (2015); Zarinelli et al. (2015) — via search
- Baruch MFE lecture "Square-root law" (Gatheral); Lillo Imperial lectures
- github.com/SLMolenaar/crypto-market-impact (2025 crypto exponent study)
