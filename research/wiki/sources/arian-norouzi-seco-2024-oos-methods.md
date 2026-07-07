---
type: source
source_type: paper
author: Hamid R. Arian, Daniel Norouzi M., Luis A. Seco
date_published: 2024 (Knowledge-Based Systems; SSRN 4686376)
url: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4686376
confidence: medium-high (abstract + multiple secondary summaries; full text paywalled, direct fetch 403)
key_claims:
  - Controlled synthetic environment (Heston, Merton jump-diffusion, drift-burst, regime-switching) where ground truth is known
  - CPCV dominates K-fold, purged K-fold, AND walk-forward on Probability of Backtest Overfitting and DSR stability
  - Walk-forward specifically shows "increased temporal variability and weaker stationarity" — poor false-discovery prevention because it evaluates one historical path
---

# Backtest Overfitting in the ML Era: Comparison of OOS Testing Methods (Arian, Norouzi, Seco 2024)

The closest thing to a controlled experiment on our exact question: when the data
generating process is known (synthetic), which OOS validation scheme best separates
real from spurious strategies? Findings:

- **Walk-forward / single chronological split**: performance estimate is contingent
  on one specific historical path → high variance, weakest false-discovery control
  of the tested schemes. This is the scheme our factor gate currently uses (single
  80/20 chronological split).
- **CPCV**: multiple purged+embargoed train/test recombinations → a distribution of
  OOS estimates, lower PBO, more stable deflated-Sharpe test statistics.
- Novel variants (bagged CPCV, adaptive CPCV) improve robustness further but are
  incremental over plain CPCV.

Caveat for our use: their strategies are ML trading rules, not per-date rank-IC
factor gates; the variance argument transfers, the exact PBO magnitudes do not.

Feeds [[questions/research-oos-gate-design]].
