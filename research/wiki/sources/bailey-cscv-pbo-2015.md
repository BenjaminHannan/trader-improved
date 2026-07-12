---
type: source
source_type: paper
author: David H. Bailey, Jonathan Borwein, Marcos Lopez de Prado, Qiji Jim Zhu
date_published: 2015 (J. Computational Finance 2017)
url: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253
confidence: high (canonical; abstract + secondary summaries verified, full PDF fetch blocked/binary)
key_claims:
  - Hold-out (single train/test split) is "unreliable and inaccurate" for investment backtests
  - CSCV — S blocks, all C(S, S/2) train/test recombinations — yields a DISTRIBUTION of OOS outcomes
  - PBO = fraction of combinations where the in-sample-best config ranks below median out-of-sample
---

# The Probability of Backtest Overfitting (Bailey, Borwein, Lopez de Prado, Zhu)

The paper that formalizes why a single chronological hold-out is a weak instrument:
one split gives one draw from a high-variance OOS estimator, so a selection decision
made on it inherits that variance. CSCV (combinatorially symmetric cross-validation)
partitions the sample into S blocks and evaluates all C(S, S/2) recombinations,
producing a distribution of OOS ranks. PBO is the probability that the configuration
picked in-sample falls below the OOS median — i.e., the probability the selection
step itself was the source of the observed performance.

Relation to CPCV (AFML ch. 12): CPCV adds purging (drop train observations whose
labels overlap the test period) and embargo (drop a buffer after each test fold,
h ≈ 1% of bars usually suffices) to make the recombinations leakage-free for
overlapping-label panels.

Contribution to this repo: motivates a within-train CPCV sign-stability diagnostic
for the factor gate — the single 80/20 split stays as the binding criterion, CSCV
supplies the error bar around it. See [[questions/research-oos-gate-design]].
