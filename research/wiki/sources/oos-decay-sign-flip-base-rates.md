---
type: source
source_type: paper (two papers, one page — same question)
author: R. David McLean & Jeffrey Pontiff (JF 2016); Andrew Y. Chen & Tom Zimmermann (Critical Finance Review 2022)
date_published: 2016 / 2022
url: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2156623 ; https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3604626
confidence: high (headline numbers replicated across many secondary sources; openassetpricing.com data is public)
key_claims:
  - "McLean-Pontiff: published anomaly portfolio returns are 26% lower out-of-sample, 58% lower post-publication"
  - "Real (published, replicable) predictors DECAY out-of-sample but overwhelmingly RETAIN SIGN — full reversal is rare"
  - "Chen-Zimmermann: 98% of clearly-significant predictors replicate in-sample (t>1.96); decay is stronger for predictors that were stronger in-sample"
---

# Base rates: what happens to real vs overfit factors out-of-sample

The empirical anchor for interpreting a train/OOS IC **sign flip**:

- **McLean & Pontiff 2016** (97 published predictors): mean return decays 26%
  between the original sample and the pre-publication OOS window, 58% after
  publication — but the typical predictor **remains positive**. Data-mining alone
  would imply ~100% decay (to zero); they attribute the 26% to mild data-mining
  bias and the extra 32pp to post-publication arbitrage.
- **Chen & Zimmermann 2022** (319 predictors, open-source replication): decay
  post-publication, no general sign reversal; decay correlates with in-sample
  strength (regression-to-the-mean of selected estimates — a Harvey-Liu style
  selection effect).

**Implication for the gate**: for a *real* factor, the base rate of an OOS sign
flip is low — driven by sampling error of the OOS window, not by the factor.
For a *spurious* factor the flip probability is ~50% by construction (sign is
noise). So the same-sign criterion is a reasonable likelihood-ratio test, BUT its
power depends entirely on the OOS window's standard error, which for small
cross-sections with overlapping labels is large (see the SE arithmetic in
[[questions/research-oos-gate-design]]). A flip in a high-SE cell (fx N=8,
futures COT N~30) is weak evidence of overfit; a flip in a 500-name equity
cross-section is strong evidence.
