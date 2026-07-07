---
type: source
source_type: practitioner (vendor docs + blog forensics)
author: FirstRate Data; Portfolio Optimizer blog; Alvarez Quant Trading; EODHD/Koyfin docs
date_published: 2020-2025 (evergreen methodology)
url: https://firstratedata.com/about/price_adjustment ; https://portfoliooptimizer.io/blog/adjusted-prices-without-look-ahead-bias/
confidence: high (mechanisms independently documented by multiple vendors)
key_claims:
  - Vendors differ in the ANCHOR price for the adjustment factor (official close vs last trade before ex-date)
  - Vendors differ in ROUNDING (2dp vs 4dp factors) — divergence compounds going back in time
  - Backward-adjusted series depend on RETRIEVAL DATE — every new distribution rewrites the whole history (a look-ahead trap for cached series)
  - Proportional (multiplicative, CRSP-style) vs absolute-subtraction adjustment produce different series, worst for high-yield/monthly-distribution ETFs
---

# Why cross-vendor adjusted closes disagree for bond/FX ETFs — and the practitioner normalization

The four documented mechanisms behind cross-vendor adjusted-close divergence:

1. **Anchor price**: adjustment factor = 1 − div/P_anchor; some vendors use the
   official close before ex-date, others the last trade — differs on illiquid ETFs.
2. **Rounding**: 2-decimal vs 4-decimal adjustment factors. Tiny per-event, but a
   monthly-distribution bond ETF has ~12 events/yr × 10+ yrs — divergence grows
   monotonically with lookback. This is exactly the month-end signature our
   cross-check quarantined (66 instruments, FX/bond-ETF class).
3. **Backward adjustment is retrieval-date-dependent**: the entire history shifts
   every ex-date. Two vendors' series pulled on different days disagree everywhere
   even with identical methodology; a cached series silently embeds look-ahead
   (Portfolio Optimizer documents the look-ahead-free construction).
4. **Proportional vs absolute** subtraction of the distribution.

**Practitioner normalization** (the fix, at the measurement layer):
- NEVER cross-check adjusted closes across vendors. Cross-check (a) UNADJUSTED
  closes (should agree to the tick) and (b) the distribution-event table
  (ex-date, amount) separately.
- Build total-return series in-house from raw close + events with ONE documented
  convention (multiplicative factor, recompute full series per new event, stamped
  `available_from` = ex-date), rather than ingesting any vendor's adj-close.

Feeds [[questions/research-data-remediation]] and backlog #18 (quarantine
follow-ups: the FX-ETF distribution-adjustment class is explained by mechanisms
1-4; the instruments are healthy, the comparison series was the bug).
