---
type: source
source_type: web/docs
title: "Kalshi maker fee schedule + order-book data availability (2026-07)"
date_published: 2026-03..2026-07
url: https://kalshi.com/docs/kalshi-fee-schedule.pdf
confidence: high on formulas (two independent sources agree); medium on tier details (PDF itself 429-blocked this round)
key_claims:
  - "Taker fee per contract = $0.07 x C x (1-C); maker fee = 25% of taker (max 0.44c vs taker max 1.75c at C=0.50)"
  - "July 2026 schedule adds volume tiers: taker 12.0bp (Tier 0) down to 2.6bp; maker 5.0bp down to 1.2bp"
  - "Maker fees charge only on execution of resting limit orders; cancels are free"
  - "Historical order books are reconstructable from free WebSocket streams + REST snapshots (Marriott, SSRN 6583921); a vendor (lycheedata) sells 36GB+ of historical Kalshi orderbook/trades"
---

# Kalshi maker fees + LOB data availability

## Fees (verified against two independent secondaries; PDF pending)
- Taker: `$0.07 × C × (1−C)` per contract — EXACTLY our production
  `taker_fee_fraction` (events sleeve, commit 17616cf). Stable since launch.
- **Maker: 25% of taker.** Max 0.44c/contract at mid prices. Charged only when
  a resting order fills; cancels free.
- 2026-07-07 fee-schedule update introduces volume tiers (taker 12.0→2.6bp,
  maker 5.0→1.2bp by tier); our sleeve is Tier 0 — the base formulas govern.
  Fetch the PDF (429-blocked this round) before building anything tier-aware.

## Order-book data paths
- Public REST orderbook endpoint (depth param 0-100) — LIVE state only; no
  official history.
- WebSocket `orderbook_delta` channel (auth required, per-market subscribe):
  server sends `orderbook_snapshot` then incremental deltas. KNOWN PITFALL:
  books desync permanently after a sequence gap unless you resnapshot
  (kalshi-python-sdk issue #189) — any capture daemon must detect gaps and
  re-anchor with REST snapshots.
- Marriott (SSRN 6583921, 2026): full-depth tick-level LOB reconstruction for
  ALL Kalshi markets from the free streams, anchored by periodic REST
  snapshots. (Paper landing 403-blocked this round; method summary from
  abstract/secondary coverage.)
- Vendor: lycheedata.com sells 36GB+ historical Kalshi trades + orderbook;
  kalshibacktest.com has historical books for crypto series. Cost unknown —
  owner decision.

Sources: [fee schedule PDF](https://kalshi.com/docs/kalshi-fee-schedule.pdf),
[pm.wiki fee explainer](https://pm.wiki/learn/kalshi-fees-explained),
[Kalshi help: fees](https://help.kalshi.com/en/articles/13823805-fees),
[orderbook responses docs](https://docs.kalshi.com/getting_started/orderbook_responses),
[WS quickstart](https://docs.kalshi.com/getting_started/quick_start_websockets),
[Marriott SSRN 6583921](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6583921),
[lycheedata](https://lycheedata.com/kalshi-historical-data),
[SDK desync issue](https://github.com/TexasCoding/kalshi-python-sdk/issues/189)
