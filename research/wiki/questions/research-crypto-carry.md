---
type: synthesis
title: "Research: crypto funding carry evidence (iteration 10)"
created: 2026-07-06
status: researched — design confirmed; basis factor added to backlog
---

# Research: does funding carry hold up, and what's better?

## Key Findings
- BIS WP 1087 ("Crypto carry") + CMU "The Crypto Carry Trade": funding-rate carry is
  a real, large premium (market-neutral variant Sharpe 6.45 in 2020-2025 samples)
  but is **decaying hard** — ~4 from 2024, negative stretches in 2025. Crowding.
- Cross-sectional crypto futures studies: **basis** (perp/futures vs spot) is the
  strongest cross-sectional return predictor, ahead of funding momentum.
- Funding rates themselves are predictable next-period (AR structure) — supports
  using a trailing mean (our 7d) rather than last print.

## Verdict for trader-improved
- `carry_funding` (−7d mean funding, cross-sectional) is supported; keep. Its decay
  risk is exactly what the gate re-runs + IC decay monitor exist for — no special
  casing.
- **Backlog add**: `basis` factor (perp mark vs spot) — needs a ccxt perp-mark-price
  loader (same BaseLoader shape); strongest documented cross-sectional signal in the
  asset class. Stage-2 data work, then a registered signal through the harness.

## Sources
- BIS Working Paper 1087, "Crypto carry" (bis.org; fetch blocked)
- Christin et al. (CMU), "The Crypto Carry Trade" (andrew.cmu.edu; fetch blocked)
- "Two-Tiered Structure of Cryptocurrency Funding Rate Markets" (MDPI 2026)
- SSRN 5576424, "Predictability of Funding Rates"
