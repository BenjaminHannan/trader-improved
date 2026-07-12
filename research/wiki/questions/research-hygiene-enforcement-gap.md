---
type: synthesis
title: "Research: hygiene enforcement gap (iteration 15, internal audit)"
created: 2026-07-06
status: implemented
---

# Internal audit: the hygiene layer is dead code

## Finding
CLAUDE.md hard rule: "Delisted-symbol-reuse blocklist + sub-$0.10 price backstop in
reference/hygiene.py." The module exists and is unit-tested — and is called from
NOWHERE in production: not the loaders, not universe construction, not the engine.
The ticker-reuse class of bug the rule exists to kill is currently unmitigated in
the actual data flow. (Same failure shape as iteration 9's sector gap: capability
built, wiring forgotten. Pattern noted for the loop: audit WIRING, not just
existence, for every hard rule.)

## Decision → implementation
Enforce at the earliest possible point — the price loaders' transform (the
transcript lesson: pinpoint issues as far upstream as possible):
- yfinance_prices and stooq_prices transforms apply `apply_price_backstop`
  (drop close < $0.10 rows) and drop rows whose vendor symbol is blocklisted at
  that obs_date (`is_blocked(symbol, obs_date)`), counting drops into the audit
  record (never silent).
- Shared helper `apply_hygiene(df, symbol_col)` added to hygiene.py so both loaders
  (and future price vendors) share one implementation.

## Verification (in-repo)
Loader-level: canned payload containing a $0.05 row and an FB row inside the
blocklist window -> both dropped, drop counts land in the audit record, clean rows
unaffected; a blocklisted symbol OUTSIDE its window survives.

## Sources
Internal audit; CLAUDE.md; reference-repo lesson (ticker_hygiene.py pattern).
