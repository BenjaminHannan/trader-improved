# Practitioner mechanism scan (deep-research sweep, 2026-07-07)

Deep-research agent sweep of practitioner / non-academic sources (YouTube quants,
forums, podcasts, forensic blog pieces) for niche, mechanism-backed trading ideas the
academic literature has not operationalized. Full brief + per-idea 10-item schema
(mechanism, losing counterparty, academic anchor, testability, red flags, and a
PRE-REGISTERED falsifiable prediction) preserved below in condensed form. Trial-ledger
discipline: none of these touch the gate until promoted explicitly; at most 1-2 per
quarter.

## Ranked candidates (agent's ranking, with local annotations)

| # | Mechanism | Anchor | Testable with current lake? | Local annotation |
|---|---|---|---|---|
| 1 | Kalshi economic-data ladders: fade 5-20c longshot buckets | Bürgi-Deng-Whelan 2025 (MPRA 126350): <10c contracts lose >60%; taker-side bias | **NEEDS BACKFILL** — we snapshot active markets only; resolved-contract history (2023-2025 bucket prices + settlements) requires a one-time Kalshi API backfill (~1 day) | Best capacity fit for this book (thin books deter institutional arb). Fee curve is parabolic — net-of-fee test mandatory |
| 2 | Macro ladder pre-release drift toward Cleveland Fed nowcast | Diercks-Katz-Wright FEDS 2026-010 (accuracy only, no drift regression); Jia et al. arXiv 2604.20421 | **NEEDS BACKFILL** (same as #1) + Cleveland Fed nowcast VINTAGE history | Sign risk: Jia et al. hint the market may LEAD the nowcast — the pre-registered coefficient test is decisive either way. Our ALFRED vintages kill the look-ahead trap |
| 3 | Under-reaction drift after large 1-day probability moves | Angelini-De Angelis arXiv 2606.07811 (β=0.63, sports only) | needs backfill; daily-bar analogue only | Risks collapsing into generic momentum; must condition on liquidity tercile |
| 4 | Kalshi-Polymarket cross-venue convergence | arXiv 2601.01706 (LOOP violations ~3-7c) | PARTIAL — contract-semantics mapping is the labor | Resolution-basis risk (venues have resolved opposite ways) |
| 5 | Polymarket 90-97c settlement-lag theta | near-none (PolySyncer resolution-lag data) | needs backfill | Carry/insurance-selling in disguise; fat left tail; tiny capacity |
| 6 | Political-market underconfidence (calibration slope >1) | Le arXiv 2602.19520 (292M trades) | needs backfill | OPPOSITE sign to #1's bias — domain-conditional, never pool |
| 7 | Turn-of-year tax-loss rebound in small/illiquid prior-year losers | well-covered lineage; residual confined to no-institutional-ownership corner | **YES — testable tonight** (daily OHLCV + PIT membership) | Lowest novelty; must show incrementality vs plain December reversal; S&P 500 may be too large-cap |
| 8 | Perp funding-sign extremes as daily reversal signal | funding carry heavily studied | YES (funding history in lake) | ~Redundant: carry_funding is already a gated candidate; only the extremes-reversal framing is new. Do NOT double-count trials |

## Rejected by the agent (kept for the record)
- S&P 500 index-inclusion effect: decayed 7.4% (1990s) → 0.3% (2010s) per Greenwood-Sammon JF 2025
- ETF rebalance front-running: published (Li SSRN 3967799), competed
- Gamma pinning / OpEx: intraday + paid options data — out of scope
- VIX ETP roll decay: short-vol carry with catastrophic left tail
- USO roll front-running: intraday roll-window, documented temporary impact
- Pure cross-venue arb: bot/latency territory

## Promotion protocol (per gate discipline)
1. Build the Kalshi resolved-market backfill (free API) — unblocks #1/#2/#3/#5/#6.
2. Run ONLY the pre-registered item-9 tests for the top 2 (#1, #2). Each run increments
   n_trials when it touches the registry gate — so run them as research-first
   diagnostics; promote to candidate factors only on a pass.
3. #7 may run any time as a cheap diagnostic on the existing lake (also n_trials-guarded
   if registered).
4. Re-rank trigger: if #1's bias is insignificant in the most recent year (Whelan notes
   shrinkage), demote it and promote #2/#6.
