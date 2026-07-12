---
type: synthesis
title: "Research: breadth sleeves — rates, international, sectors (iteration 23)"
created: 2026-07-06
status: implemented
---

# Breadth wave: three new ETF sleeves

## Design (pinned)
All three ride the existing yfinance/stooq path, the small-sleeve LW-CC covariance
treatment, and the ERC combiner. New sleeves + id prefixes:
- `rates_etf` (RT:): TLT 2002-07-26, IEF 2002-07-26, SHY 2002-07-26, LQD 2002-07-26,
  HYG 2007-04-11, EMB 2007-12-19, TIP 2003-12-05, BNDX 2013-06-04.
  Factors: tsmom, mom_12_1 + NEW `carry_curve` (fx-style carry from FRED yields:
  for each duration ETF, carry = long-end yield minus cash yield scaled by
  duration bucket; encoded as a per-symbol series map like FX_RATE_SERIES).
- `intl_etf` (IE:): EWJ 1996-03-12, EWG 1996-03-12, EWU 1996-03-12, EWQ 1996-03-12,
  EWA 1996-03-12, EWC 1996-03-12, EWZ 2000-07-10, INDA 2012-02-02, FXI 2004-10-05,
  EWY 2000-05-09, EWT 2000-06-20, EWW 1996-03-12. Factors: mom_12_1, tsmom,
  str_reversal_1m.
- `sector_etf` (SE:): XLK XLF XLE XLV XLI XLP XLY XLU XLB (all 1998-12-16),
  XLRE 2015-10-07, XLC 2018-06-18. Factors: mom_12_1 (relative), str_reversal_1m.
Position caps 10% (rates/intl) / 15% (sector, N=11); cost floors 5bp (rates/intl)
/ 5bp (sector); sleeve risk caps: no new sleeve > 25% of total risk; ERC handles
the rest. Breadth math: 4 -> 7 sleeves of imperfectly correlated premia ≈ +25-30%
IR if edge quality holds; the gate decides per sleeve.

## Why ETFs again
Same execution venue (Alpaca), same daily-bar PIT machinery, zero new loader work —
the marginal sleeve costs config + inception dates + one signal.
