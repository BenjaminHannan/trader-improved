---
type: synthesis
title: "Research: drawdown-control overlay shape (iteration 13)"
created: 2026-07-06
status: implemented
---

# Research: binary vs graduated drawdown control

## Key Findings
- Continuous drawdown-responsive scaling (risk-control indices, vol-scaled momentum
  literature) achieves the tail protection of hard rules with materially less
  whipsaw; hard on/off thresholds toggle repeatedly when the equity curve oscillates
  around the trigger, paying costs both ways.
- Whipsaw cost is the dominant failure mode of overlay rules (same channel as the
  vol-targeting critique in iteration 3 — the deadband logic applies here too).
- Re-entry asymmetry (hysteresis) is the standard fix: de-risk at the threshold,
  re-risk only after meaningful recovery, never at the same boundary.

## Decision → implementation
`drawdown_multiplier` becomes a graduated ramp with hysteresis, config-driven and
back-compatible:
- multiplier = 1 for DD < threshold; linear ramp from 1 at threshold down to `scale`
  at `2*threshold`; = scale beyond.
- hysteresis: once scaled below 1, return to 1 only when DD recovers below
  `threshold * recovery_frac` (default 0.75) — needs prev state, same optional
  `prev_multiplier` pattern as the vol-target deadband.
- `mode: step` reproduces the old binary rule bit-identically (default for absent
  config keys).

## Verification (in-repo)
Known-answer ramp values at DD = 0.5x/1x/1.5x/2x/3x threshold; hysteresis holds
scaled state during partial recovery and releases below the recovery bound; step
mode identical to old behavior; PIT (equity strictly before t) unchanged; whipsaw
property: oscillating equity around the threshold produces strictly fewer multiplier
changes under ramp+hysteresis than under step.

## Sources
- CSSA "Building a Risk Control Index with Drawdown Protection"; Barroso-style
  vol-scaled momentum (crash reduction); Man Group trend/drawdown notes — search-level
