---
status: accepted
date: 2026-08-17
supersedes: 0092
---

# 0093 — Front-load the 10B continuation decay and finish at 1.5e-5

## Context

The step-15,500 diagnostic cooldown showed lower validation loss than the original flat-`3e-4` 100M/10B trajectory within roughly 500 updates. The completed 100M/2B run also showed no obvious learning slowdown late in its cooldown. Together these observations argue against spending a large fraction of the remaining 10B trajectory near the original `3e-4` peak.

ADR 0092 therefore remained too conservative: its pure inverse-square-root continuation would still be near `2.74e-4` after another approximately 400M targets, even though the diagnostic cooldown traverses a much lower LR range over the same span.

The user authorized a more aggressive front-loaded schedule and a terminal LR of `1.5e-5` rather than `3e-5`. The lower terminal target affects only the final cooldown; it does not force the multi-billion-token middle of the run to train at near-terminal LR.

## Decision

Supersede ADR 0092's pure inverse-square-root continuation. The new main continuation must still fork the exact original uncooled `100m-10b-data-001/checkpoints/step-00015500` state, preserving model, optimizer, scaler, RNG, data cursor, architecture, hybrid Muon+AdamW recipe, FP16 precision, microbatch 4, validation prefix, and exact 10B corpus order. Change only the LR scheduler.

Use three post-fork phases:

```text
source step:                 15,500
source targets:              2,031,616,000
source LR:                   3.0e-4

phase 1:                     cosine settle
requested settle span:       300,000,000 targets
block-aligned settle span:   300,023,808 targets
settle updates:              2,289
settle end step:             17,789
settle end targets:          2,331,639,808
settle end LR:               1.5e-4

phase 2:                     inverse-square-root base
formula:                     1.5e-4 * sqrt(2,331,639,808 / committed_targets)
terminal cooldown start:     step 73,242
cooldown-start targets:      9,599,975,424
LR at cooldown start:        ~7.39243e-5

phase 3:                     linear terminal cooldown
cooldown updates:            3,052
cooldown targets:            400,031,744
terminal LR ratio:           0.05 of original 3e-4 peak
terminal LR:                 1.5e-5
final step:                  76,294
final targets:               10,000,007,168
```

The 300M settling phase intentionally tracks the successful diagnostic more closely at the beginning: around 500 updates after the fork its LR is approximately `2.83e-4`, close to the diagnostic cooldown's LR at the same point, while stopping at `1.5e-4` rather than collapsing to a terminal LR. The long inverse-square-root phase then continues learning on fresh data without returning to a high flat plateau.

Lowering the terminal target from `3e-5` to `1.5e-5` makes the final 400M linear cooldown about 34% steeper in LR drop while leaving phases 1 and 2 unchanged.

Implement the schedule as WSqD-style continuation with optional `settle_tokens` and `settle_lr_ratio` trainer parameters. Historical constant/WSD checkpoint identity must remain unchanged, and WSqD configs without a settling phase must retain their previous serialized identity.

Use `beam/aggressive_wsqd_10b_from_15500.py` with run ID `100m-10b-aggressive-wsqd-from-step15500`. Keep this run separate from the ongoing 400M diagnostic and from the superseded ADR-0092 run namespace.

## Consequences

- The old flat-`3e-4` WSD trajectory and ADR-0092 pure inverse-square-root continuation are no longer the authorized main 10B schedule.
- The first approximately 300M targets after step 15,500 spend LR budget more aggressively instead of waiting billions of tokens for meaningful decay.
- The middle of the run remains at materially higher LR than a terminal cooldown, preserving capacity to learn from fresh data.
- The exact 10B endpoint is annealed to `1.5e-5`.
- Any future extension beyond 10B should branch from a pre-terminal-cooldown checkpoint rather than reheating the final model.

## Links

- [`0092-resume-step15500-with-wsqd-style-decay-through-10b.md`](0092-resume-step15500-with-wsqd-style-decay-through-10b.md)
- [`0091-use-step15500-for-controlled-400m-cooldown-probe.md`](0091-use-step15500-for-controlled-400m-cooldown-probe.md)
- [`../runbooks/100m_10b_beam.md`](../runbooks/100m_10b_beam.md)
