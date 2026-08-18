---
status: accepted
date: 2026-08-18
supersedes: 0093
---

# 0094 — Lower the post-15,500 LR scale throughout the 10B continuation

## Context

The first aggressive continuation in ADR 0093 forked the exact uncooled `step-00015500`, cosine-decayed from `3e-4` to `1.5e-4` over approximately 300M targets, then switched to an inverse-square-root base. Its fixed-prefix validation loss fell rapidly during the fast settling phase, bottomed near the end of that phase, and then began rising once the schedule changed to the much gentler inverse-square-root continuation.

The user judged this sufficient evidence that the learning rate should be more aggressive not only during the initial settling phase but also throughout the long middle of the 10B run.

Recent continuation-schedule work still favors a decreasing base close to inverse-square-root rather than a long flat WSD plateau. WSqD uses a shifted inverse-square-root base plus terminal linear cooldown, and the related power schedule uses an empirically fitted exponent close to `-0.5`. We therefore keep the approximately inverse-square-root long-phase shape and lower its scale rather than introducing an unsupported much steeper exponent from a single project run.

## Decision

Supersede ADR 0093 as the authorized main 100M/10B continuation. Always fork the exact original uncooled `100m-10b-data-001/checkpoints/step-00015500` state; do not continue from the partially trained ADR-0093 branch and do not reheat a cooled checkpoint.

Preserve model, optimizer, scaler, RNG, data cursor, architecture, hybrid Muon+AdamW recipe, FP16 precision, microbatch 4, frozen validation prefix, and exact 10B corpus order. Change only the LR scheduler.

Use this three-phase schedule:

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
settle end LR:               1.0e-4

phase 2:                     inverse-square-root base
formula:                     1.0e-4 * sqrt(2,331,639,808 / committed_targets)
terminal cooldown start:     step 73,242
cooldown-start targets:      9,599,975,424
LR at cooldown start:        ~4.92828e-5

phase 3:                     linear terminal cooldown
cooldown updates:            3,052
cooldown targets:            400,031,744
terminal LR ratio:           0.05 of original 3e-4 peak
terminal LR:                 1.5e-5
final step:                  76,294
final targets:               10,000,007,168
```

Compared with ADR 0093, phase 1 now settles to `1e-4` instead of `1.5e-4`. Because phase 2 is anchored at that lower LR, every subsequent inverse-square-root LR before terminal cooldown is one-third lower than the ADR-0093 counterpart. Approximate base LR landmarks are `9.43e-5` at step 20,000, `7.70e-5` at step 30,000, `6.67e-5` at step 40,000, `5.96e-5` at step 50,000, and `4.93e-5` at the terminal-cooldown boundary.

Use `beam/strong_decay_10b_from_15500.py` with run ID `100m-10b-strong-decay-from-step15500`. Keep its local, W&B, and Hugging Face checkpoint namespaces separate from the ADR-0093 run and all earlier diagnostics.

## Consequences

- The ADR-0093 run becomes schedule evidence, not the parent of the new main trajectory.
- LR is more aggressive both in the first 300M-target settling phase and throughout the multi-billion-token middle phase.
- The long phase retains the state-of-the-art near-inverse-square-root continuation shape rather than inventing a substantially steeper power exponent from one run.
- The exact 10B endpoint remains annealed to `1.5e-5` over the final approximately 400M targets.
- Any later extension beyond 10B should branch from a pre-terminal-cooldown checkpoint rather than reheating the final model.

## Links

- [`0093-front-load-10b-decay-and-use-1p5e-5-terminal-lr.md`](0093-front-load-10b-decay-and-use-1p5e-5-terminal-lr.md)
- [`0091-use-step15500-for-controlled-400m-cooldown-probe.md`](0091-use-step15500-for-controlled-400m-cooldown-probe.md)
- [`../runbooks/100m_10b_beam.md`](../runbooks/100m_10b_beam.md)
