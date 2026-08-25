---
status: accepted
date: 2026-08-25
---

# ADR 0125: define aggressive decay from token zero for fresh training runs

## Decision

The aggressive learning-rate policy validated through the current 100M/10B continuation must be treated as a **full-run policy for future fresh training**, not as a schedule intrinsically tied to the historical step-15,500 fork.

For any new pretraining run started from random initialization, and for the accepted 100M/2B 10% SFT run whose post-training optimizer starts from its own token-zero clock, define the scheduler against the complete run horizon beginning at target token 0.

The historical `step-00015500` continuation remains evidence for the policy family, but it is not a reusable absolute schedule anchor.

## Fresh-run geometry

A fresh run may retain the standard short LR warmup required when optimizer state starts from zero. After that warmup, enter the aggressive schedule immediately:

1. warm up from the initial LR to the run-specific peak;
2. immediately perform the short cosine settle from peak LR to the lower settle LR;
3. spend the large majority of the remaining horizon in monotonic power-law decay;
4. reserve only the final short segment for the linear cooldown to the final LR floor.

There must be **no artificial stable plateau corresponding to the old 0 -> step-15,500 pretraining history** and no attempt to place the start of deep decay at the same ~20% normalized coordinate merely because the validating continuation forked there.

For fresh SFT, the same rule applies to the SFT target-token clock: token 0 means the start of SFT optimization, not the parent model's already-consumed pretraining tokens.

## Relationship to ADR 0124

ADR 0124 remains accepted in choosing the pretraining aggressive-decay policy family for the 100M/2B 10% SFT trial, but its re-anchoring requirement is now clarified by this ADR: the schedule must be solved as a fresh token-zero trajectory, with no continuation-style 20% flat section.

The exact LR magnitudes for SFT remain a separate numerical choice until explicitly frozen. The current leading candidate remains the previously proposed 0.1x scaling of the validated pretraining LR anchors (`3e-5 -> 1e-5 -> 1e-6 -> 5e-7`), but this ADR does not independently promote those candidate values.

## Future pretraining requirement

Future fresh pretraining launchers must expose the aggressive policy directly from run initialization rather than requiring a later fork/continuation command. The existing step-15,500 continuation implementation is historical/experimental evidence and compatibility machinery, not the desired architecture for the next fresh pretraining trajectory.
