---
status: accepted
date: 2026-08-18
supersedes: null
---

# Diagnose the first dual-T4 update and stabilize the Triton cache contract
## Context and problem statement

No separate context was recorded at the time; section added for the template.


Date: 2026-08-18

For the 100M/10B Kaggle deep-decay continuation, do not rerun the opaque post-prewarm path unchanged after the live step-18000 retry stalled for more than eleven minutes after the successful startup rendezvous.

The next retry must:

- emit per-rank milestones around DDP construction, first session-step entry, first batch acquisition, local microbatch progress, gradient synchronization, and optimizer-step completion;
- keep these diagnostics execution-only and scientifically neutral;
- stop treating `kaggle/dual_t4_train.py` and `kaggle/dual_t4_train_block64.py` as Triton kernel-source inputs, because barrier/logging/control-flow edits in those wrappers do not change the compiled model/FLA kernels and should not invalidate a compatible portable cache;
- continue validating the actual model/FLA-facing source files plus the frozen runtime/geometry contract.

The scientific checkpoint, data cursor, optimizer state, LR schedule, block64 global batch, and microbatch-2 execution slicing remain unchanged.
## Considered options

No alternative was recorded at the time; section added for the template.

## Decision outcome

No separate decision outcome was recorded at the time; section added for the template.


## Consequences

No consequences were recorded at the time; section added for the template.
