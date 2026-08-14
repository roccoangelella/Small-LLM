---
status: accepted
date: 2026-08-14
---

# ADR 0075 — Bound dual-T4 SFT inline qualification

## Context and problem statement

The live 100M/2B SFT run successfully completed, saved, and remotely published `step-00000250`, then entered the rank-zero-only inline validation phase with two DDP workers still resident. The inline gate requested 16 validation blocks and 16 deterministic behavior cases. Validation produced no intermediate completion event for 1,437 seconds (about 24 minutes), after which rank 0 was terminated with `SIGKILL`; rank 1 then failed only because its Gloo control-barrier peer disappeared.

The durability changes from ADR 0073 worked: `step-00000250` was already remotely durable before validation began. The remaining problem is therefore the cost and resource exposure of running a comparatively large qualification suite inside the two-process Kaggle DDP training lifetime. Rank-zero telemetry showed roughly 15.7 GiB host RSS at validation start, while the second DDP worker remained resident and idle during rank-zero-only qualification.

## Considered options

1. Keep 16 validation blocks and 16 behavior cases inline and continue increasing control-plane timeouts. This does not address the rank-zero `SIGKILL` and wastes the second T4 during a long side effect.
2. Remove all inline evaluation. This maximizes training robustness but gives up the useful periodic health signal.
3. Keep a deliberately small inline health gate during DDP and reserve comprehensive qualification for the separate post-SFT evaluation stage after training workers have exited.

## Decision outcome

Use option 3 for the Kaggle dual-T4 100M/2B SFT runtime.

At each 250-step training cadence, keep the established durability order of checkpoint then remote publication then evaluation, but bound the inline evaluation to:

- 1 validation block;
- 2 deterministic behavior cases.

The inline result is a training-health signal only and is not the final SFT qualification. Comprehensive post-SFT qualification remains mandatory and runs separately after training, when the dual-T4 DDP worker footprint is no longer required.

The existing `step-00000250` checkpoint remains resume-compatible because these evaluation-size arguments do not alter the optimizer/trainer checkpoint identity.

## Consequences

- The next Kaggle launch resumes from remotely published `step-00000250`; steps 1–250 are not repeated.
- Rank-zero-only cadence evaluation should be shorter by roughly an order of magnitude compared with the failed 16+16 gate, reducing exposure to Kaggle host-memory/process termination while retaining a periodic validation/behavior smoke signal.
- Full qualification metrics must not be inferred from the bounded inline gate.
- The 100M/2B profile pins the implementation containing the bounded cadence arguments before the next live run.
