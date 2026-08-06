---
status: accepted
date: 2026-08-06
supersedes: null
---

# 0004 — Run the 100M schedule in one session with 250-step durability

## Context and problem statement

The first approximately-20M-model / approximately-100M-token attempt failed at optimizer update 500 because held-out validation evaluated the complete 16-sequence block at once and exhausted T4 memory.

The experiment launcher also imposed a repository-level maximum of 749 additional updates per Kaggle invocation. At the observed throughput of approximately 4.0k target tokens per second, the complete finite one-pass schedule is expected to fit within a normal T4 notebook session, while frequent verified remote publication can bound interruption loss.

The user does not require recovery of the failed run's local step-250 checkpoint and wants the next attempt to avoid artificial segmentation while increasing validation and remote-upload frequency.

## Considered options

- Keep the 749-update default and continue segmented execution with validation and remote publication every 500 updates.
- Keep segmented execution but shorten segments and publish more frequently.
- Attempt the complete remaining finite one-pass schedule in one invocation, with validation, local checkpointing, and verified remote publication every 250 updates.

## Decision outcome

Chosen option: **attempt the complete remaining one-pass schedule in one Kaggle invocation and perform validation, local checkpointing, and verified remote publication every 250 successful updates**.

The launcher retains `--max-steps-this-session` as an explicit diagnostic or manual override. The finite qualification plan remains authoritative, so removing the default session cap does not permit wraparound or updates beyond the one-pass dataset boundary.

Validation uses a dedicated inference microbatch of one sequence. The training microbatch remains four sequences and the effective optimizer block remains 16 sequences.

A fresh W&B identity, `20m-100m-data-004`, is used for the corrected attempt.

## Consequences

### Positive

- The repository no longer stops a healthy run at update 749 solely for artificial segmentation.
- Validation and verified remote publication happen twice as often as before.
- A platform interruption should lose at most approximately 250 successful updates after the most recent verified remote checkpoint.
- Validation memory is bounded independently of the training microbatch.
- The scientific training recipe, finite data order, optimizer block, and one-pass schedule remain unchanged.

### Negative or limiting

- A single invocation is more exposed to Kaggle's platform-level runtime limit or unexpected notebook termination.
- Validation and remote upload every 250 updates add more overhead than the former 500-update cadence.
- `sys.maxsize` is used by the one-click wrapper to neutralize the legacy default cap without removing the explicit bounded-session interface from the underlying fail-closed launcher.
- The corrected run still requires an empirical T4 validation boundary to confirm real peak memory and runtime behavior.

## Validation

The decision is validated when the corrected run:

1. completes validation at update 250 without CUDA OOM;
2. publishes a verified remote checkpoint at update 250;
3. repeats those operations at update 500;
4. continues beyond update 749 without a repository-imposed stop;
5. never consumes beyond the exact finite one-pass plan;
6. completes with `remaining_steps: 0`, or can resume exactly from the latest 250-step remote checkpoint after a platform interruption.

## Links

- [`../evidence/20m_100m/validation_oom_step_500_2026-08-06.md`](../evidence/20m_100m/validation_oom_step_500_2026-08-06.md)
- [`../runbooks/20m_100m_runbook.md`](../runbooks/20m_100m_runbook.md)
- [`../reference/training_system.md`](../reference/training_system.md)
