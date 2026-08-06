---
status: accepted
date: 2026-08-06
supersedes: null
---

# 0006 — Calibrate FP16 loss scale before failing a block

## Context and problem statement

The repaired 20M / 100M run passed the former GDN-2 failure boundary and completed optimizer update 1,497. The next block then produced scaled-gradient overflow on four consecutive attempts. The trainer's fixed `max_overflow_retries=3` policy raised immediately after the fourth skipped attempt.

The last successful telemetry reported five cumulative overflow events. With the default GradScaler initial scale 65,536, backoff factor 0.5, and no possible 2,000-step growth interval yet, the likely scale entering the failed block was 2,048. Four skipped attempts therefore tested scales 2,048, 1,024, 512, and 256, then stopped after reducing the scale to 128 without trying it.

A skipped GradScaler step does not mutate model or optimizer parameters. The prepared block also remains unacknowledged. Retrying the same block at a lower loss scale is therefore the intended atomic recovery action, not a change to the data schedule.

## Considered options

- Keep the fixed three-retry limit and repeatedly restart from the previous durable checkpoint.
- Raise the serialized `max_overflow_retries` value, making the restored trainer configuration incompatible with existing checkpoints.
- Derive an execution-time retry allowance from the restored scaler scale and backoff factor while keeping the serialized trainer configuration unchanged.
- Switch the experiment globally to FP32 or another precision.

## Decision outcome

Chosen option: **derive an adaptive retry allowance sufficient to reach loss scale 1.0**, because it preserves checkpoint compatibility and lets dynamic loss scaling perform its intended calibration.

The configured retry count remains a minimum. For each prepared block, the trainer computes how many backoff reductions are required to reach scale 1.0 from the block's initial restored scale. It permits at least that many skipped attempts and gives the block a final attempt at scale 1.0.

If gradients remain non-finite at that boundary, training still fails closed and reports the block ID, attempts, initial scale, current scale, and derived retry limit. A non-finite forward loss fails immediately because reducing the backward loss scale cannot repair a forward-pass non-finite value.

## Consequences

### Positive

- Existing step-1,250 and earlier checkpoints remain configuration-compatible.
- Model, optimizer, scheduler, scaler, RNG, and data cursor atomicity remain unchanged.
- Rare high-gradient blocks can calibrate the scaler below the former arbitrary three-retry cutoff.
- A true mixed-precision failure still stops rather than applying a non-finite update.
- Future failures carry enough scale diagnostics to distinguish calibration exhaustion from model divergence.

### Negative or limiting

- A difficult block may be recomputed several additional times.
- This repair does not prove that every block will succeed at loss scale 1.0.
- W&B retains the failed and replayed history tails as already documented.

## Validation

- Unit-test a simulated block that skips four optimizer attempts even though the configured retry count is three, then succeeds and commits exactly once.
- Confirm the derived retry limit from scale 2,048 and backoff 0.5 is 11, allowing a final attempt at scale 1.0.
- Resume from the actual latest verified remote checkpoint and require passage beyond global update 1,497.
- Continue checking that overflow retries do not advance the block cursor, scheduler, global step, or optimizer state.

## Links

- [`../evidence/20m_100m/fp16_overflow_step_1497_2026-08-06.md`](../evidence/20m_100m/fp16_overflow_step_1497_2026-08-06.md)
- [`../reference/training_system.md`](../reference/training_system.md)
- [`../runbooks/20m_100m_runbook.md`](../runbooks/20m_100m_runbook.md)
