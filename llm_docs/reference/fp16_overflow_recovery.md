# FP16 Overflow Recovery

_Last updated: 2026-08-06_

## Purpose

This document defines how one atomic prepared block responds to mixed-precision overflow without changing the data schedule or applying a partial optimizer update.

## GradScaler contract

For CUDA FP16 training, the trainer:

1. accumulates the complete prepared block under autocast;
2. scales each microbatch loss with one shared GradScaler scale;
3. unscales accumulated gradients once after the full block;
4. clips the unscaled global gradient norm;
5. asks GradScaler to execute or skip the optimizer step;
6. commits tokens, scheduler state, global step, and block acknowledgement only after a successful optimizer step.

When scaled gradients contain `inf` or `nan`, GradScaler skips the underlying optimizer step and reduces its scale. The model and optimizer remain at the preceding completed boundary, so the same unacknowledged block can be recomputed safely.

## Adaptive retry allowance

`TrainerConfig.max_overflow_retries` remains serialized in checkpoints and remains a minimum retry allowance. It is no longer the sole hard cutoff.

At the beginning of each block, the trainer reads:

```text
initial_scale = scaler.get_scale()
backoff = scaler.get_backoff_factor()
```

It derives the number of reductions needed to reach scale 1.0:

```text
ceil(log(1 / initial_scale) / log(backoff))
```

The block retry limit is the maximum of that value and the configured retry count. This gives the block a final optimizer attempt at scale 1.0.

Example:

```text
initial scale: 2,048
backoff: 0.5
reductions to scale 1: 11
configured retries: 3
effective retry limit: 11
```

The former policy would stop after four skipped attempts. The adaptive policy may continue through the scale sequence until the attempt at 1.0 succeeds or proves that ordinary FP16 loss-scale calibration is insufficient.

## Fail-closed boundaries

The trainer aborts with the block unacknowledged when:

- gradients remain non-finite after the adaptive retry allowance;
- the scaler exposes an invalid scale or backoff factor;
- the forward loss itself is non-finite.

A non-finite forward loss is not retried by reducing the GradScaler value because loss scaling affects backward gradients, not the already-computed forward value.

The terminal gradient-overflow error records:

```text
block ID
number of skipped attempts
initial scale
current scale
derived retry limit
```

## Compatibility

The policy changes execution logic only. It does not change:

- model parameters or state-dict keys;
- optimizer routing or optimizer state format;
- TrainerConfig serialization;
- checkpoint identity;
- scheduler horizons or learning rate;
- block order, effective tokens per update, or acknowledgement rules;
- W&B run identity.

Existing verified checkpoints therefore remain valid restore sources.

## Validation

Repository tests simulate four skipped optimizer steps with a configured retry count of three. The block must then succeed exactly once, commit its token cursor exactly once, and report four overflow retries.

Operational acceptance additionally requires the T4 run to pass the former global-update-1,497 boundary and publish the next scheduled verified remote checkpoint.

## Related records

- [`../decisions/0006-calibrate-fp16-loss-scale-before-failing-block.md`](../decisions/0006-calibrate-fp16-loss-scale-before-failing-block.md)
- [`../evidence/20m_100m/fp16_overflow_step_1497_2026-08-06.md`](../evidence/20m_100m/fp16_overflow_step_1497_2026-08-06.md)
