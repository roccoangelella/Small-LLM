---
status: accepted
date: 2026-08-12
supersedes: null
---

# 0056 — Adopt exact-batch dual-T4 DDP for Kaggle only

## Context and problem statement

ADR 0051 authorized a disposable two-T4 qualification but deliberately stopped short of changing production training. The live Kaggle qualification has now passed every predeclared numerical and throughput gate on two Tesla T4 GPUs using the real 20M/2B dataset geometry.

The warmed median throughput was 20,183.50 target tok/s on one T4 and 34,292.22 target tok/s under two-T4 DDP, a 1.6990x median speedup against the 1.60x promotion threshold. Loss, gradient, parameter, and optimizer-state parity all passed with substantial margin.

The project also has a distinct Modal training path. Modal uses a single H100 and has different optimizer-block/microbatch utilization decisions; the Kaggle DDP result is not evidence for changing that topology.

## Decision outcome

Chosen option: **make exact-batch two-T4 DDP the standard production execution mode for Kaggle training, while keeping Modal training single-H100.**

For Kaggle production training:

- launch two replicated-model processes through `torch.distributed.run` / NCCL;
- require exactly two visible Tesla T4 GPUs for this qualified path;
- preserve the existing 16-sequence global optimizer block;
- give each rank eight ordered sequences from the same global block;
- keep microbatch size four, yielding two local microbatches per rank;
- execute every non-final local backward under `DistributedDataParallel.no_sync()` and synchronize the final local backward;
- multiply each local summed loss by `world_size` before dividing by the global block target-token count, compensating DDP gradient averaging;
- clip gradients once after the synchronized DDP reduction;
- synchronize forward/non-finite and gradient-overflow status across ranks **before either optimizer replica can step**;
- on a global FP16 overflow, both ranks skip the optimizer step, reduce the scaler identically, and retry the same unacknowledged optimizer block;
- advance global step, consumed-token count, schedule, and dataset cursor only after the synchronized optimizer update succeeds;
- keep W&B, held-out validation, checkpoint writes, and remote publication rank-zero-only;
- serialize and restore the unwrapped raw model so checkpoints never acquire `module.` keys and remain topology-neutral;
- retain the per-experiment pinned model/trainer worktree; DDP is injected as a Kaggle execution adapter rather than moving the scientific trajectory to the controlling checkout's newer trainer implementation;
- pin the Kaggle DDP subprocess to the qualified T4 runtime: PyTorch 2.10.0+cu128 / CUDA 12.8, Triton 3.6.0, `fla-core==0.5.2`;
- use the same six-representative-config Triton autotune policy that was used by the successful qualification;
- prewarm the raw rank-zero model once at the exact local 4x2048 shape before DDP wrapping, with no optimizer step or token/cursor mutation, so rank one can reuse the populated Triton cache instead of independently cold-autotuning the same kernels.

For Modal:

- **do not enable DDP or multi-GPU execution from this decision;**
- keep Modal training on one H100 per training run;
- retain Modal's independently qualified block/microbatch behavior and checkpoint/publication infrastructure.

Offline Kaggle microbatch probes and the disposable dual-T4 qualification command remain separate from the production DDP rewrite.

## Evidence

The accepted qualification reported:

```text
status: passed
single-T4 median:     20,183.496 target tok/s
dual-T4 DDP median:  34,292.221 target tok/s
median speedup:      1.699023x
minimum gate:        1.60x

maximum loss delta:               4.76837158203125e-06
maximum gradient relative delta:  7.535586156002224e-06
parameter relative L2:            2.7796250767485542e-05
optimizer relative L2:            0.00021572932322973847
```

All loss, gradient, parameter, optimizer, and throughput checks were true.

Canonical evidence: [`../evidence/20m/20m_2b_dual_t4_ddp_qualification_2026-08-12.md`](../evidence/20m/20m_2b_dual_t4_ddp_qualification_2026-08-12.md)

## Consequences

### Positive

- Kaggle uses both available T4s instead of leaving one idle.
- The measured optimizer-step throughput improves by about 70% without changing the scientific global optimizer batch.
- Checkpoint identity remains independent of whether a run is resumed under one-device or two-device execution.
- The production overflow path is stronger than the disposable harness because optimizer stepping cannot diverge across ranks after an asymmetric non-finite event.
- Rank-zero-only side effects prevent duplicate W&B runs, validation, checkpoints, or remote publications.
- Modal remains simple and tuned for the very different single-H100 environment.

### Negative or limiting

- Kaggle training now requires a notebook/session configured with two T4 GPUs; it fails closed rather than silently falling back to one T4.
- A fresh Kaggle runtime can still pay several minutes of one-time FLA/Triton compilation/autotuning before the first scientific optimizer update.
- DDP reductions are numerically equivalent within the qualified tolerances, not bitwise identical to serial accumulation.
- The six-config autotune cap is an execution-performance policy, not proof that the globally fastest Triton configuration was selected.

## Validation

The standard Kaggle train launcher must expose `execution=dual_t4_ddp` and rewrite only the online production trainer subprocess to two-process torchrun. The pinned trainer flags, dataset block geometry, schedule, optimizer, checkpoint cadence, and W&B identity must otherwise remain unchanged.

Production tests must cover at least:

- two-process torchrun command construction;
- qualified runtime pins;
- preservation of the offline single-process probe path;
- synchronized non-finite/overflow decision before optimizer stepping;
- raw-model checkpoint/evaluation adapter behavior;
- absence of the Kaggle DDP adapter from Modal launch code.

The first live production resume should be checked for the expected startup banner, successful topology-neutral checkpoint restore, rank-zero-only W&B continuation, and warmed per-block throughput consistent with the qualified order of magnitude.

## Links

- [`0051-qualify-exact-batch-dual-t4-ddp-before-kaggle-adoption.md`](0051-qualify-exact-batch-dual-t4-ddp-before-kaggle-adoption.md)
- [`../../kaggle/dual_t4_runtime.py`](../../kaggle/dual_t4_runtime.py)
- [`../../kaggle/dual_t4_train.py`](../../kaggle/dual_t4_train.py)
- [`../../kaggle/launch.py`](../../kaggle/launch.py)
- [`../evidence/20m/20m_2b_dual_t4_ddp_qualification_2026-08-12.md`](../evidence/20m/20m_2b_dual_t4_ddp_qualification_2026-08-12.md)
