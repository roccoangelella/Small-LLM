# Training system

_Last reviewed: 2026-08-13_

## Core optimizer-step contract

One prepared dataset block is one atomic optimizer/update/checkpoint unit. The trainer may split that block into accelerator microbatches, but all loss-bearing targets in the block contribute to one gradient accumulation and one optimizer update.

```text
microbatch size = accelerator memory/execution unit
prepared block  = optimizer, scheduler, cursor, and checkpoint unit
```

A block is acknowledged only after the synchronized optimizer update succeeds. FP16 overflow/non-finite failure leaves it unacknowledged and replayable; skipped updates do not advance schedule, token count, or dataset cursor.

## Model execution

Production CUDA GDN-2 uses `fla-core==0.5.2`, FP32 master parameters, and CUDA FP16 autocast. Saved model config uses `gdn_chunk_size=32`; FLA's internal runtime chunk is 64. The adaptive PyTorch recurrence and reference chunkwise implementations remain correctness/fallback tools, not the selected production CUDA path.

## Optimizer and schedule

The production pretraining optimizer is the project hybrid Muon + AdamW implementation with fail-closed parameter routing. Ordinary feature-transform matrices use Muon; tied embeddings, normalization parameters, GDN dynamics/reference-required offsets, biases, and temporal depthwise filters remain on AdamW. Optimizer recipe/routing is checkpoint state and resume rejects drift.

Schedules advance by successfully committed loss-bearing target tokens/blocks. Current finite experiments use frozen WSD plans derived from their exact dataset horizon. ADR 0057 defines the standard 100M/10B WSD policy; ADR 0058 freezes its exact 76,294-update realization.

## Precision and overflow

For FP16:

1. forward/backward execute under autocast;
2. `GradScaler` scales/unscales gradients;
3. non-finite/overflow state is resolved before optimizer mutation;
4. global clipping occurs once on the complete block gradient;
5. Muon and AdamW update only on success;
6. schedule/token/cursor state commits only after success.

Sensitive GDN state/decay math and Muon orthogonalization/state remain FP32 according to their backend contracts.

## Kaggle topology

ADR 0056 makes exact-batch two-T4 DDP the standard production execution mode for Kaggle training where that qualified path is selected:

- two T4 ranks via NCCL/DDP;
- global prepared block semantics preserved;
- rank-local ordered sequence partitions;
- `no_sync()` on non-final local microbatches;
- synchronized non-finite/overflow decision before either replica can step;
- rank-zero-only W&B, validation, checkpoints, and publication;
- raw unwrapped model serialization so checkpoints remain topology-neutral.

The completed 20M/2B run used the 16-sequence global block / microbatch-4 contract.

## Modal topology

Modal pretraining remains one H100 per run. ADR 0056 does not authorize Modal DDP. The completed 100M/2B final artifact records microbatch 16 on its block-64 dataset geometry.

For the conditional fresh 100M/10B trajectory, cheap CPU producer/staging functions establish verified dataset readiness before H100 dispatch. The H100 consumes the incremental READY frontier through the dynamic schema-v2 reader; it never owns source-corpus production.

## Checkpoint state

A native trainer checkpoint contains, at minimum:

- model parameters/buffers and complete model config;
- optimizer state and routing/recipe identity;
- LR scheduler state;
- FP16 scaler state when applicable;
- global step and consumed-target counters;
- dataset cursor/last acknowledged block;
- validation/best state where applicable;
- Python/PyTorch/CUDA RNG state required for exact resume.

Joint checkpoints are legal only at completed block boundaries with no outstanding accumulation.

## Remote model durability

ADR 0055 uses one Hugging Face model repository:

```text
run/<run_id>/latest.json
run/<run_id>/checkpoints/<checkpoint_id>/last/...
models/<run_id>/artifact.json
models/<run_id>/<checkpoint_id>/...
```

`run/...` is the two-phase live exact-resume namespace. `models/...` is the stable completed-artifact namespace. Stable artifacts verify their native `local_manifest.json`; the live publication manifest belongs to the two-phase `run/...` protocol.

## W&B and validation

Run IDs are stable across exact resume. Validation/checkpoint cadence is profile-specific and recorded in the run configuration/ADR; platform adapters may change execution topology but must not silently change the scientific dataset, optimizer, schedule, or checkpoint identity.

## Completed scaling state

The 20M/2B endpoint is complete at step 61,066 / 2,001,000,448 targets. The 100M/2B endpoint is complete at step 15,267 / 2,001,000,448 targets. Do not treat either as an active unfinished training run.

See [`training_and_evaluation.md`](training_and_evaluation.md), [`gdn2_fla_backend.md`](gdn2_fla_backend.md), and current status for measured endpoints.
