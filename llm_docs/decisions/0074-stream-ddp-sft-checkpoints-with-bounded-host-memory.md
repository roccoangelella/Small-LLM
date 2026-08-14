---
status: accepted
date: 2026-08-14
---

# ADR 0074: Stream DDP SFT checkpoints with bounded host memory

## Context and problem statement

The checkpoint-first 100M/2B SFT rerun again completed 250 healthy optimizer updates, then rank 0 was killed by `SIGKILL` immediately after emitting `checkpoint:start`. At that boundary rank 0 reported approximately 16.145 GB resident host memory and 16.158 GB peak resident host memory. No `checkpoint:done` event appeared.

The existing trainer checkpoint path first constructed a full CPU copy of every model and optimizer tensor via recursive `detach().cpu().clone()` calls and only then pickled the aggregate state. That design adds a complete second model/optimizer image to host memory before any checkpoint bytes become durable, which is incompatible with the observed Kaggle host-memory headroom.

## Considered options

1. Keep the existing full CPU-clone checkpoint and reduce SFT training microbatch further. This does not address the host-memory spike because checkpoint serialization happens after the optimizer update and is independent of execution microbatch.
2. Remove optimizer/scheduler/scaler state from checkpoints. This would reduce memory but would violate exact-resume semantics.
3. Stream device-native PyTorch checkpoint storages to disk, preserve all exact-resume state, and retain legacy pickle readability. This avoids materializing one complete extra CPU tensor tree before serialization.

## Decision outcome

Use option 3 for the Kaggle DDP trainer path.

At a DDP checkpoint boundary, the trainer keeps the raw unwrapped model for topology-neutral keys, constructs a device-native state mapping, and writes it with PyTorch serialization rather than recursively cloning the entire model and optimizer to CPU first. Before memory-sensitive checkpoint I/O, collect Python garbage and request release of free glibc heap pages when supported.

Checkpoint loading accepts both the new PyTorch zip serialization and historical plain-pickle `trainer_state.pkl` files. New streamed checkpoints are loaded through the shared state loader, with CPU mmap enabled where applicable so tensor storages can be faulted in lazily. Exact model, optimizer, scheduler, scaler, counters, and RNG state remain part of the checkpoint.

The streaming save is scoped to the DDP production engine. Ordinary single-device pretraining retains its established plain-pickle checkpoint format. The 100M/2B SFT profile pins the implementation commit containing this behavior before another live Kaggle attempt.

## Consequences

- The step-250 checkpoint no longer requires a complete second CPU-resident copy of model and optimizer state before writing.
- Exact-resume semantics and topology-neutral raw-model keys are preserved.
- Historical checkpoints remain readable.
- The parent/SFT loader explicitly releases no-longer-needed host state after model materialization, reducing residual launch-time RSS.
- The next live gate is `checkpoint:start` -> `checkpoint:done` -> local checkpoint -> verified remote publication at step 250. Only after durability succeeds should validation/behavior evaluation run.
- If the next run is still externally killed during checkpointing, the phase telemetry and bounded serializer will have removed the known full-tree clone, so further investigation should focus on remaining process/cgroup memory rather than training microbatch or evaluation ordering.
