---
status: accepted
date: 2026-08-14
---

# ADR 0076: Bound continuous Kaggle SFT host memory

## Context and problem statement

The 100M/2B SFT run is scientifically stable under the qualified two-T4 DDP geometry, but repeated Kaggle runs have ended with rank 0 receiving external `SIGKILL` while rank 1 is subsequently terminated by torchrun. The latest failure occurred during ordinary training immediately after successful optimizer step 657, not during checkpointing or inline evaluation. The final step was finite and healthy.

Earlier cadence telemetry also showed rank 0 near 16 GiB host RSS. Rank 0 uniquely owns W&B, checkpoint/publication, behavior/validation side effects, and the cold FLA/Triton prewarm. In addition, the qualification optimizer instrumentation clones parameter tensors on every optimizer step to derive update diagnostics and the SFT loop serializes the resulting large nested statistics to stdout and W&B. Those diagnostics are not part of optimizer state and are not required for exact resume or training correctness.

## Decision drivers

- Preserve the exact 100M/2B SFT scientific protocol: data order, 4% target budget, microbatch 2 per rank, FP16, LR 3e-5, optimizer update equations, scheduler, checkpoint identity, and 250-step durability cadence.
- Prefer preventing host-memory exhaustion in a single continuous run rather than forcing process segmentation when a direct execution fix is available.
- Preserve exact resume from existing instrumented checkpoints.
- Keep useful scalar training telemetry while removing nonessential per-matrix diagnostic churn.
- Bound glibc allocator arena growth across the two Python/DDP workers sharing Kaggle host memory.

## Considered options

1. Keep the existing runtime and automatically restart torchrun every 250 steps.
2. Reduce or disable W&B entirely.
3. Change the scientific optimizer, batching, or precision.
4. Keep continuous training while removing qualification-only optimizer instrumentation in the Kaggle SFT child process and bounding the host allocator.

## Decision outcome

Choose option 4.

The canonical 100M/2B Kaggle SFT child process is launched with:

- `SMALL_LLM_DISABLE_OPTIMIZER_TELEMETRY=1`, which makes `TrainerEngine` construct the ordinary `HybridMuonAdamW` rather than `InstrumentedHybridMuonAdamW` for the same `hybrid_muon_adamw` configuration;
- `MALLOC_ARENA_MAX=2` to bound glibc arena proliferation;
- `MALLOC_TRIM_THRESHOLD_=131072` to encourage timely return of free heap pages.

The instrumentation-only optimizer fields are not checkpoint state. Existing instrumented optimizer `state_dict()` payloads therefore load into the base hybrid optimizer; the model, Muon momentum, AdamW moments, param groups, scheduler, scaler, RNG state, and counters remain unchanged.

The bounded inline qualification policy from ADR 0075 remains in force: one validation block and two behavior cases while both DDP workers are alive, with full qualification after training.

The 100M/2B profile is pinned to implementation commit `fac40563b7ccaf8b4880e8c4853bc27f0ff337fa`.

## Consequences

Positive consequences:

- Large per-parameter before/after telemetry tensor clones are removed from every Kaggle SFT optimizer step.
- Per-step stdout/W&B payloads become small because `optimizer_update_statistics` is empty on this execution path.
- Host allocator fragmentation/growth is explicitly bounded before Python starts.
- The run can continue from the existing durable checkpoint without altering scientific state.

Negative consequences:

- Kaggle SFT no longer records per-matrix optimizer update diagnostics during the production run. Scalar loss, LR, gradient norm, optimizer-role gradient norms, loss scale, throughput, memory, overflow, validation, and behavior telemetry remain available.
- The mitigation is strongly motivated by the observed rank-0-only SIGKILL pattern, but only a live Kaggle continuation can prove that it eliminates the external kill completely.

If a continuous run still experiences external host-memory SIGKILL after this mitigation, process segmentation at durable 250-step boundaries remains the fallback rather than changing the scientific training protocol.
