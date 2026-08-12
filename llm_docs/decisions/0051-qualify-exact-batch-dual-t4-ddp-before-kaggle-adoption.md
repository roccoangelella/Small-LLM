---
status: accepted
date: 2026-08-12
supersedes: null
---

# 0051 — Qualify exact-batch dual-T4 DDP before Kaggle adoption

## Context and problem statement

Kaggle exposes two Tesla T4 GPUs while the current approximately-20M / 2B pretraining path intentionally uses one T4 with a 16-sequence optimizer block executed as four serial microbatches of four. The second GPU may provide a large throughput gain, but changing global batch size or introducing unqualified distributed reduction semantics would invalidate the frozen scientific trajectory.

The existing single-device trainer normalizes each microbatch loss by the full optimizer block target-token count, accumulates all four microbatch gradients, clips once, and performs one hybrid Muon + AdamW update. PyTorch DDP averages gradients across ranks, so an exact-batch two-rank implementation must compensate that averaging rather than silently halve the effective gradient.

## Considered options

- Keep Kaggle training permanently single-T4 and leave the second accelerator idle.
- Use two T4s by doubling the global optimizer batch from 16 to 32 sequences.
- Split each existing 16-sequence optimizer block across two replicated-model DDP ranks, preserving the global optimizer batch and qualifying numerical parity plus throughput before any production adoption.

## Decision outcome

Chosen option: **add a disposable exact-batch two-T4 DDP qualification harness, without changing production training yet**.

The qualification contract is:

- only the 20M / 2B Kaggle profile is accepted;
- use the real attached schema-v2 `20m-2b-dataset-001` training blocks;
- use the existing 16-sequence global optimizer block and microbatch 4;
- rank 0 receives eight rows and rank 1 receives eight rows;
- each rank executes two microbatches of four;
- the first local backward is inside `DistributedDataParallel.no_sync()` and the second performs the synchronization;
- each local loss contribution is multiplied by `world_size` before division by the global block target-token count, compensating DDP's gradient averaging so the intended global gradient matches the single-T4 formulation;
- use the same approximately-20M GDN-2 hybrid geometry, configured GDN chunk 32, FLA-preferred CUDA execution, FP16 autocast, FP32 master parameters, hybrid Muon + AdamW hyperparameters, gradient clipping, and dynamic loss scaling;
- run the single-T4 and dual-T4 paths from the same deterministic seed over the same ordered blocks;
- compare per-block loss and pre-clipping gradient norm, final model tensors, optimizer-state tensors, and warmed throughput;
- require the first two visible devices to actually be Tesla T4s;
- default throughput promotion gate is at least 1.60x median measured tokens/s, in addition to the numerical-parity gates;
- write only a disposable qualification report; do not write production checkpoints, W&B runs, or remote checkpoint state.

The canonical command is:

```bash
python kaggle/launch.py qualify-dual-t4 --model 20M --tokens 2B
```

The launcher executes the harness through the project `model` uv extra so it uses the same qualified `fla-core==0.5.2` dependency path as Kaggle training.

Passing this harness is evidence that dual-T4 DDP is a viable execution backend. It is **not** authorization to migrate the already-running 20M / 2B trajectory or to change the production trainer. A separate decision is required after measured Kaggle results are available.

## Consequences

### Positive

- The otherwise idle second T4 can be evaluated without changing optimizer batch geometry.
- The test explicitly catches the DDP averaging/normalization error that would otherwise halve gradients.
- Numerical and optimizer-state comparisons make the gate stronger than a throughput-only benchmark.
- The production run remains untouched while the distributed path is still experimental.

### Negative or limiting

- DDP reduction order is not expected to be bitwise identical to the serial single-GPU accumulation order, so qualification uses bounded numerical tolerances rather than hash equality.
- The qualification duplicates a small amount of optimizer-step logic by design; production integration is deferred until the experiment passes.
- Actual NCCL topology and dual-T4 speedup can only be established on Kaggle hardware.

## Validation

Run the canonical qualification command in a Kaggle notebook configured with two T4 GPUs and the exact private 2B dataset attached. The generated `/kaggle/working/dual-t4-qualification.json` must report `status: passed`, all parity checks true, and median dual-T4 throughput at or above the configured speedup threshold.

If parity fails, do not use DDP for production. If parity passes but speedup is below the gate, keep the single-T4 path unless a later decision deliberately lowers the performance threshold for another reason.

## Links

- [`../../kaggle/qualify_dual_t4.py`](../../kaggle/qualify_dual_t4.py)
- [`../../tests/test_kaggle_dual_t4_qualification.py`](../../tests/test_kaggle_dual_t4_qualification.py)
- [`../runbooks/20m_2b_runbook.md`](../runbooks/20m_2b_runbook.md)
- [`0023-run-2b-20m-probe-via-vps-kaggle-dataset.md`](0023-run-2b-20m-probe-via-vps-kaggle-dataset.md)
