---
status: current
last_reviewed: 2026-08-08
---

# Current GDN-2 backend qualification status

## Diagnosis

The completed approximately-20M / 100M run slowed from roughly 3,830 target tok/s early to roughly 445 target tok/s late while validation kept improving. Controlled FLA tests on the same Tesla T4 strongly support the explanation that stronger learned GDN-2 decay exposed pathological chunk subdivision / synchronization in the correctness-first adaptive PyTorch backend rather than a need to clip learned decay.

## Standalone FLA operator qualification — PASSED

Environment:

```text
Tesla T4, compute capability 7.5
PyTorch 2.10.0+cu128
CUDA 12.8
Triton 3.6.0
fla-core 0.5.1
flash-linear-attention 0.5.1
```

Key results:

```text
forward_correctness: True
backward_correctness: True (normal-decay FP16 gradient parity)
FLA speedup over adaptive, normal forward: 20.830x
FLA speedup over adaptive, strong-decay forward: 162.541x
adaptive strong-decay forward retention: 0.086x
FLA strong-decay forward retention: 0.671x
FLA speedup over adaptive, strong-decay forward+backward: 135.441x
```

FLA therefore preserves the same recurrence while removing most of the catastrophic strong-decay runtime collapse. Decay clipping/bounding is not justified by the slowdown evidence.

## First full-layer integration probe — PASSED, but precision coverage was incomplete

The first integrated layer probe reported:

```text
layer_forward_backward_parity: True
checkpoint_parity: None
INTEGRATION QUALIFIED for checkpoint evaluation; fresh-training authorization remains separate.
```

However, that probe converted the whole candidate/reference layer to FP16 with `model.half()`. It did not reproduce the real trainer precision contract of FP32 master parameters plus CUDA FP16 autocast.

That distinction became material in the first resumed 500M attempt.

## First 500M FLA resume attempt — FAILED CLOSED BEFORE UPDATE 4001

The launcher successfully restored the verified step-4000 checkpoint and attempted global steps 4001–15264. The first resumed update did not complete. Triton compilation failed inside FLA WY recomputation:

```text
Both operands must be same dtype. Got fp32 and fp16
b_u = tl.dot(b_A, b_vb)
```

No successful update 4001 was produced, so the latest verified checkpoint remains step 4000. Model weights, optimizer state, scheduler/WSD position, scaler, RNG state, and data cursor remain intact at that checkpoint.

Root cause: under the real trainer's FP32-master + FP16-autocast path, normalized q/k can enter the FLA adapter as FP32 while v/write are FP16. FLA v0.5.1 allocates its solved WY matrix with `k.dtype`, then dots it with a v/write block. Triton requires both dot operands to use the same dtype.

This is an adapter precision-contract bug, not a recurrence mismatch, checkpoint mismatch, strong-decay failure, or T4 incompatibility.

Detailed evidence: [`../evidence/gdn2_fla_500m_resume_amp_dtype_failure_2026-08-08.md`](../evidence/gdn2_fla_500m_resume_amp_dtype_failure_2026-08-08.md)

## AMP-safe adapter fix on main

`model/gdn2_fla.py` now canonicalizes the ordinary FLA compute tensors:

```text
q, k, v, erase, write -> v.dtype
```

In the active trainer this gives the already-qualified FP16 compute contract. Log-decay and recurrent state remain FP32. The FLA result is cast back to the original Small-LLM q dtype before returning, and the internal casts remain differentiable.

The integration probe has also been corrected. It now keeps model parameters in FP32 and executes both reference and FLA paths under CUDA FP16 autocast, matching the real trainer instead of blanket-casting the model to FP16.

## Historical chunk-32 checkpoint vs FLA64 runtime

The active 20M/500M checkpoint stores:

```text
gdn_chunk_size = 32
```

That configuration remains unchanged for strict checkpoint restore. CUDA recurrence execution uses FLA's fixed internal chunk size 64. CPU/reference execution keeps the historical adaptive chunk 32. Chunk size is execution grouping, not learned model state.

## 500M launcher wiring

The normal wrapper creates a detached training worktree, so changing `model/` on current `main` alone is insufficient. The wrapper is now repinned to the implementation containing the AMP-safe FLA adapter and revised probe:

```text
efa3d10327af1ade96db5363616e00c870b164dc
```

The pinned worktree also includes `fla-core==0.5.1` in the `model` runtime extra. The trainer still passes `--gdn-chunk-size 32` so historical checkpoint configuration remains strict-load compatible.

## Mandatory gate before retrying 500M resume

Run on the T4:

```bash
git pull --ff-only
python kaggle/run_gdn2_fla_layer_probe.py
```

Require:

```text
layer_forward_backward_parity: True
trainer_amp_contract_tested: True
```

Only after that revised AMP-realistic probe passes should the ordinary resume command be retried:

```bash
python kaggle/run_20m_500m.py
```

## Current decision boundary

- Do **not** clip/bound GDN-2 decay based on the slowdown.
- Standalone FLA operator qualification remains passed.
- The first layer probe was insufficient for the real trainer precision contract.
- The active 500M trajectory remains safely resumable from verified step 4000; no update was committed by the failed FLA attempt.
- Retry the 500M resume only after the revised AMP-realistic integration probe passes.
- Preserve `gdn_chunk_size=32` in the existing checkpoint/model configuration while FLA uses fixed chunk 64 internally on CUDA.
- A fresh 500M FLA-from-update-1 run remains a separate later decision.

Evidence:
- [`../evidence/gdn2_fla_t4_full_probe_2026-08-08.md`](../evidence/gdn2_fla_t4_full_probe_2026-08-08.md)
- [`../evidence/gdn2_fla_layer_integration_2026-08-08.md`](../evidence/gdn2_fla_layer_integration_2026-08-08.md)
- [`../evidence/gdn2_fla_500m_resume_amp_dtype_failure_2026-08-08.md`](../evidence/gdn2_fla_500m_resume_amp_dtype_failure_2026-08-08.md)

Decisions:
- [`../decisions/0018-integrate-fla-gdn2-as-checkpoint-compatible-cuda-backend.md`](../decisions/0018-integrate-fla-gdn2-as-checkpoint-compatible-cuda-backend.md)
- [`../decisions/0019-resume-500m-checkpoint-with-fla-gdn2-execution.md`](../decisions/0019-resume-500m-checkpoint-with-fla-gdn2-execution.md)
