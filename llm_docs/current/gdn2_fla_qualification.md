---
status: current
last_reviewed: 2026-08-08
---

# Current GDN-2 backend qualification status

## Diagnosis now strongly supported

The completed approximately-20M / 100M run slowed from roughly 3,830 target tok/s early to roughly 445 target tok/s late, with validation slowing by almost the same factor. Data loading was not the bottleneck. Controlled FLA tests on the same Tesla T4 strongly support the explanation that stronger learned GDN-2 decay exposed pathological chunk subdivision / synchronization in the correctness-first adaptive PyTorch backend rather than a need to clip learned decay.

## Standalone T4 qualification

Environment:

```text
Tesla T4, compute capability 7.5
PyTorch 2.10.0+cu128
CUDA 12.8
Triton 3.6.0
fla-core 0.5.1
flash-linear-attention 0.5.1
```

Qualification summary:

```text
forward_correctness: True
backward_correctness: True (normal-decay FP16 gradient parity)
FLA speedup over adaptive, normal forward: 20.830x
FLA speedup over adaptive, strong-decay forward: 162.541x
adaptive strong-decay forward retention: 0.086x
FLA strong-decay forward retention: 0.671x
FLA speedup over adaptive, strong-decay forward+backward: 135.441x
```

At batch 4 / context 2,048:

```text
normal:
  adaptive forward:             177,861 tok/s
  FLA forward:                3,704,843 tok/s
  adaptive forward+backward:    58,623 tok/s
  FLA forward+backward:       1,062,705 tok/s

strong log_decay=-6:
  adaptive forward:              15,296 tok/s
  FLA forward:                2,486,218 tok/s
  adaptive forward+backward:      6,050 tok/s
  FLA forward+backward:         819,392 tok/s
```

FLA therefore preserves the same recurrence while removing most of the catastrophic strong-decay runtime collapse. Decay clipping/bounding is not justified by the slowdown evidence.

## Full Small-LLM integration result

FLA is integrated below the existing `StableGatedDeltaNet2` layer. The Small-LLM projections, convolutions, learned decay, erase/write gates, output path, parameter names, and checkpoint tensors remain unchanged.

The user ran:

```bash
python kaggle/run_gdn2_fla_layer_probe.py
```

and reported:

```text
layer_forward_backward_parity: True
checkpoint_parity: None
INTEGRATION QUALIFIED for checkpoint evaluation; fresh-training authorization remains separate.
```

This qualifies the complete layer forward/backward path. The optional checkpoint branch was not run in that invocation.

## Historical chunk-32 checkpoint vs FLA64 runtime

The active 20M/500M trainer historically passes and serializes:

```text
gdn_chunk_size = 32
```

Trainer restore compares model configuration strictly, so changing the resumed CLI/config to 64 would reject the existing checkpoint.

The checkpoint-compatible integration now separates the two concepts:

```text
saved checkpoint/model config: gdn_chunk_size = 32
adaptive CPU/reference fallback: chunk 32
CUDA FLA recurrence execution: fixed chunk 64
```

This is valid because chunk size changes execution grouping, not the mathematical GDN-2 recurrence or learned parameters. `kaggle/run_gdn2_fla_layer_probe.py` has been refined to test exactly this chunk-32-config / FLA64-runtime path and to accept historical chunk-32 checkpoints.

## 500M resume wiring

The 500M wrapper is fail-closed and creates a detached training worktree at a pinned commit, so changing `model/` on `main` alone would not migrate the training subprocess.

The normal entry point now pins the FLA-integrated implementation commit:

```text
a1471472ca9b5d07f70c844460acffe5c96c5200
```

That worktree contains:

- checkpoint-compatible FLA CUDA recurrence execution;
- `fla-core==0.5.1` in the `model` runtime extra used by the existing trainer command;
- unchanged model/state-dict structure;
- support for historical saved `gdn_chunk_size=32` while running FLA64 internally on CUDA.

The ordinary command remains:

```bash
python kaggle/run_20m_500m.py
```

It still restores only the latest verified remote 500M checkpoint, verifies the dataset/Drive-manifest cursor, and strict-loads model weights, optimizer, scheduler, scaler, RNG state, consumed-token position, and WSD position. After restore, GDN-2 CUDA execution uses FLA.

This is an explicit implementation migration inside the existing 500M trajectory. Exact bitwise continuation versus a hypothetical all-adaptive continuation is not expected because floating-point operation ordering differs.

## Packaging/autotuning

`fla-core==0.5.1` now belongs to the model runtime extra so normal CUDA training cannot accidentally start without the required backend. On Tesla T4, first backward execution may still trigger CPU-heavy Triton autotuning because no matching packaged tuning profile exists; steady-state execution is fast after tuning.

## Current decision boundary

- Do **not** clip/bound GDN-2 decay based on this slowdown.
- FLA is accepted for checkpoint evaluation and for resuming the active 500M trajectory.
- Preserve `gdn_chunk_size=32` in existing checkpoint/model configuration.
- A fresh 500M run from update 1 with FLA remains a separate later decision if a clean single-backend scientific reference is desired.

Evidence:
- [`../evidence/gdn2_fla_t4_full_probe_2026-08-08.md`](../evidence/gdn2_fla_t4_full_probe_2026-08-08.md)
- [`../evidence/gdn2_fla_layer_integration_2026-08-08.md`](../evidence/gdn2_fla_layer_integration_2026-08-08.md)

Decisions:
- [`../decisions/0018-integrate-fla-gdn2-as-checkpoint-compatible-cuda-backend.md`](../decisions/0018-integrate-fla-gdn2-as-checkpoint-compatible-cuda-backend.md)
- [`../decisions/0019-resume-500m-checkpoint-with-fla-gdn2-execution.md`](../decisions/0019-resume-500m-checkpoint-with-fla-gdn2-execution.md)
