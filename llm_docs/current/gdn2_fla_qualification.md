---
status: current
last_reviewed: 2026-08-08
---

# Current GDN-2 backend qualification status

## Problem being investigated

The completed approximately-20M / 100M run slowed from roughly 3,830 target tok/s early to roughly 445 target tok/s late, with validation slowing by almost the same factor. Data loading was not the bottleneck. The leading explanation is that stronger learned GDN-2 decay makes the correctness-first adaptive PyTorch backend repeatedly subdivide chunks and synchronize with Python, destroying GPU efficiency while preserving the intended recurrence.

## Current experiment

ADR 0016 authorizes qualification of Flash Linear Attention (FLA) GDN-2 before changing/clipping learned decay. The one-click probe is:

```bash
python kaggle/run_gdn2_fla_t4_probe.py
```

Backward qualification is explicit:

```bash
python kaggle/run_gdn2_fla_t4_probe.py --with-backward
```

## Forward result on Tesla T4

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
backward_correctness: pending
FLA speedup over adaptive, normal decay: 21.361x
FLA speedup over adaptive, strong decay: 160.719x
adaptive strong-decay speed retention: 0.090x
FLA strong-decay speed retention: 0.676x
```

Interpretation:

- The current adaptive backend keeps only 9% of its normal forward speed in the strong-decay stress regime.
- FLA keeps 67.6% of its normal forward speed under the same strong decay.
- FLA is already about 21.4x faster in the normal case and about 160.7x faster in the pathological strong-decay case.
- The same strong-decay GDN-2 recurrence therefore runs correctly on T4 without the catastrophic collapse of the current backend.
- This strongly supports the hypothesis that the late-training slowdown is an implementation/backend problem, not evidence that learned decay itself must be clipped.

## Packaging/autotuning findings

FLA's PyPI packaging is split. `fla-core` contains `fla.ops` and the kernels. Installing only `flash-linear-attention --no-deps` does not provide `fla.ops`; the probe now explicitly installs/checks `fla-core==0.5.1` without changing PyTorch/Triton.

The first backward attempt appeared stuck after forward correctness with CPU near 100% and GPU nearly idle. Investigation showed that FLA falls back to Triton autotuning when no matching cached GPU configuration exists. There is no packaged Tesla T4 tuning profile; GDN-2 backward includes multiple kernels, with at least one non-Hopper kernel enumerating 36 configurations. Treat the first backward compile/autotune phase as a qualification cost, not as proof of a GDN mathematical hang.

## Current decision boundary

Do **not** clip or bound learned GDN-2 decay based on the slowdown evidence at this point.

Do **not** integrate FLA into production training yet.

The next mandatory gate is backward qualification: gradients must match the recurrent oracle and the strong-decay forward+backward path must execute reliably and materially faster than the current adaptive backend. Only then should a checkpoint-compatible FLA adapter/full-layer integration be considered.

Detailed evidence: [`../evidence/gdn2_fla_t4_forward_qualification_2026-08-08.md`](../evidence/gdn2_fla_t4_forward_qualification_2026-08-08.md)

Decision: [`../decisions/0016-qualify-fla-gdn2-before-changing-decay.md`](../decisions/0016-qualify-fla-gdn2-before-changing-decay.md)
