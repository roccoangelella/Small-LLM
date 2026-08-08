---
status: current
last_reviewed: 2026-08-08
---

# Current GDN-2 backend qualification status

## Problem being investigated

The completed approximately-20M / 100M run slowed from roughly 3,830 target tok/s early to roughly 445 target tok/s late, with validation slowing by almost the same factor. Data loading was not the bottleneck. The leading explanation is that stronger learned GDN-2 decay makes the correctness-first adaptive PyTorch backend repeatedly subdivide chunks and synchronize with Python, destroying GPU efficiency while preserving the intended recurrence.

## Current experiment

ADR 0016 authorized qualification of Flash Linear Attention (FLA) GDN-2 before changing/clipping learned decay. The one-click probe is:

```bash
python kaggle/run_gdn2_fla_t4_probe.py
```

The full probe including backward was run with:

```bash
python kaggle/run_gdn2_fla_t4_probe.py --with-backward
```

## Full T4 result

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

Raw backend rates at batch 4 / context 2048:

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

Interpretation:

- Forward recurrence parity passes for normal decay, `log_decay=-6`, and extreme `log_decay=-10`.
- Normal-decay backward gradients for q, k, v, log-decay, erase, write, and initial state match the recurrent oracle within the probe tolerance.
- The full FLA strong-decay forward+backward path executes successfully and is about 135.4x faster than the adaptive backend in the same stress case.
- The current adaptive backend keeps only about 8.6% of its normal forward speed in the strong-decay stress regime; FLA keeps about 67.1%.
- FLA is already about 20.8x faster in the normal forward case, showing the existing PyTorch backend is intrinsically expensive even before pathological splitting begins.
- This strongly supports the hypothesis that the late-training slowdown is an implementation/backend problem, not evidence that learned decay itself must be clipped.

Important qualification detail: the current probe performs recurrent-oracle gradient parity only for the normal-decay backward case. The strong-decay forward+backward path is executed and benchmarked successfully, but stress-case gradient parity should be included in the next integration-level test.

## Packaging/autotuning findings

FLA's PyPI packaging is split. `fla-core` contains `fla.ops` and the kernels. Installing only `flash-linear-attention --no-deps` does not provide `fla.ops`; the probe explicitly installs/checks `fla-core==0.5.1` without changing PyTorch/Triton.

On Tesla T4 there is no packaged matching tuning profile, so the first backward invocation falls back to Triton autotuning. At least one GDN-2 non-Hopper backward kernel enumerates 36 configurations. This causes a one-time CPU-heavy compile/autotune phase with low GPU utilization. Once tuned for the geometry, steady-state execution is fast.

## Current decision boundary

Do **not** clip or bound learned GDN-2 decay based on the slowdown evidence.

FLA is now sufficiently qualified at the standalone operator level to justify a Small-LLM integration experiment, but it is **not yet authorized as the production training backend**.

The next gate is a checkpoint-compatible FLA adapter/full-layer integration:

1. preserve the existing Small-LLM GDN layer, projections, parameter names, and checkpoint keys;
2. replace only the chunkwise recurrence calculator with `fla.ops.gdn2.chunk_gdn2`;
3. run full-layer forward/backward parity, including strong-decay gradient parity;
4. verify existing checkpoints load unchanged;
5. run a short optimizer-step replay/mini-training test;
6. only after those pass decide whether to restart the 500M experiment from update 1 with FLA or explicitly migrate a checkpoint.

Detailed full-probe evidence: [`../evidence/gdn2_fla_t4_full_probe_2026-08-08.md`](../evidence/gdn2_fla_t4_full_probe_2026-08-08.md)

Decision: [`../decisions/0016-qualify-fla-gdn2-before-changing-decay.md`](../decisions/0016-qualify-fla-gdn2-before-changing-decay.md)
