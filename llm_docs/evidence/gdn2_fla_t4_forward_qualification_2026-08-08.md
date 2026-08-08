---
status: evidence
date: 2026-08-08
scope: gdn2_backend_qualification
---

# FLA GDN-2 T4 forward qualification — 2026-08-08

## Why this probe exists

The completed approximately-20M / 100M pretraining run suffered a severe late-training throughput collapse. Training throughput fell from about 3,830 target tok/s early in the accepted trajectory to about 445 target tok/s at the end, while validation time slowed by nearly the same factor. Data wait remained negligible. The leading implementation hypothesis is that learned GDN-2 decay becomes stronger as training progresses, causing the correctness-first `AdaptiveChunkwiseGDN2Backend` to repeatedly subdivide large chunks into smaller numerical subchunks. The model remains mathematically valid, but the execution becomes increasingly sequential and synchronization-heavy.

The purpose of this qualification is to test whether the slowdown is fundamentally caused by learned decay itself or by the current chunkwise implementation. Flash Linear Attention (FLA) v0.5.1 provides an MIT-licensed optimized GDN-2 Triton training kernel with the same recurrence. If FLA can execute the same strong-decay recurrence without the catastrophic slowdown, then clipping/bounding learned decay should not be treated as the primary fix.

## Probe entry point

```bash
python kaggle/run_gdn2_fla_t4_probe.py
```

Environment used by the successful forward qualification:

```text
GPU: Tesla T4
compute capability: 7.5
PyTorch: 2.10.0+cu128
CUDA runtime: 12.8
Triton: 3.6.0
fla-core: 0.5.1
flash-linear-attention: 0.5.1
VRAM: about 14.6 GiB
```

The probe uses the approximately-20M model's GDN geometry and defaults to batch 4, context 2,048 for the benchmark. It compares FLA against both the tokenwise recurrent oracle and the current adaptive chunkwise backend.

## Installation correction discovered during qualification

The initial probe installed `flash-linear-attention==0.5.1 --no-deps` and then failed with:

```text
ModuleNotFoundError: No module named 'fla.ops'
```

This was a probe packaging mistake, not a GPU failure. FLA's PyPI packaging is split: `fla-core` provides `fla.ops` and the kernels, while `flash-linear-attention` is the higher-level extension package. The probe was corrected to explicitly install `fla-core==0.5.1 --no-deps` while leaving the notebook's existing PyTorch/Triton stack untouched.

## First backward attempt and Triton autotuning behavior

After the packaging fix, the first run reached:

```text
[correctness] fla_vs_recurrent:normal:fp16
output PASS
state  PASS
```

and then appeared stuck with CPU near 100% and GPU nearly idle. This occurred after forward correctness, when the probe entered its first gradient test.

Inspection of FLA's kernel cache/autotuning path showed that when no matching GPU config cache is present, FLA falls back to Triton autotuning. There is no packaged Tesla T4 tuning profile. GDN-2 backward contains several Triton kernels; one non-Hopper backward kernel alone enumerates 36 candidate configurations. Therefore the observed CPU-heavy/GPU-idle period is consistent with first-use Triton compilation/autotuning rather than a mathematical GDN hang.

The probe was changed so that default execution qualifies forward first and makes backward qualification explicit with `--with-backward`.

## Forward qualification result

Successful summary:

```text
forward_correctness: True
backward_correctness: None
forward_fla_speedup_over_adaptive_normal: 21.361x
forward_fla_speedup_over_adaptive_stress: 160.719x
forward_adaptive_stress_retention: 0.090x
forward_fla_stress_retention: 0.676x
train_fla_speedup_over_adaptive_stress: None
verdict: FORWARD QUALIFIED only. This is enough to test the strong-decay runtime hypothesis, but not enough to authorize training integration.
```

Definitions:

- `forward_adaptive_stress_retention = adaptive strong-decay speed / adaptive normal-decay speed`.
- `forward_fla_stress_retention = FLA strong-decay speed / FLA normal-decay speed`.
- `forward_fla_speedup_over_adaptive_normal = FLA normal-decay speed / adaptive normal-decay speed`.
- `forward_fla_speedup_over_adaptive_stress = FLA strong-decay speed / adaptive strong-decay speed`.

## Interpretation

The current adaptive backend retains only 9.0% of its normal forward speed under the strong-decay stress case. In other words, the same backend becomes about 11.1x slower purely because decay enters the problematic regime.

FLA retains 67.6% of its normal forward speed under the same stress. Strong decay still has a performance cost, but it does not trigger the catastrophic collapse seen in the current backend.

FLA is about 21.4x faster than the current adaptive backend even in the normal-decay benchmark, showing that the correctness-first eager PyTorch implementation is intrinsically expensive. In the strong-decay regime, FLA is about 160.7x faster than the current adaptive backend.

This is strong evidence for two distinct implementation facts:

1. the current PyTorch adaptive GDN-2 backend is substantially slower than a specialized fused implementation even before pathological decay; and
2. strong learned decay causes an additional catastrophic slowdown specifically in the adaptive implementation.

Most importantly, the experiment demonstrates that the same GDN-2 recurrence can execute correctly on a Tesla T4 under strong decay without the current backend's collapse. Therefore strong learned decay is not, by itself, sufficient reason to clip or bound the model's decay. The primary problem is the implementation used to evaluate the recurrence.

## What is and is not qualified

Qualified so far:

- FLA GDN-2 imports and executes on Tesla T4.
- Forward outputs and final recurrent state match the project's recurrent oracle within the probe tolerances.
- The optimized forward path tolerates the strong-decay regime used to stress the current backend.
- FLA strongly reduces the strong-decay performance collapse on T4.

Not yet qualified:

- backward/gradient correctness on T4;
- forward+backward steady-state speed;
- optimizer-step integration;
- full `StableGatedDeltaNet2` layer parity;
- checkpoint-compatible training integration;
- exact resume/replay behavior after integration.

No production backend replacement is authorized from the forward result alone.

## Next gate

Run:

```bash
python kaggle/run_gdn2_fla_t4_probe.py --with-backward
```

The first invocation may spend substantial CPU time compiling/autotuning Triton backward kernels. The probe now emits explicit phase output so this work is distinguishable from a hang.

Training integration should be considered only if backward gradients match the recurrent oracle and the strong-decay forward+backward path remains materially faster than the current adaptive backend.

## Current conclusion

The forward experiment strongly supports the original runtime hypothesis: the severe late-training slowdown is primarily an execution/backend problem caused by adaptive numerical chunk fragmentation and synchronization as learned decay strengthens. It does not currently justify clipping learned GDN-2 decay. FLA is a credible replacement candidate, but backward qualification remains the next mandatory gate.

Related decision: [`../decisions/0016-qualify-fla-gdn2-before-changing-decay.md`](../decisions/0016-qualify-fla-gdn2-before-changing-decay.md)
