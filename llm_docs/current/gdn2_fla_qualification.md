---
status: current
last_reviewed: 2026-08-08
---

# Current GDN-2 backend qualification status

## Bottom line

FLA v0.5.1 `chunk_gdn2` is **not qualified for training the active 20M/500M trajectory**.

The real verified `step-00004000` checkpoint produces decay values that overlap the trainer-AMP backward failure region measured in controlled full-layer tests. Do not resume update 4001 with FLA chunk backward.

The latest accepted trajectory point remains:

```text
checkpoint: step-00004000
last_consumed_block_id: 3999
next update: 4001
```

No FLA experiment has committed update 4001.

## Why FLA was investigated

The completed approximately-20M / 100M run slowed from roughly 3,830 target tok/s early to roughly 445 target tok/s late while validation kept improving. Data wait stayed tiny. The strongest diagnosis remains that stronger learned GDN-2 decay made the correctness-first adaptive PyTorch backend repeatedly subdivide chunks and synchronize with Python.

Standalone Tesla T4 tests showed very large FLA forward and forward+backward throughput advantages, especially under strong decay. Those results still support the throughput diagnosis, but they did not fully qualify strong-decay gradient correctness.

## Qualification history

### Standalone operator

Environment:

```text
Tesla T4
PyTorch 2.10.0+cu128
CUDA 12.8
Triton 3.6.0
fla-core 0.5.1
```

Observed:

```text
forward correctness: pass
normal-decay backward gradient parity: pass
normal forward speedup vs adaptive: 20.830x
strong-decay forward speedup vs adaptive: 162.541x
strong-decay forward+backward benchmark speedup: 135.441x
```

Important correction: the original strong-decay backward benchmark timed `.backward()` but did not inspect strong-decay gradients for finiteness/parity.

### First integrated layer probe

The first Small-LLM layer probe passed, but it used `model.half()` and therefore did not reproduce the real trainer precision contract.

### First 500M resume attempt

The verified step-4000 checkpoint restored correctly, but the first resumed update failed before completion because the real trainer AMP path delivered mixed q/k versus v/write dtypes to FLA. Triton rejected an FP32/FP16 dot product.

The adapter was corrected to canonicalize ordinary FLA compute tensors to the low-precision value dtype while keeping log-decay and recurrent state FP32.

### Trainer-realistic AMP layer test

With FP32 master parameters plus CUDA FP16 autocast:

```text
normal decay:
  forward: pass
  gradients: pass

forced g=-6:
  forward: pass
  backward: fail with non-finite/incorrect gradients
```

Testing `disable_recompute=True` also failed, ruling out recomputation as the sole cause.

### Forced-decay AMP sweep

User-reported full-layer results:

```text
passing: [-0.25, -0.5]
failing: [-0.75, -1.0, -1.25, -1.5, -2.0, -3.0, -4.0, -5.0, -6.0]
first failing tested point: g=-0.75
64-token cumulative magnitude at constant g=-0.75: 48.0
```

The exact boundary lies somewhere between tested `g=-0.5` and `g=-0.75`; `-0.75` is the first tested failing point, not an exact threshold.

### Real step-4000 decay telemetry — decisive overlap

The user then ran the forward-only real-data telemetry probe on the verified step-4000 checkpoint and the next training block.

Reported summary:

```text
any_individual_g_le_minus_0.75: True
any_64tok_mean_g_le_minus_0.75: True
real checkpoint overlaps the tested FLA failure region; do not resume chunk-GDN2 training
```

This closes the previous uncertainty. The real trained model is already operating in the same decay regime where FLA v0.5.1 chunk backward fails under the trainer AMP contract.

Detailed evidence: [`../evidence/gdn2_step4000_real_decay_overlap_2026-08-08.md`](../evidence/gdn2_step4000_real_decay_overlap_2026-08-08.md)

## Historical checkpoint compatibility

The active checkpoint stores:

```text
gdn_chunk_size = 32
```

Preserve that model configuration for strict restore. Do not rewrite it to 64.

## Current production boundary

- **Do not resume the active 500M trajectory with FLA v0.5.1 `chunk_gdn2`.**
- The verified step-4000 checkpoint remains clean and usable.
- The old adaptive PyTorch backend remains the correctness/reference fallback and can be used to continue the trajectory if immediate progress is more important than throughput.
- Before changing learned decay semantics, qualify an exact-recurrence optimized alternative. FLA's fused/recurrent GDN-2 path is a candidate to investigate, but is not yet qualified here.
- Do not clip/bound learned GDN-2 decay merely to make the current FLA chunk kernel pass.
- A fresh FLA chunk run from update 1 is also not justified while the same backward numerical issue remains.

## Upstream context

As of 2026-08-08, FLA v0.5.1 is the latest released version previously verified in this project. A later upstream PR explored a bounded/safe GDN-2 gate after extreme learned gate states produced non-finite forward/backward behavior, but that PR was not merged into the release used here. This is supporting context, not authorization to modify Small-LLM's learned decay.

## Evidence

- [`../evidence/gdn2_fla_t4_full_probe_2026-08-08.md`](../evidence/gdn2_fla_t4_full_probe_2026-08-08.md)
- [`../evidence/gdn2_fla_layer_integration_2026-08-08.md`](../evidence/gdn2_fla_layer_integration_2026-08-08.md)
- [`../evidence/gdn2_fla_500m_resume_amp_dtype_failure_2026-08-08.md`](../evidence/gdn2_fla_500m_resume_amp_dtype_failure_2026-08-08.md)
- [`../evidence/gdn2_fla_strong_decay_amp_retained_failure_2026-08-08.md`](../evidence/gdn2_fla_strong_decay_amp_retained_failure_2026-08-08.md)
- [`../evidence/gdn2_fla_amp_decay_sweep_2026-08-08.md`](../evidence/gdn2_fla_amp_decay_sweep_2026-08-08.md)
- [`../evidence/gdn2_step4000_real_decay_overlap_2026-08-08.md`](../evidence/gdn2_step4000_real_decay_overlap_2026-08-08.md)

## Decisions

- [`../decisions/0018-integrate-fla-gdn2-as-checkpoint-compatible-cuda-backend.md`](../decisions/0018-integrate-fla-gdn2-as-checkpoint-compatible-cuda-backend.md)
- [`../decisions/0019-resume-500m-checkpoint-with-fla-gdn2-execution.md`](../decisions/0019-resume-500m-checkpoint-with-fla-gdn2-execution.md) — authorization is currently blocked by failed qualification; no resumed update has been accepted.
