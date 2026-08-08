---
status: current
last_reviewed: 2026-08-08
---

# Current GDN-2 backend qualification status

## Bottom line

The active 20M/500M trajectory must **not** resume update 4001 with released FLA `chunk_gdn2` training.

Both tested released versions, v0.5.1 and v0.5.2, have trainer-AMP backward failures in decay regimes that overlap the real verified step-4000 checkpoint.

Latest accepted state:

```text
checkpoint: step-00004000
last_consumed_block_id: 3999
next update: 4001
```

No FLA experiment has committed update 4001.

For full chronology, implementation details, important commits, evidence list, and next engineering branches, read [`gdn2_fla_investigation_handoff.md`](gdn2_fla_investigation_handoff.md).

## Why FLA was investigated

The completed approximately-20M / 100M run slowed from roughly 3,830 target tok/s early to roughly 445 target tok/s late while validation kept improving and data wait remained tiny. Controlled tests strongly support the diagnosis that stronger learned GDN-2 decay made the correctness-first adaptive PyTorch backend repeatedly subdivide chunks and synchronize with Python.

Standalone FLA T4 tests were extremely fast and forward-correct, so FLA was investigated as a checkpoint-compatible execution replacement rather than clipping learned decay.

## Historical checkpoint compatibility

The active checkpoint stores:

```text
gdn_chunk_size = 32
```

Preserve that model configuration for strict restore. The attempted CUDA FLA adapter kept saved/configured chunk 32 but used FLA's fixed internal chunk 64. This execution grouping added no learned parameters or state-dict keys.

## Initial standalone v0.5.1 results

```text
Tesla T4 / SM75
PyTorch 2.10.0+cu128
CUDA 12.8
Triton 3.6.0
fla-core 0.5.1
```

Representative evidence:

```text
forward correctness: pass
normal-decay backward gradient parity: pass
normal forward speedup vs adaptive: 20.830x
strong-decay forward speedup vs adaptive: 162.541x
strong-decay forward+backward benchmark speedup: 135.441x
```

Important correction: the original strong-decay forward+backward benchmark timed `.backward()` but did not inspect the strong-decay gradients for finiteness/parity.

## First integrated layer probe — insufficient precision coverage

The first Small-LLM layer probe reported:

```text
layer_forward_backward_parity: True
checkpoint_parity: None
INTEGRATION QUALIFIED for checkpoint evaluation
```

But it used `model.half()`, not the real trainer contract of FP32 master parameters + CUDA FP16 autocast. It therefore missed a real mixed-dtype path and is not sufficient production qualification.

## First 500M FLA resume attempt — failed before update 4001

The launcher restored verified step 4000 correctly, but Triton compilation failed before update 4001 completed:

```text
Both operands must be same dtype. Got fp32 and fp16
b_u = tl.dot(b_A, b_vb)
```

Under trainer AMP, q/k could reach FLA as FP32 while v/write were FP16. The Small-LLM adapter was changed to canonicalize ordinary FLA compute tensors to the same low-precision dtype while keeping log-decay and recurrent state FP32.

The checkpoint remained untouched because no update 4001 completed.

Evidence: [`../evidence/gdn2_fla_500m_resume_amp_dtype_failure_2026-08-08.md`](../evidence/gdn2_fla_500m_resume_amp_dtype_failure_2026-08-08.md)

## Trainer-realistic AMP layer result

After the dtype fix, the revised probe used FP32 parameters + FP16 autocast.

```text
normal decay:
  forward: pass
  tested gradients: pass

forced g=-6:
  forward: pass
  backward: fail with non-finite/incorrect gradients
```

A retained-intermediate test with `disable_recompute=True` failed too, ruling out backward recomputation as the sole cause.

Evidence: [`../evidence/gdn2_fla_strong_decay_amp_retained_failure_2026-08-08.md`](../evidence/gdn2_fla_strong_decay_amp_retained_failure_2026-08-08.md)

## v0.5.1 trainer-AMP forced-decay sweep

User-reported full-layer results:

```text
passing: [-0.25, -0.5]
failing: [-0.75, -1.0, -1.25, -1.5, -2.0, -3.0, -4.0, -5.0, -6.0]
first failing tested point: g=-0.75
64-token cumulative magnitude at constant g=-0.75: 48.0
```

`g=-0.75` is the first tested failing point, not a proven exact threshold.

Evidence: [`../evidence/gdn2_fla_amp_decay_sweep_2026-08-08.md`](../evidence/gdn2_fla_amp_decay_sweep_2026-08-08.md)

## Real step-4000 telemetry — actual model overlaps tested failure region

The diagnostic restored the verified remote checkpoint, required `step-00004000` with last consumed block `3999`, read real training block 4000, and measured actual GDN log-decay on a real 4x2048 microbatch.

User-reported summary:

```text
any_individual_g_le_minus_0.75: True
any_64tok_mean_g_le_minus_0.75: True
real checkpoint overlaps the tested FLA failure region; do not resume chunk-GDN2 training
```

This made the v0.5.1 production rejection decisive.

Evidence: [`../evidence/gdn2_step4000_real_decay_overlap_2026-08-08.md`](../evidence/gdn2_step4000_real_decay_overlap_2026-08-08.md)

## FLA v0.5.2 — also fails trainer-AMP backward

Project memory initially missed that v0.5.2 had been released on 2026-07-27. It is 79 commits ahead of v0.5.1 and includes changes touching GDN/GDN-2/shared chunk infrastructure, so the exact same trainer-AMP sweep was rerun with `fla-core==0.5.2` forced.

User-reported result:

```text
fla_core_version: 0.5.2
passing: [-0.25, -0.5, -1.0]
failing: [-0.75, -1.25, -1.5, -2.0, -3.0, -4.0, -5.0, -6.0]
first failing tested point: g=-0.75
64-token cumulative magnitude at constant g=-0.75: 48.0
VERDICT: v0.5.2 still has a tested trainer-AMP backward failure
```

Important nuance: the v0.5.2 pattern is **non-monotonic** because `g=-1.0` passed while `g=-0.75` and `g=-1.25` failed. Do not model the failure as a simple threshold in `|g|`.

The production conclusion is nevertheless unchanged: v0.5.2 still fails in a tested decay regime relevant to the real checkpoint and is not qualified for resumed training.

Evidence: [`../evidence/gdn2_fla_052_amp_decay_sweep_2026-08-08.md`](../evidence/gdn2_fla_052_amp_decay_sweep_2026-08-08.md)

## Checked upstream alternatives

### FLA fused recurrent

`fused_recurrent_gdn2` is upstream's inference-time, forward-only recurrent kernel. It does not track gradients; upstream states that GDN-2 training uses the chunkwise kernel. It cannot serve as a pretraining fallback.

### FlashQLA

FLA v0.5.2 includes FlashQLA dispatch for an older gated-delta-rule/GDN backend, but its upstream verifier requires SM90/SM100-class hardware and K=V=128. The active GPU is Tesla T4 / SM75 and the active model uses GDN-2 geometry, so it is not the solution for this run.

## Current production boundary

- Do **not** resume the active 500M trajectory with FLA v0.5.1 `chunk_gdn2`.
- Do **not** resume it with FLA v0.5.2 `chunk_gdn2`.
- Do **not** rely on the old `layer_forward_backward_parity: True` result; later trainer-realistic AMP tests supersede it.
- Do **not** use fused recurrent as a training backend.
- Do **not** use FlashQLA as a T4/GDN-2 fallback.
- Do **not** change saved `gdn_chunk_size=32` to 64.
- Do **not** clip/bound learned decay merely to satisfy a kernel.
- The verified step-4000 checkpoint remains clean and usable.

The next project decision must choose between:

1. restoring the adaptive PyTorch production backend and resuming from step 4000 despite expected slowness; or
2. engineering/researching and qualifying another exact differentiable GDN-2 training kernel suitable for Tesla T4 / SM75 and the real checkpoint's decay regime.

## Evidence index

- [`../evidence/gdn2_fla_t4_full_probe_2026-08-08.md`](../evidence/gdn2_fla_t4_full_probe_2026-08-08.md)
- [`../evidence/gdn2_fla_layer_integration_2026-08-08.md`](../evidence/gdn2_fla_layer_integration_2026-08-08.md)
- [`../evidence/gdn2_fla_500m_resume_amp_dtype_failure_2026-08-08.md`](../evidence/gdn2_fla_500m_resume_amp_dtype_failure_2026-08-08.md)
- [`../evidence/gdn2_fla_strong_decay_amp_retained_failure_2026-08-08.md`](../evidence/gdn2_fla_strong_decay_amp_retained_failure_2026-08-08.md)
- [`../evidence/gdn2_fla_amp_decay_sweep_2026-08-08.md`](../evidence/gdn2_fla_amp_decay_sweep_2026-08-08.md)
- [`../evidence/gdn2_step4000_real_decay_overlap_2026-08-08.md`](../evidence/gdn2_step4000_real_decay_overlap_2026-08-08.md)
- [`../evidence/gdn2_fla_052_amp_decay_sweep_2026-08-08.md`](../evidence/gdn2_fla_052_amp_decay_sweep_2026-08-08.md)

## Decisions

- [`../decisions/0018-integrate-fla-gdn2-as-checkpoint-compatible-cuda-backend.md`](../decisions/0018-integrate-fla-gdn2-as-checkpoint-compatible-cuda-backend.md)
- [`../decisions/0019-resume-500m-checkpoint-with-fla-gdn2-execution.md`](../decisions/0019-resume-500m-checkpoint-with-fla-gdn2-execution.md) — prior authorization is operationally blocked by later failed qualification evidence; no resumed FLA update has been accepted.
