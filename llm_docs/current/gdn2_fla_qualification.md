---
status: current
last_reviewed: 2026-08-08
---

# Current GDN-2 backend qualification status

## Bottom line

The active 20M/500M trajectory must **not** resume update 4001 with the currently integrated FLA v0.5.1 `chunk_gdn2` backend.

The real verified `step-00004000` checkpoint produces decay values inside the same regime where v0.5.1 chunk backward fails under the real trainer precision contract.

The latest accepted trajectory point remains:

```text
checkpoint: step-00004000
last_consumed_block_id: 3999
next update: 4001
```

No FLA experiment has committed update 4001.

## Why FLA was investigated

The completed approximately-20M / 100M run slowed from roughly 3,830 target tok/s early to roughly 445 target tok/s late while validation kept improving. Data wait stayed tiny. Controlled tests strongly support the diagnosis that stronger learned GDN-2 decay made the correctness-first adaptive PyTorch backend repeatedly subdivide chunks and synchronize with Python.

FLA remained attractive because isolated T4 tests showed very large forward and forward+backward throughput advantages, especially under strong decay.

## v0.5.1 qualification history

### Standalone operator

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

The original strong-decay backward benchmark timed `.backward()` but did not inspect those gradients for finiteness/parity.

### First integrated layer probe

The first Small-LLM layer probe passed, but it used `model.half()` and therefore did not reproduce the trainer's FP32-master-parameter + FP16-autocast contract.

### First 500M resume attempt

The verified step-4000 checkpoint restored correctly, but update 4001 failed before completion because q/k and v/write reached an FLA Triton dot with incompatible FP32/FP16 dtypes. The adapter was corrected to canonicalize ordinary FLA compute tensors to the low-precision value dtype while keeping log-decay and recurrent state FP32.

### Trainer-realistic AMP layer test

```text
normal decay:
  forward: pass
  gradients: pass

forced g=-6:
  forward: pass
  backward: fail with non-finite/incorrect gradients
```

`disable_recompute=True` also failed, so backward recomputation was not the sole cause.

### Forced-decay AMP sweep

User-reported v0.5.1 full-layer results:

```text
passing: [-0.25, -0.5]
failing: [-0.75, -1.0, -1.25, -1.5, -2.0, -3.0, -4.0, -5.0, -6.0]
first failing tested point: g=-0.75
64-token cumulative magnitude at constant g=-0.75: 48.0
```

The exact boundary lies somewhere between tested `g=-0.5` and `g=-0.75`; `-0.75` is the first tested failing point, not an exact threshold.

### Real step-4000 decay telemetry — decisive overlap for v0.5.1

Reported summary:

```text
any_individual_g_le_minus_0.75: True
any_64tok_mean_g_le_minus_0.75: True
real checkpoint overlaps the tested FLA failure region; do not resume chunk-GDN2 training
```

Therefore the real trained model is already operating in the tested decay region where FLA v0.5.1 chunk backward fails.

Evidence: [`../evidence/gdn2_step4000_real_decay_overlap_2026-08-08.md`](../evidence/gdn2_step4000_real_decay_overlap_2026-08-08.md)

## Important upstream correction: v0.5.2 exists

The project previously recorded v0.5.1 as the latest release. That is stale. FLA **v0.5.2 was released on 2026-07-27** and is the latest released version as of this review. It is 79 commits ahead of v0.5.1 and includes changes touching GDN/GDN-2/shared chunk infrastructure.

A dedicated Small-LLM probe now exists:

```bash
python kaggle/run_gdn2_fla_052_amp_decay_sweep.py
```

It forces `fla-core==0.5.2` even when a Kaggle session previously installed 0.5.1 and reruns the exact same full-layer FP32-parameters + FP16-autocast decay sweep.

Do not modify the production launcher to v0.5.2 until this correctness gate passes.

## Fused recurrent is not a training fallback

Upstream `fla/ops/gdn2/fused_recurrent.py` explicitly describes `fused_recurrent_gdn2` as the token-by-token **inference-time, forward-only** counterpart of the chunkwise training kernels. It does not track gradients; upstream states that GDN-2 training uses the chunkwise kernel.

Therefore fused recurrent cannot replace `chunk_gdn2` for resumed pretraining.

## FlashQLA does not solve the T4/GDN-2 case

FLA v0.5.2 also added FlashQLA dispatch for the older gated-delta-rule/GDN path. The upstream verifier requires Hopper/SM90 or SM100-class hardware and K=V=128. Our training GPU is Tesla T4 / SM75 and the Small-LLM GDN-2 geometry differs, so this is not the direct fallback for the active run.

## Historical checkpoint compatibility

The active checkpoint stores:

```text
gdn_chunk_size = 32
```

Preserve that model configuration for strict restore. Do not rewrite it to 64.

## Current production boundary

- Do **not** resume the active 500M trajectory with FLA v0.5.1 `chunk_gdn2`.
- First run the v0.5.2 trainer-AMP decay sweep.
- If v0.5.2 still fails around the real checkpoint's decay region, reject released FLA chunk training for this trajectory and choose between continuing with the adaptive reference backend or engineering/qualifying another exact differentiable training kernel.
- If v0.5.2 passes the synthetic sweep through `g=-6`, require a direct real step-4000 forward/backward parity test before production integration.
- Do not clip/bound learned GDN-2 decay merely to make a kernel work.
- The verified step-4000 checkpoint remains clean and usable.

## Evidence

- [`../evidence/gdn2_fla_t4_full_probe_2026-08-08.md`](../evidence/gdn2_fla_t4_full_probe_2026-08-08.md)
- [`../evidence/gdn2_fla_layer_integration_2026-08-08.md`](../evidence/gdn2_fla_layer_integration_2026-08-08.md)
- [`../evidence/gdn2_fla_500m_resume_amp_dtype_failure_2026-08-08.md`](../evidence/gdn2_fla_500m_resume_amp_dtype_failure_2026-08-08.md)
- [`../evidence/gdn2_fla_strong_decay_amp_retained_failure_2026-08-08.md`](../evidence/gdn2_fla_strong_decay_amp_retained_failure_2026-08-08.md)
- [`../evidence/gdn2_fla_amp_decay_sweep_2026-08-08.md`](../evidence/gdn2_fla_amp_decay_sweep_2026-08-08.md)
- [`../evidence/gdn2_step4000_real_decay_overlap_2026-08-08.md`](../evidence/gdn2_step4000_real_decay_overlap_2026-08-08.md)

## Decisions

- [`../decisions/0018-integrate-fla-gdn2-as-checkpoint-compatible-cuda-backend.md`](../decisions/0018-integrate-fla-gdn2-as-checkpoint-compatible-cuda-backend.md)
- [`../decisions/0019-resume-500m-checkpoint-with-fla-gdn2-execution.md`](../decisions/0019-resume-500m-checkpoint-with-fla-gdn2-execution.md) — authorization remains blocked by the failed v0.5.1 qualification gate; no resumed update has been accepted.
