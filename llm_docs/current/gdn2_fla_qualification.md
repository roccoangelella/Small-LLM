---
status: current
last_reviewed: 2026-08-08
---

# Current GDN-2 backend qualification status

## Diagnosis

The completed approximately-20M / 100M run slowed from roughly 3,830 target tok/s early to roughly 445 target tok/s late while validation kept improving. Controlled FLA tests on the same Tesla T4 strongly support the explanation that stronger learned GDN-2 decay exposed pathological chunk subdivision / synchronization in the correctness-first adaptive PyTorch backend rather than a need to clip learned decay.

## Standalone FLA operator qualification — FORWARD/normal-backward passed

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
FLA speedup over adaptive, strong-decay forward+backward benchmark: 135.441x
```

Important correction: the original strong-decay forward+backward benchmark only completed/timed `.backward()`; it did not inspect strong-decay gradients for finiteness/parity. Therefore it did not qualify strong-decay backward correctness.

## First full-layer integration probe — PASSED, but precision coverage was incomplete

The first integrated layer probe reported:

```text
layer_forward_backward_parity: True
checkpoint_parity: None
INTEGRATION QUALIFIED for checkpoint evaluation; fresh-training authorization remains separate.
```

However, that probe converted the whole candidate/reference layer to FP16 with `model.half()`. It did not reproduce the real trainer precision contract of FP32 master parameters plus CUDA FP16 autocast.

## First 500M FLA resume attempt — FAILED CLOSED BEFORE UPDATE 4001

The launcher successfully restored the verified step-4000 checkpoint and attempted global steps 4001–15264. The first resumed update did not complete. Triton compilation failed inside FLA WY recomputation:

```text
Both operands must be same dtype. Got fp32 and fp16
b_u = tl.dot(b_A, b_vb)
```

No successful update 4001 was produced, so the latest verified checkpoint remains step 4000. Model weights, optimizer state, scheduler/WSD position, scaler, RNG state, and data cursor remain intact at that checkpoint.

Root cause: under the real trainer's FP32-master + FP16-autocast path, normalized q/k can enter the FLA adapter as FP32 while v/write are FP16. FLA v0.5.1 allocates its solved WY matrix with `k.dtype`, then dots it with a v/write block. Triton requires both dot operands to use the same dtype.

Detailed evidence: [`../evidence/gdn2_fla_500m_resume_amp_dtype_failure_2026-08-08.md`](../evidence/gdn2_fla_500m_resume_amp_dtype_failure_2026-08-08.md)

## AMP-safe adapter fix — normal decay passes, strong decay fails backward

`model/gdn2_fla.py` canonicalizes the ordinary FLA compute tensors to the low-precision value dtype while keeping log-decay and recurrent state FP32. The revised integration probe keeps model parameters in FP32 and uses CUDA FP16 autocast, matching the actual trainer.

Results reported by the user:

```text
normal_decay_amp:
  layer output: PASS
  all tested input/parameter gradients: PASS

strong_decay_-6_amp:
  layer output: PASS
  backward parity: FAIL
  NaN/non-finite gradients observed
```

Thus the dtype-contract bug is fixed, but FLA v0.5.1 chunk backward remains unqualified under the forced strong-decay AMP stress case.

## Retained-intermediate test — FAILS too

A focused probe then tested FLA with `disable_recompute=True`, retaining the forward WY/state intermediates instead of reconstructing them during backward.

User-reported verdict:

```text
VERDICT: FAIL — retained-intermediate FLA is still not qualified for resumed training.
```

This rules out backward recomputation as the sole cause. The strong-decay AMP failure is deeper in the v0.5.1 chunk-backward numerical path.

Detailed evidence: [`../evidence/gdn2_fla_strong_decay_amp_retained_failure_2026-08-08.md`](../evidence/gdn2_fla_strong_decay_amp_retained_failure_2026-08-08.md)

## Trainer-AMP decay sweep — first tested failure at g=-0.75

The user ran a full-layer forced-decay sweep under FP32 parameters + CUDA FP16 autocast.

```text
passing: [-0.25, -0.5]
failing: [-0.75, -1.0, -1.25, -1.5, -2.0, -3.0, -4.0, -5.0, -6.0]
first failing tested point: g=-0.75
64-token cumulative magnitude at constant g=-0.75: 48.0
```

This is much closer to the learned-decay regime that could plausibly have caused the adaptive backend slowdown than the original `g=-6` stress point. FLA v0.5.1 chunk backward therefore remains blocked until the actual step-4000 checkpoint is measured on real training data.

Detailed evidence: [`../evidence/gdn2_fla_amp_decay_sweep_2026-08-08.md`](../evidence/gdn2_fla_amp_decay_sweep_2026-08-08.md)

A forward-only probe now exists at `kaggle/run_gdn2_step4000_decay_telemetry.py`. It loads the real checkpoint, reads train block 4000 from the attached 500M manifest/shard, uses the first four 2048-token sequences (matching one training microbatch), records each GDN layer's log-decay tensor, and computes per-token quantiles plus 64-token cumulative-magnitude/mean statistics. It uses FLA only for the already-qualified forward path and does no backward/optimizer/checkpoint mutation.

## Historical chunk-32 checkpoint vs FLA64 runtime

The active 20M/500M checkpoint stores:

```text
gdn_chunk_size = 32
```

That configuration remains unchanged for strict checkpoint restore. CUDA FLA chunk execution uses fixed internal chunk size 64; CPU/reference execution keeps historical adaptive chunk 32. This execution grouping does not add learned state.

## Current production boundary — FLA resume BLOCKED

Do not retry the 500M resume with FLA chunk training yet.

The valid trajectory remains safely resumable from verified step 4000. The FLA experiments have not committed update 4001.

Next gate:

1. run `kaggle/run_gdn2_step4000_decay_telemetry.py` on the restored step-4000 checkpoint and attached 500M dataset;
2. compare actual per-layer 64-token mean/cumulative decay against the synthetic passing `g=-0.5` and first-failing `g=-0.75` points;
3. if real data overlaps the failing region, reject FLA v0.5.1 chunk backward for this trajectory and qualify an exact-recurrence alternative (for example FLA fused recurrent) before considering any decay bound;
4. if real data stays clearly outside the failing region, still require a direct real-checkpoint forward/backward finite-gradient test before resuming training.

## Upstream context

As of 2026-08-08, FLA v0.5.1 is still the latest release. A later upstream PR (`#1007`, closed without merge) explored an opt-in bounded/safe GDN-2 gate path after a training failure involving extreme learned gate state. It reported numerical failures in the default extreme-gate regime and finite behavior under the bounded path. This is relevant evidence that strong-decay robustness is a real upstream concern, but it does not authorize Small-LLM to bound/clamp decay.

## Current decision boundary

- Do **not** clip/bound learned GDN-2 decay solely because the old adaptive backend slows down.
- FLA v0.5.1 forward remains strongly qualified and normal-decay AMP backward parity passes.
- FLA v0.5.1 AMP chunk backward passes tested constant `g=-0.5` but fails by tested constant `g=-0.75`.
- `disable_recompute=True` does not fix the strong-decay backward failure.
- Do **not** resume the active 500M trajectory with FLA chunk training until real step-4000 decay telemetry and a subsequent real-checkpoint backward gate are evaluated.
- The latest verified usable training checkpoint remains step 4000.
- A fresh FLA-from-update-1 run remains unauthorized while the same backward issue is unresolved.

Evidence:
- [`../evidence/gdn2_fla_t4_full_probe_2026-08-08.md`](../evidence/gdn2_fla_t4_full_probe_2026-08-08.md)
- [`../evidence/gdn2_fla_layer_integration_2026-08-08.md`](../evidence/gdn2_fla_layer_integration_2026-08-08.md)
- [`../evidence/gdn2_fla_500m_resume_amp_dtype_failure_2026-08-08.md`](../evidence/gdn2_fla_500m_resume_amp_dtype_failure_2026-08-08.md)
- [`../evidence/gdn2_fla_strong_decay_amp_retained_failure_2026-08-08.md`](../evidence/gdn2_fla_strong_decay_amp_retained_failure_2026-08-08.md)
- [`../evidence/gdn2_fla_amp_decay_sweep_2026-08-08.md`](../evidence/gdn2_fla_amp_decay_sweep_2026-08-08.md)

Decisions:
- [`../decisions/0018-integrate-fla-gdn2-as-checkpoint-compatible-cuda-backend.md`](../decisions/0018-integrate-fla-gdn2-as-checkpoint-compatible-cuda-backend.md)
- [`../decisions/0019-resume-500m-checkpoint-with-fla-gdn2-execution.md`](../decisions/0019-resume-500m-checkpoint-with-fla-gdn2-execution.md) — currently blocked by failed qualification gate; no resumed update has been accepted.
