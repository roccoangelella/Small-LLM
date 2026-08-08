---
status: current
last_reviewed: 2026-08-08
purpose: long-chat handoff / project memory
---

# GDN-2 / FLA investigation handoff — 2026-08-08

This file is the consolidated handoff for the long investigation into the 20M/500M run's GDN-2 slowdown and attempted FLA migration. A new chat should read this file together with `current/status.md` and `current/gdn2_fla_qualification.md` before proposing any further production change.

## 1. Active experiment state

The active experiment is the approximately-20M-parameter GDN-2 hybrid on the fixed approximately-500M-token dataset.

```text
profile: 20m-500m-data-scaling-v1
dataset run ID: 20m-500m-dataset-001
W&B run ID: 20m-500m-data-001
context length: 2048
microbatch: 4
architecture: gdn2_hybrid
layer pattern: gdn, gdn, gdn, mha, repeated across 8 layers
saved/configured gdn_chunk_size: 32
GPU: Tesla T4 / SM75
precision: FP16 trainer autocast with FP32 master parameters
```

Latest verified accepted trajectory point:

```text
checkpoint: step-00004000
last_consumed_block_id: 3999
next update: 4001
```

The checkpoint is clean. No FLA experiment has successfully committed update 4001. Model weights, optimizer state, WSD scheduler position, scaler, RNG state, data cursor and consumed-token cursor remain valid at step 4000.

**Do not run the ordinary 500M launcher for production continuation until the backend decision is changed/fixed.** The launcher was previously repinned for FLA integration, but released FLA chunk training is now unqualified for this checkpoint.

## 2. Why the backend investigation started

The completed 20M/100M pretraining run stayed numerically trainable and validation improved, but throughput collapsed severely:

```text
early: ~3830 target tok/s
late:  ~445 target tok/s
slowdown: ~8.6x
```

Validation slowed by almost the same factor. Data wait was tiny:

```text
median data wait: ~4.23 ms
p95 data wait: ~12.25 ms
peak reserved VRAM: ~9.127 GiB
```

This pointed to model forward/backward compute rather than dataset I/O.

The adaptive PyTorch GDN-2 backend was the leading suspect. It computes cumulative log-decay span, performs a GPU-to-CPU `.item()` synchronization, and repeatedly halves chunks when the span becomes too large. It can split configured chunks down through 32,16,8,4,2,1. Its chunkwise reference path also performs FP32 cumulative decay, dense matmuls, triangular solves and other expensive operations.

As learned decay becomes more negative, this correctness-first backend becomes computationally pathological.

## 3. GDN-2 recurrence / decay meaning

Per head, the recurrence is conceptually:

```text
e_t = b_t ⊙ k_t
z_t = w_t ⊙ v_t
S_bar_t = Diag(exp(g_t)) S_(t-1)
S_t = S_bar_t + k_t (z_t - S_bar_t^T e_t)^T
o_t = S_t^T q_t / sqrt(d_k)
```

`g_t <= 0` is log-decay and `alpha_t = exp(g_t)` is the survival fraction of memory. More-negative `g` means faster memory decay.

Small-LLM computes decay through learned parameters approximately as:

```text
g = -exp(A_log) * softplus(decay_features + dt_bias)
```

so there is no intrinsic lower bound on `g`.

## 4. Initial FLA candidate and standalone qualification

Candidate originally selected:

```text
fla-org/flash-linear-attention
fla-core 0.5.1
```

Standalone Tesla T4 tests showed very large performance gains.

Representative results:

```text
normal forward:
  adaptive ~177,861 tok/s
  FLA      ~3,704,843 tok/s
  speedup  20.83x

normal forward+backward:
  adaptive ~58,623 tok/s
  FLA      ~1,062,705 tok/s
  speedup  ~18.13x

forced strong g=-6 forward:
  adaptive ~15,296 tok/s
  FLA      ~2,486,218 tok/s
  speedup  ~162.54x

forced strong g=-6 forward+backward benchmark:
  adaptive ~6,050 tok/s
  FLA      ~819,392 tok/s
  speedup  ~135.44x
```

Forward parity passed at normal, `g=-6` and `g=-10`. Normal-decay backward gradient parity passed.

Important later correction: the original strong-decay forward+backward benchmark only timed `.backward()`; it did not inspect strong-decay gradients for finiteness/parity. Therefore it never qualified strong-decay backward correctness.

The standalone evidence strongly supported the original throughput diagnosis: FLA retained much more throughput under strong decay while the adaptive backend collapsed.

## 5. First Small-LLM FLA integration

A checkpoint-compatible adapter was added so CUDA would prefer FLA while CPU/reference remained adaptive.

Historical checkpoints store:

```text
gdn_chunk_size = 32
```

FLA `chunk_gdn2` internally requires/favors fixed 64-token execution. To preserve strict checkpoint model-config identity, Small-LLM deliberately kept the saved/configured value at 32 and made the CUDA FLA adapter use 64 internally.

Thus the intended migration was:

```text
saved/model config:              gdn_chunk_size = 32
CPU/reference adaptive backend:  configured chunk 32
CUDA FLA execution:              internal chunk 64
```

No learned parameters or state-dict keys were added by FLA.

The dependency was also added to the `model` optional runtime extra because production training uses `uv run --extra model ...`.

## 6. First integrated layer probe — misleadingly passed

The first full Small-LLM GDN layer probe reported:

```text
layer_forward_backward_parity: True
checkpoint_parity: None
INTEGRATION QUALIFIED for checkpoint evaluation
```

But the test had a coverage flaw: it converted the whole model/layer with `model.half()`.

The real trainer uses FP32 master parameters with CUDA FP16 autocast. Therefore the first test did not reproduce the real trainer dtype contract.

This result is preserved as historical evidence but must **not** be treated as sufficient production qualification.

## 7. First real 500M FLA resume attempt — dtype compilation failure

The normal 500M launcher correctly restored the verified step-4000 checkpoint and attempted global steps 4001 onward.

Before update 4001 completed, Triton compilation failed inside FLA:

```text
AssertionError: Both operands must be same dtype. Got fp32 and fp16
...
b_u = tl.dot(b_A, b_vb)
```

Root cause: under trainer AMP, normalized q/k could enter the FLA adapter in FP32 while v/write were FP16. FLA's WY path created one matrix with `k.dtype` and later dotted it with a value/write tensor in another dtype. Triton rejected `fp32 x fp16`.

The failure occurred before a successful update 4001. Therefore step 4000 stayed intact.

## 8. AMP dtype adapter fix

The Small-LLM FLA adapter was changed so ordinary compute tensors entering FLA are canonicalized to the same low-precision dtype while log-decay and recurrent state stay FP32.

Intended runtime contract after the fix:

```text
q        -> FP16
k        -> FP16
v        -> FP16
erase    -> FP16
write    -> FP16
log_decay -> FP32
state     -> FP32
```

The revised layer qualification test was also changed to use:

```text
FP32 model parameters + CUDA FP16 autocast
```

instead of `model.half()`.

## 9. Trainer-realistic AMP layer probe — normal pass, strong-decay backward fail

Under the corrected trainer-realistic contract:

```text
normal_decay_amp:
  output: PASS
  input gradient: PASS
  all tested parameter gradients: PASS

strong_decay_-6_amp:
  output: PASS
  backward: FAIL
  non-finite/NaN gradients observed
```

This established that the dtype mismatch was fixed, but FLA v0.5.1 strong-decay chunk backward was numerically unsafe.

## 10. `disable_recompute=True` hypothesis — ruled out

The standalone qualification had used `disable_recompute=True`, while the integration initially used the default recompute behavior. A focused probe therefore retained forward intermediates for backward.

User-reported result:

```text
VERDICT: FAIL — retained-intermediate FLA is still not qualified for resumed training.
```

Therefore backward recomputation was not the sole cause. The failure is deeper in the v0.5.1 chunk-backward numerical path.

## 11. v0.5.1 trainer-AMP decay sweep

A forced constant-decay sweep used the exact trainer contract and compared the entire Small-LLM GDN layer against the adaptive reference.

User-reported summary:

```text
passing: [-0.25, -0.5]
failing: [-0.75, -1.0, -1.25, -1.5, -2.0, -3.0, -4.0, -5.0, -6.0]
first failing tested point: g=-0.75
64-token cumulative magnitude at constant g=-0.75: 48.0
```

`-0.75` is the first tested failing point, not a mathematically exact threshold. The important finding was that failure occurred at a much milder decay than the original `g=-6` stress case.

## 12. Real step-4000 decay telemetry — decisive overlap

A forward-only diagnostic restored the verified remote step-4000 checkpoint, read the real next training block (`block 4000`), used a real 4x2048 microbatch, and captured every GDN layer's actual `log_decay` values and 64-token means/spans.

Restoration was verified as:

```text
checkpoint_id: step-00004000
last_consumed_block_id: 3999
```

User-reported telemetry summary:

```text
any_individual_g_le_minus_0.75: True
any_64tok_mean_g_le_minus_0.75: True
real checkpoint overlaps the tested FLA failure region; do not resume chunk-GDN2 training
```

This was decisive for v0.5.1: the real model is already operating inside the same tested regime where trainer-AMP chunk backward fails.

## 13. Upstream/release correction: FLA v0.5.2

During investigation it was discovered that project memory was stale: FLA v0.5.2 had been released on 2026-07-27 and was newer than v0.5.1.

The v0.5.1 -> v0.5.2 comparison contains 79 commits and touches GDN/GDN-2/shared chunk infrastructure. Therefore one bounded v0.5.2 qualification sweep was warranted before rejecting released FLA entirely.

## 14. v0.5.2 trainer-AMP decay sweep — still fails

User-reported summary:

```text
fla_core_version: 0.5.2
passing: [-0.25, -0.5, -1.0]
failing: [-0.75, -1.25, -1.5, -2.0, -3.0, -4.0, -5.0, -6.0]
first failing tested point: g=-0.75
64-token cumulative magnitude at constant g=-0.75: 48.0
VERDICT: v0.5.2 still has a tested trainer-AMP backward failure; do not resume training yet.
```

Important nuance: this pattern is **non-monotonic** because `g=-1.0` passed while `g=-0.75` and `g=-1.25` failed. Therefore do not describe this as a simple magnitude threshold such as "all g below -0.75 fail." It is evidence of a numerical/kernel instability pattern under particular decay states.

Production implication is nevertheless clear: v0.5.2 still has backward failures in a decay regime that overlaps the real step-4000 model, so upgrading to v0.5.2 does not qualify released FLA chunk training for the active trajectory.

## 15. FLA alternatives that were checked

### `fused_recurrent_gdn2`

Upstream explicitly describes this as inference-time / forward-only. It does not track gradients and upstream says training uses the chunkwise kernel.

Therefore it cannot be a pretraining fallback.

### FlashQLA

FLA v0.5.2 includes FlashQLA dispatch for the older gated-delta-rule/GDN path. The upstream verifier requires Hopper/SM90 or SM100-class hardware and K=V=128 for that backend.

The active GPU is Tesla T4 / SM75, and the active architecture is GDN-2 with different geometry. Therefore FlashQLA is not a direct solution for this run.

## 16. Current scientific conclusions

### Strongly supported

1. The historical adaptive-backend slowdown is caused primarily by backend execution pathology exposed by stronger learned decay, not by data I/O.
2. FLA's chunk forward implementation is dramatically faster and forward-correct on the T4.
3. Normal-decay trainer-AMP backward can match the adaptive reference.
4. Released FLA v0.5.1 and v0.5.2 chunk GDN-2 training both have decay-dependent trainer-AMP backward failures relevant to the active checkpoint.
5. The real verified step-4000 checkpoint produces decay regions overlapping tested FLA failure cases.
6. The active FLA migration must therefore remain blocked.

### Not established / must not be overstated

1. There is no proven single monotonic `g` failure threshold; v0.5.2's pass at `-1.0` demonstrates non-monotonic behavior in the tested points.
2. We have not proven that every historical slowdown component comes from the adaptive decay subdivision, although evidence is very strong.
3. We have not qualified another fast differentiable GDN-2 backend yet.
4. We have not authorized clipping/bounding learned decay.
5. We have not authorized a fresh FLA-from-update-1 500M run.

## 17. Current production boundary

**DO NOT:**

- do not resume `step-00004000` with FLA v0.5.1 chunk training;
- do not resume it with FLA v0.5.2 chunk training;
- do not assume the old first layer-probe `True` result supersedes the later AMP failures;
- do not use fused recurrent as a training backend;
- do not use FlashQLA as a T4/GDN-2 fallback;
- do not change checkpoint `gdn_chunk_size=32` to 64;
- do not clip/bound learned decay merely to satisfy the current kernel;
- do not claim update 4001 ever completed under FLA.

**SAFE FACTS:**

```text
latest accepted checkpoint: step-00004000
last consumed block: 3999
next update: 4001
checkpoint is clean
no FLA update accepted
```

## 18. Next decision / engineering options

The next discussion should choose between these branches rather than launching more full 500M jobs blindly:

### Option A — Resume with the old adaptive PyTorch backend

Pros:
- already correctness-qualified;
- exact existing model semantics;
- simplest way to continue the trajectory.

Cons:
- late-run throughput may again be extremely poor as learned decay strengthens;
- defeats the purpose of the backend optimization investigation.

A production launcher change would be required because the current launcher had been repinned for FLA integration.

### Option B — Engineer/qualify another exact differentiable GDN-2 training kernel

This is the preferred technical direction if practical. Requirements:

- exact recurrence semantics / no learned-state change;
- supports Tesla T4 / SM75;
- differentiable training path;
- stable under the real step-4000 decay distribution;
- trainer-realistic FP32-master + FP16-autocast tests;
- strong-decay gradient finiteness/parity;
- direct real step-4000 forward/backward dry run before production resume;
- benchmark steady-state throughput after compilation/autotuning.

Possible avenues need fresh research; do not assume FLA fused recurrent or FlashQLA satisfy these requirements.

### Option C — Change the model's decay parameterization / bound decay

This would alter optimization/model semantics and should be treated as a new architecture/training decision, not a runtime implementation fix. It is currently **not authorized** for the existing trajectory.

## 19. Important implementation/history commits

Key Small-LLM commits from this investigation include:

```text
27bc5d5  initial model/gdn2_fla.py integration
5c1a0c5  preserve saved chunk32 while CUDA FLA executes chunk64
f234ff7  put fla-core in model runtime extra
bdd6ba6  structural tests for configured32/runtime64
a147147  integration probe/checkpoint-compat migration implementation
78cc22d  repin 500M launcher to FLA migration implementation
efa3d10  AMP dtype-fix implementation pinned by launcher later
5afc7ea  initial integrated-layer evidence
4410ba3  ADR 0019 resume existing 500M checkpoint with FLA
0fc08d2  add trainer-AMP decay sweep
06e9d1f  add remote step-4000 telemetry wrapper
0945c8d  fix telemetry restore path
92b266d  add FLA v0.5.2 AMP decay sweep
11fdb14  record v0.5.2 decay-sweep failure evidence
```

Exact current commit history can be re-read from GitHub if needed; this list is a navigation aid, not a replacement for Git history.

## 20. Evidence files

Relevant durable evidence under `llm_docs/evidence/`:

```text
gdn2_fla_t4_full_probe_2026-08-08.md
gdn2_fla_layer_integration_2026-08-08.md
gdn2_fla_500m_resume_amp_dtype_failure_2026-08-08.md
gdn2_fla_strong_decay_amp_retained_failure_2026-08-08.md
gdn2_fla_amp_decay_sweep_2026-08-08.md
gdn2_step4000_real_decay_overlap_2026-08-08.md
gdn2_fla_052_amp_decay_sweep_2026-08-08.md
```

Current source-of-truth summary files:

```text
llm_docs/current/status.md
llm_docs/current/gdn2_fla_qualification.md
llm_docs/current/gdn2_fla_investigation_handoff.md   <- this file
```

## 21. Durable decisions still in force

- Preserve the historical checkpoint/model config `gdn_chunk_size=32`.
- Keep the adaptive PyTorch backend as the correctness/reference implementation.
- Do not clip/bound learned decay solely because an implementation slows down or fails.
- The 500M trajectory remains the independent seed-17 run with microbatch 4.
- Validation/local checkpoint/verified remote publication cadence remains every 250 successful updates.
- FP16 loss scaling may calibrate down to scale 1.0 before failing an otherwise atomic block.
- Fresh FLA-from-update-1 training remains a separate decision and is not authorized while released FLA backward is unqualified.
- ADR 0019's prior authorization to resume with FLA is operationally blocked by later failed qualification evidence; no FLA-resumed update was accepted.

## 22. Recommended first action in the next chat

Before any new code change, inspect:

```text
llm_docs/current/status.md
llm_docs/current/gdn2_fla_qualification.md
llm_docs/current/gdn2_fla_investigation_handoff.md
```

Then decide whether to:

1. restore the old adaptive production backend and resume the experiment despite expected slowness, or
2. research/implement a different exact differentiable GDN-2 training kernel suitable for SM75.

Do not spend another Kaggle launch on FLA v0.5.1/v0.5.2 chunk training without materially new evidence or an upstream numerical fix.
