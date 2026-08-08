---
status: current
last_reviewed: 2026-08-08
purpose: long-chat handoff / project memory
---

# GDN-2 / FLA investigation handoff — final August 8 state

This is the consolidated handoff for the 20M/500M GDN-2 backend investigation. Read it together with `current/status.md`, `current/gdn2_fla_qualification.md`, and ADR 0021 before changing production training.

## 1. Active experiment state

```text
profile: 20m-500m-data-scaling-v1
dataset run ID: 20m-500m-dataset-001
W&B run ID: 20m-500m-data-001
architecture: gdn2_hybrid
layer pattern: gdn, gdn, gdn, mha, repeated across 8 layers
context length: 2048
microbatch: 4
saved/configured gdn_chunk_size: 32
GPU: Tesla T4 / SM75
trainer precision: FP16 autocast with FP32 master parameters
```

Latest accepted trajectory point at the end of this investigation:

```text
checkpoint: step-00004000
last_consumed_block_id: 3999
next update: 4001
```

The checkpoint is clean. No qualification diagnostic executed an optimizer step, acknowledged block 4000, advanced the scheduler/data cursor, wrote W&B state, or published a checkpoint. **No FLA update 4001 has yet been accepted.**

## 2. Current production decision

The active trajectory is now qualified to resume from `step-00004000` using the existing mixed-precision FLA GDN-2 execution path on:

```text
fla-core==0.5.2
PyTorch 2.10.0+cu128
CUDA runtime 12.8
Triton 3.6.0
Tesla T4 / SM75
```

Checkpoint/model semantics remain unchanged:

```text
saved gdn_chunk_size: 32
FLA internal execution chunk: 64
recurrence equation: unchanged
learned decay parameterization: unchanged
state-dict keys: unchanged
decay clipping/bounding: none
```

Full-FP32 FLA also passed the final qualification but is slower. Mixed FLA is the selected production path because it passed the same corrected synthetic and real-checkpoint gates and was fastest in the warmed benchmark.

## 3. Why the investigation started

The completed 20M/100M pretraining run stayed numerically trainable and validation improved, but throughput collapsed severely:

```text
early: ~3830 target tok/s
late:  ~445 target tok/s
slowdown: ~8.6x
```

Data wait remained only milliseconds. The adaptive PyTorch GDN-2 backend was identified as the dominant execution pathology: it computes cumulative decay span, synchronizes GPU-to-CPU through `.item()`, and recursively subdivides configured chunks as learned decay strengthens.

FLA's fixed chunkwise CUDA path was therefore investigated as an exact execution replacement rather than changing model decay semantics.

## 4. GDN-2 semantic boundary that must remain fixed

Per head, the intended recurrence is conceptually:

```text
e_t = b_t ⊙ k_t
z_t = w_t ⊙ v_t
S_bar_t = Diag(exp(g_t)) S_(t-1)
S_t = S_bar_t + k_t (z_t - S_bar_t^T e_t)^T
o_t = S_t^T q_t / sqrt(d_k)
```

Small-LLM computes learned log-decay approximately as:

```text
g = -exp(A_log) * softplus(decay_features + dt_bias)
```

The investigation did **not** clip or bound this learned decay, change its parameterization, alter recurrence equations, or modify checkpoint keys.

## 5. Early FLA evidence and checkpoint-compatible integration

Initial `fla-core==0.5.1` standalone T4 tests showed very large performance gains and forward correctness. Representative historical measurements included approximately 18x normal forward/backward speedup and over 100x speedup under forced strong decay versus the adaptive implementation.

The integration preserved historical checkpoint geometry:

```text
saved/model config:              gdn_chunk_size = 32
CPU/reference adaptive backend:  configured chunk 32
CUDA FLA execution:              internal chunk 64
```

FLA adds no learned state-dict entries.

## 6. Genuine first production failure: AMP dtype mismatch

The first real FLA resume attempt correctly restored `step-00004000` and failed before update 4001 completed with a Triton compile assertion:

```text
Both operands must be same dtype. Got fp32 and fp16
b_u = tl.dot(b_A, b_vb)
```

Under trainer AMP, q/k could reach FLA as FP32 while v/write were FP16. This was a real adapter/runtime bug.

The adapter was corrected so ordinary compute tensors entering FLA are canonicalized consistently under AMP while decay/state remain FP32:

```text
q         -> FP16
k         -> FP16
v         -> FP16
erase     -> FP16
write     -> FP16
log_decay -> FP32
state     -> FP32
```

No update 4001 was committed by the failed attempt.

## 7. Historical decay-dependent backward rejection

After the dtype fix, trainer-realistic full-layer probes and forced-decay sweeps were interpreted as evidence that released FLA chunk backward became non-finite for some decay configurations. Historical summaries included:

```text
v0.5.1 passing: [-0.25, -0.5]
v0.5.1 failing: [-0.75, -1.0, -1.25, -1.5, -2.0, -3.0, -4.0, -5.0, -6.0]

v0.5.2 passing: [-0.25, -0.5, -1.0]
v0.5.2 failing: [-0.75, -1.25, -1.5, -2.0, -3.0, -4.0, -5.0, -6.0]
```

Real forward-only step-4000 decay telemetry showed that the active model's decay distribution overlapped those tested configurations. Based on the information available then, production FLA was correctly blocked and ADR 0020 authorized a full-FP32 investigation.

Those historical evidence files and ADRs are intentionally preserved. They document the decisions made with the then-available diagnostics.

## 8. Final correction: the synthetic comparison oracle was invalid

The live T4 execution of the prepared FP32 qualification exposed a critical detail: the supposedly failing rows had non-finite gradients in the **adaptive reference**, while FLA's corresponding gradients were finite.

Example from the old harness:

```text
g=-0.50:
  x              ref_nonfinite=8192   fla_nonfinite=0
  A_log          ref_nonfinite=1      fla_nonfinite=0
  dt_bias        ref_nonfinite=6      fla_nonfinite=0
  q_proj.weight  ref_nonfinite=16384  fla_nonfinite=0
```

The cause was CUDA autocast contamination of the reference. The adaptive recurrence explicitly casts inputs to FP32, but it was still invoked inside the outer FP16 autocast region. Eligible matrix multiplications inside the reference could therefore be executed in FP16.

The old sweep had a second reproducibility flaw: source/upstream tensors were seeded, but layer initialization was not reset per decay row, so different decay values used different randomly initialized layers. This helped create a misleading non-monotonic pattern.

Therefore the historical decay-sweep results do not establish an FLA-specific backward instability. They remain evidence of a failed qualification harness, not evidence to erase.

## 9. Corrected reference contract

The current oracle wraps only the adaptive recurrence in:

```text
CUDA autocast disabled
```

while the surrounding Small-LLM layer still executes under:

```text
FP32 master parameters + CUDA FP16 autocast
```

The corrected synthetic harness also fixes:

```text
layer initialization seed: 20260808
input/upstream seed: 12345
```

A candidate can fail only against a finite FP32 adaptive reference.

A separate reference-only T4 sweep confirmed finite output and gradients at every requested constant decay from `-0.25` through `-6.0`.

## 10. Corrected `fla-core==0.5.2` synthetic qualification

Command:

```text
python kaggle/run_gdn2_fla_fp32.py
```

Live result:

```text
mixed FLA passing:
[-0.25, -0.5, -0.75, -1.0, -1.25, -1.5, -2.0, -3.0, -4.0, -5.0, -6.0]

mixed FLA failing: []

full-FP32 FLA passing:
[-0.25, -0.5, -0.75, -1.0, -1.25, -1.5, -2.0, -3.0, -4.0, -5.0, -6.0]

full-FP32 FLA failing: []
invalid reference rows: []
```

This means the full-FP32 mode passes, but the experiment no longer supports claiming that FP32 fixed a mixed-precision FLA bug because mixed FLA itself passes the corrected gate.

## 11. Verified real checkpoint / next-block restoration

The remote restore path verified:

```text
checkpoint_id: step-00004000
global_step: 4000
last_consumed_block_id: 3999
next block: 4000
consumed target tokens: 131072000
checkpoint GradScaler scale: 256.0
```

The attached private 500M dataset manifest matched the restored checkpoint's Drive manifest.

Real next block geometry:

```text
16 sequences x 2048 tokens
microbatch: 4
target tokens: 32768
```

## 12. Real step-4000 / block-4000 forward-backward parity

`kaggle/run_gdn2_fla_step4000_parity.py` runs the complete next training block using:

- real checkpoint weights;
- exact next block 4000;
- FP32 master parameters + FP16 autocast;
- microbatch 4;
- checkpoint loss scale 256 before backward;
- the trainer's summed cross-entropy normalized by full-block target count;
- unscaled gradients for comparison;
- no clipping, optimizer step, scheduler step, acknowledgement, W&B, or checkpoint publication.

Result:

```text
REAL_STEP_4000_PARITY: PASS
```

Mixed FLA:

```text
forward parity: PASS
all gradients finite: PASS
all parameter gradient parity: PASS
gradient failures: 0
loss: 3.907714068889618
|loss-reference|: 5.161762237548828e-05
max full-logit abs diff: 0.078125
max parameter-gradient abs diff: 0.000125885009765625
```

Full-FP32 FLA:

```text
forward parity: PASS
all gradients finite: PASS
all parameter gradient parity: PASS
gradient failures: 0
loss: 3.9077218174934387
|loss-reference|: 4.38690185546875e-05
max full-logit abs diff: 0.0625
max parameter-gradient abs diff: 0.000133514404296875
```

FP32 adaptive reference loss:

```text
3.9077656865119934
```

## 13. Warmed real-block benchmark

After Triton/JIT/autotune warmup, two full real-block forward/backward measurements per backend gave:

```text
adaptive FP32 recurrence:
  16.5891 s
  16.7668 s
  median: 1964.75 target tok/s

FLA mixed:
  1.43426 s
  1.44445 s
  median: 22765.80 target tok/s
  speedup: 11.587x adaptive

FLA full FP32:
  1.53569 s
  1.54912 s
  median: 21244.76 target tok/s
  speedup: 10.813x adaptive
```

All measured gradients were finite.

Mixed FLA is therefore the fastest tested backend that satisfies the exact-semantics correctness gate.

## 14. Production dependency alignment

The final live qualification was on `fla-core==0.5.2`. Production declarations were therefore aligned after qualification:

```text
model/gdn2_fla.py FLA_CORE_VERSION = 0.5.2
pyproject model extra: fla-core==0.5.2
pyproject fla extra:   fla-core==0.5.2
uv.lock:               fla-core 0.5.2
```

The implementation commit containing the qualified runtime is:

```text
c0214d00047c61a290d9a138a6bd94ed5701337c
```

The 500M launcher is pinned to that implementation commit. The later documentation/authorization commit intentionally points to the already-complete implementation commit so the launcher pin is not self-referential.

## 15. Current scientific conclusions

Strongly supported:

1. The adaptive backend is the main cause of the historical decay-related throughput pathology.
2. The original FLA AMP dtype mismatch was a genuine integration bug and is fixed by the existing adapter dtype canonicalization.
3. The later reported decay-dependent FLA NaN pattern was a false-positive attribution caused by an autocast-contaminated adaptive reference plus unseeded layer initialization.
4. With a deterministic finite FP32 oracle, `fla-core==0.5.2` mixed and full-FP32 FLA pass the entire requested synthetic decay sweep on T4.
5. Both modes pass the true step-4000/full-next-block forward and all-gradient parity gate with checkpoint loss scaling.
6. Warmed mixed FLA is approximately 11.59x faster than the adaptive reference on the real next block and is faster than full-FP32 FLA.
7. The mixed `fla-core==0.5.2` path is qualified for production continuation of the active checkpoint.

Still true and must not be overstated:

1. `fla-core==0.5.1` was not rerun through the final corrected full gate; production qualification is specifically for 0.5.2.
2. Qualification did not execute update 4001 or prove the remainder of the 500M trajectory will never encounter a future unrelated failure.
3. The existing fail-closed trainer, scaler, checkpointing, and validation protections remain necessary.
4. No model-semantic change has been authorized or implemented.

## 16. Production boundary and next action

Authorized:

- resume the existing clean `step-00004000` checkpoint using the launcher pinned to the qualified `c0214d0...` implementation and `fla-core==0.5.2`;
- preserve exact checkpoint restore, microbatch 4, FP16 scaler behavior, and 250-update durability cadence.

Not authorized:

- changing saved `gdn_chunk_size=32`;
- clipping/bounding learned decay;
- changing decay parameterization or recurrence;
- changing checkpoint/state-dict keys;
- treating the qualification as if update 4001 already happened.

The next operational action is the ordinary fail-closed 500M resume. The first newly accepted trajectory point is the first actually completed optimizer update 4001.

## 17. Durable evidence

Final evidence:

```text
llm_docs/evidence/gdn2_fla_corrected_oracle_and_step4000_qualification_2026-08-08.md
llm_docs/evidence/gdn2_fla_fp32_qualification_corrected_2026-08-08.json
llm_docs/evidence/gdn2_fla_step4000_parity_2026-08-08.json
llm_docs/evidence/gdn2_fla_step4000_benchmark_2026-08-08.json
```

Historical evidence is retained under `llm_docs/evidence/`, including the old v0.5.1/v0.5.2 failed sweeps and the genuine first AMP dtype compile failure.

## 18. Durable decisions

- ADR 0018: checkpoint-compatible FLA integration.
- ADR 0019: earlier resume authorization, later operationally blocked during investigation.
- ADR 0020: full-FP32 diagnostic authorization after the then-believed backward failures.
- ADR 0021: corrected-oracle qualification of `fla-core==0.5.2` and authorization to resume the existing step-4000 trajectory.

Earlier ADRs are not rewritten to hide failed reasoning; ADR 0021 records the superseding evidence and current decision.