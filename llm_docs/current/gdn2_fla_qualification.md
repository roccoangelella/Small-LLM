---
status: current
last_reviewed: 2026-08-08
---

# Current GDN-2 backend qualification status

## Bottom line

The active 20M/500M trajectory is now **qualified to resume from `step-00004000` with the mixed-precision FLA GDN-2 backend on `fla-core==0.5.2`**.

The accepted trajectory point itself has not advanced during qualification:

```text
checkpoint: step-00004000
last_consumed_block_id: 3999
next update: 4001
update 4001 accepted by FLA: no
```

Qualification deliberately stopped before gradient clipping, optimizer/scheduler mutation, data acknowledgement, W&B writes, or checkpoint publication.

The production execution contract remains checkpoint-compatible:

```text
FP32 master parameters + CUDA FP16 autocast
saved/model gdn_chunk_size: 32
FLA internal runtime chunk: 64
recurrence equation: unchanged
learned decay parameterization: unchanged
state-dict/checkpoint keys: unchanged
decay clipping/bounding: none
```

For the full chronology, read [`gdn2_fla_investigation_handoff.md`](gdn2_fla_investigation_handoff.md). The final live evidence is [`../evidence/gdn2_fla_corrected_oracle_and_step4000_qualification_2026-08-08.md`](../evidence/gdn2_fla_corrected_oracle_and_step4000_qualification_2026-08-08.md).

## Why FLA was investigated

The completed approximately-20M / 100M run slowed from roughly 3,830 target tok/s early to roughly 445 tok/s late while validation kept improving and data wait remained tiny. Controlled tests strongly support the diagnosis that stronger learned GDN-2 decay made the correctness-first adaptive PyTorch backend repeatedly subdivide chunks and synchronize with Python.

Standalone FLA T4 tests were dramatically faster and forward-correct, so FLA was investigated as an execution replacement rather than changing learned decay semantics.

## Genuine integration failure that remains part of history

The first real 500M FLA resume attempt restored verified step 4000 but failed before update 4001 completed with a Triton dtype assertion:

```text
Both operands must be same dtype. Got fp32 and fp16
b_u = tl.dot(b_A, b_vb)
```

Under trainer AMP, q/k could reach FLA as FP32 while v/write were FP16. The Small-LLM adapter was subsequently corrected to canonicalize ordinary FLA compute tensors to the same low-precision dtype while keeping log-decay and recurrent state FP32.

That compile failure was real. It occurred before any successful update 4001, so the checkpoint remained clean.

Evidence: [`../evidence/gdn2_fla_500m_resume_amp_dtype_failure_2026-08-08.md`](../evidence/gdn2_fla_500m_resume_amp_dtype_failure_2026-08-08.md)

## Correction to the later decay-dependent NaN diagnosis

After the dtype fix, several historical trainer-AMP probes/sweeps were interpreted as evidence that FLA v0.5.1/v0.5.2 had decay-dependent backward NaNs. Those evidence files are intentionally preserved, but the live August 8 investigation found that the comparison oracle was invalid for attributing those failures to FLA.

The key defect was that the adaptive PyTorch reference was called inside the outer CUDA FP16 autocast context. Although the recurrence implementation explicitly converted tensors to FP32, eligible matrix multiplications could still be autocast back to FP16. On the live notebook, the supposedly failing rows showed the opposite of the prior interpretation: the **adaptive reference gradients were non-finite while the FLA gradients were finite**.

Representative old-harness row:

```text
g=-0.50:
  x              ref_nonfinite=8192   fla_nonfinite=0
  A_log          ref_nonfinite=1      fla_nonfinite=0
  dt_bias        ref_nonfinite=6      fla_nonfinite=0
  q_proj.weight  ref_nonfinite=16384  fla_nonfinite=0
```

A second harness defect made the earlier non-monotonic pattern less meaningful: source/upstream tensors were seeded, but layer initialization was not reset for every decay row, so different decay points used different random layers.

Therefore the historical v0.5.1/v0.5.2 sweep reports cannot be used as proof of an FLA-specific decay-dependent backward failure. They remain historical evidence of a failed qualification attempt, not evidence to delete or rewrite.

## Corrected deterministic FP32 oracle

The current qualification script executes only the adaptive recurrence oracle with CUDA autocast disabled. The rest of the Small-LLM layer remains under the real trainer contract of FP32 parameters + FP16 autocast.

It also resets:

```text
layer initialization seed: 20260808
input/upstream seed: 12345
```

A row can count as an FLA failure only if the FP32 adaptive reference is itself finite.

A separate reference-only T4 check confirmed finite outputs and finite gradients at all requested constant decay values:

```text
[-0.25, -0.5, -0.75, -1.0, -1.25, -1.5, -2.0, -3.0, -4.0, -5.0, -6.0]
```

## Corrected `fla-core==0.5.2` synthetic sweep

Command:

```text
python kaggle/run_gdn2_fla_fp32.py
```

Environment:

```text
Tesla T4 / SM75
PyTorch 2.10.0+cu128
CUDA runtime 12.8
Triton 3.6.0
fla-core 0.5.2
```

Result:

```text
mixed FLA passing:
[-0.25, -0.5, -0.75, -1.0, -1.25, -1.5, -2.0, -3.0, -4.0, -5.0, -6.0]

mixed FLA failing: []

full-FP32 FLA passing:
[-0.25, -0.5, -0.75, -1.0, -1.25, -1.5, -2.0, -3.0, -4.0, -5.0, -6.0]

full-FP32 FLA failing: []
invalid reference rows: []
```

The corrected sweep does not establish that FP32 fixed an FLA mixed-precision problem, because mixed FLA itself passes every tested decay against the valid oracle.

Raw evidence: [`../evidence/gdn2_fla_fp32_qualification_corrected_2026-08-08.json`](../evidence/gdn2_fla_fp32_qualification_corrected_2026-08-08.json)

## Real step-4000 / exact-next-block gate

The verified remote checkpoint was restored and matched to the attached 500M dataset:

```text
checkpoint_id: step-00004000
global_step: 4000
last_consumed_block_id: 3999
next block: 4000
block geometry: 16 x 2048
target tokens: 32768
microbatch: 4
checkpoint GradScaler scale: 256.0
```

`kaggle/run_gdn2_fla_step4000_parity.py` then ran one complete accumulated forward/backward over the true next block using checkpoint loss scaling, but stopped before any optimizer-side mutation.

Result:

```text
REAL_STEP_4000_PARITY: PASS

mixed FLA:
  forward parity: PASS
  all gradients finite: PASS
  all parameter gradient parity: PASS
  gradient failures: 0
  loss: 3.907714068889618
  max full-logit abs diff: 0.078125
  max parameter-gradient abs diff: 0.000125885009765625

full-FP32 FLA:
  forward parity: PASS
  all gradients finite: PASS
  all parameter gradient parity: PASS
  gradient failures: 0
  loss: 3.9077218174934387
  max full-logit abs diff: 0.0625
  max parameter-gradient abs diff: 0.000133514404296875

FP32 adaptive reference loss: 3.9077656865119934
optimizer step executed: NO
```

Raw evidence: [`../evidence/gdn2_fla_step4000_parity_2026-08-08.json`](../evidence/gdn2_fla_step4000_parity_2026-08-08.json)

## Warmed real-block throughput

After Triton compilation/autotuning was warm, the same real block was benchmarked without parity-copy instrumentation:

```text
adaptive FP32 recurrence: 1964.75 target tok/s
FLA mixed:               22765.80 target tok/s   (11.587x adaptive)
FLA full FP32:           21244.76 target tok/s   (10.813x adaptive)
```

All measured backward passes had finite gradients.

Raw evidence: [`../evidence/gdn2_fla_step4000_benchmark_2026-08-08.json`](../evidence/gdn2_fla_step4000_benchmark_2026-08-08.json)

## Selected production backend

The qualified production choice is the existing **mixed FLA path on `fla-core==0.5.2`**:

- it passes all corrected synthetic decay points;
- it passes the real checkpoint/full-next-block forward and all-gradient parity gate;
- it is the fastest tested exact-semantics backend;
- it preserves checkpoint/model semantics.

Full-FP32 FLA remains a useful diagnostic/fallback mode but is slower and is not required by the current evidence.

The production dependency pin and adapter-declared version are aligned to `fla-core==0.5.2`. The active launcher is pinned to the implementation commit containing that qualified runtime.

## Production boundary

Production continuation from `step-00004000` is authorized under ADR 0021, but qualification itself did not execute update 4001.

At resume:

1. restore the verified step-4000 checkpoint;
2. require cursor 3999 and next block 4000 as usual;
3. run the ordinary fail-closed trainer on the qualified `fla-core==0.5.2` implementation;
4. treat the first actually completed optimizer update 4001 as the first new accepted trajectory point;
5. preserve the existing 250-update validation/checkpoint/remote-publication cadence.

Do not change decay semantics, saved `gdn_chunk_size=32`, checkpoint keys, or recurrence equations.