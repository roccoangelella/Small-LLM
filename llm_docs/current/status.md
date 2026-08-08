---
status: current
last_reviewed: 2026-08-08
---

# Current project status

## Completed 20M / 100M pretraining experiment

The approximately-20M-parameter GDN-2 hybrid completed its fixed approximately-100M-token pretraining schedule.

Canonical final evidence:

```text
W&B run ID: 20m-100m-data-004
optimizer updates: 3,053
consumed training target tokens: 100,018,176
final validation loss: 4.252758495143203
final validation perplexity: 70.29906475797992
final checkpoint: step-00003053
```

The run stayed trainable but suffered an approximately 8.6x throughput collapse, from roughly 3,830 target tok/s early to roughly 445 tok/s late. Data wait remained negligible. Controlled backend experiments strongly support the diagnosis that stronger learned GDN-2 decay exposed pathological subdivision and synchronization in the adaptive PyTorch chunk backend.

## Active 20M / 500M experiment

The 500M run is an independent seed-17 trajectory, not a continuation of the completed 100M run.

Latest accepted trajectory point:

```text
checkpoint: step-00004000
last_consumed_block_id: 3999
next intended update: 4001
W&B run ID: 20m-500m-data-001
context: 2048
microbatch: 4
saved gdn_chunk_size: 32
```

The checkpoint is clean. No FLA migration or qualification diagnostic has committed update 4001.

## FLA GDN-2 training status — QUALIFIED ON T4

The final live August 8 qualification used:

```text
GPU: Tesla T4 / SM75
PyTorch: 2.10.0+cu128
CUDA runtime: 12.8
Triton: 3.6.0
fla-core: 0.5.2
trainer contract: FP32 master parameters + CUDA FP16 autocast
saved GDN chunk: 32
FLA internal runtime chunk: 64
```

The previously reported v0.5.1/v0.5.2 decay-dependent backward failures were found to have been attributed through an invalid adaptive-reference harness. The reference was executed inside CUDA FP16 autocast, so eligible reference matrix operations could be downcast even after explicit FP32 tensor conversion. The old sweep also did not reset layer initialization per decay row.

On the live notebook, the old failing rows showed non-finite gradients on the **reference** side while FLA gradients were finite. Historical evidence remains preserved, but it is no longer valid evidence of an FLA-specific decay-dependent backward bug.

A corrected deterministic oracle now disables autocast only inside the adaptive recurrence while leaving the surrounding layer under the actual trainer AMP contract.

### Corrected synthetic decay qualification

Requested constant-decay sweep:

```text
[-0.25, -0.5, -0.75, -1.0, -1.25, -1.5, -2.0, -3.0, -4.0, -5.0, -6.0]
```

Result:

```text
mixed FLA passing: all 11
mixed FLA failing: []

full-FP32 FLA passing: all 11
full-FP32 FLA failing: []

invalid reference rows: []
```

### Real step-4000 / exact-next-block parity

The verified remote checkpoint was restored with cursor 3999, and the exact next training block 4000 was used in a complete accumulated forward/backward gate:

```text
block: 16 x 2048
microbatch: 4
target tokens: 32768
checkpoint GradScaler scale: 256.0
```

The diagnostic reproduced trainer FP16 autocast and checkpoint loss scaling but deliberately stopped before clipping, optimizer/scheduler mutation, data acknowledgement, W&B writes, or checkpoint publication.

Result:

```text
REAL_STEP_4000_PARITY: PASS

mixed FLA:
  forward parity: PASS
  all gradients finite: PASS
  all parameter gradient parity: PASS
  gradient failures: 0

full-FP32 FLA:
  forward parity: PASS
  all gradients finite: PASS
  all parameter gradient parity: PASS
  gradient failures: 0
```

### Warmed real-block throughput

```text
adaptive FP32 recurrence: 1964.75 target tok/s
FLA mixed:               22765.80 target tok/s   (11.587x adaptive)
FLA full FP32:           21244.76 target tok/s   (10.813x adaptive)
```

All benchmark backward passes remained finite.

The selected production backend is therefore **mixed FLA on `fla-core==0.5.2`**. Full-FP32 FLA is qualified as a diagnostic/fallback mode but is slower and unnecessary for the observed trajectory.

## Production resume boundary

Production continuation from the clean `step-00004000` checkpoint is authorized under ADR 0021 using the qualified `fla-core==0.5.2` implementation.

This authorization does not claim update 4001 has already happened. Qualification performed no optimizer step. The first newly accepted trajectory point remains the first actually completed production update 4001.

The production launcher is pinned to the implementation commit containing:

- the existing checkpoint-compatible FLA adapter and AMP dtype canonicalization;
- `fla-core==0.5.2` as the production model runtime dependency;
- unchanged saved `gdn_chunk_size=32` with FLA internal chunk 64;
- no learned-state or recurrence changes.

At production resume, keep the existing fail-closed restore and durability behavior and the normal 250-update validation/checkpoint/verified-remote-publication cadence.

## Frozen/accepted decisions still in force

- Preserve checkpoint/model config `gdn_chunk_size=32`.
- The latest accepted 500M checkpoint before resume is `step-00004000`.
- Keep the adaptive PyTorch backend as the correctness/reference implementation.
- Do not clip/bound GDN-2 decay solely because of backend runtime behavior.
- Start/resume the 500M experiment at microbatch 4.
- Validate/checkpoint/publish every 250 successful 500M updates.
- Let FP16 loss scaling calibrate down to scale 1.0 before failing an otherwise atomic block.
- Preserve `eval_core_v1` plus free-generation and teacher-forced confidence/rank diagnostics.
- The qualified production FLA runtime is `fla-core==0.5.2` on Tesla T4 / SM75.
- Historical failed qualification evidence is retained; changed conclusions are recorded in later evidence/ADRs rather than rewriting earlier records.

## Current source of truth

- Consolidated handoff: [`gdn2_fla_investigation_handoff.md`](gdn2_fla_investigation_handoff.md)
- Detailed qualification: [`gdn2_fla_qualification.md`](gdn2_fla_qualification.md)
- FP32/corrected-oracle gate: [`gdn2_fla_fp32_qualification.md`](gdn2_fla_fp32_qualification.md)
- Final live evidence: [`../evidence/gdn2_fla_corrected_oracle_and_step4000_qualification_2026-08-08.md`](../evidence/gdn2_fla_corrected_oracle_and_step4000_qualification_2026-08-08.md)
- Corrected synthetic JSON: [`../evidence/gdn2_fla_fp32_qualification_corrected_2026-08-08.json`](../evidence/gdn2_fla_fp32_qualification_corrected_2026-08-08.json)
- Real parity JSON: [`../evidence/gdn2_fla_step4000_parity_2026-08-08.json`](../evidence/gdn2_fla_step4000_parity_2026-08-08.json)
- Warmed benchmark JSON: [`../evidence/gdn2_fla_step4000_benchmark_2026-08-08.json`](../evidence/gdn2_fla_step4000_benchmark_2026-08-08.json)
- Durable decisions: [`../decisions/README.md`](../decisions/README.md)