---
status: accepted
date: 2026-08-08
supersedes: null
---

# 0021 — Qualify FLA GDN-2 v0.5.2 and resume step 4000

## Context and problem statement

The active 20M/500M trajectory is clean at:

```text
checkpoint: step-00004000
last_consumed_block_id: 3999
next update: 4001
```

The adaptive PyTorch GDN-2 backend is exact but becomes pathologically slow as learned decay strengthens. A checkpoint-compatible FLA CUDA adapter already exists and preserves saved `gdn_chunk_size=32` while FLA internally executes 64-token chunks.

A genuine first production FLA attempt failed before update 4001 because AMP sent incompatible tensor dtypes into a Triton WY dot product. The adapter was fixed to canonicalize ordinary compute tensors to a consistent low-precision dtype while retaining FP32 decay/state.

Later forced-decay trainer-AMP probes appeared to show decay-dependent FLA backward NaNs in v0.5.1 and v0.5.2, so production was blocked and ADR 0020 authorized a full-FP32 diagnostic.

The live Tesla T4 execution of that diagnostic exposed a qualification-harness defect: the adaptive PyTorch correctness reference itself was run inside CUDA FP16 autocast. Eligible matrix operations in the reference could therefore be downcast even after explicit FP32 tensor conversions. Failing rows showed non-finite reference gradients with finite FLA gradients. The old sweep also failed to reseed layer initialization for each decay row.

Historical failed evidence and earlier ADRs remain preserved. The new evidence corrects the current interpretation rather than rewriting them.

## Considered options

- Keep production blocked and fall back to the adaptive PyTorch backend.
- Require full-FP32 FLA for production because it was the originally authorized diagnostic candidate.
- Correct the oracle, qualify both current mixed FLA and full-FP32 FLA against it, run the exact real step-4000/next-block parity gate, benchmark warmed throughput, and select the fastest exact-semantics passing backend.
- Change learned decay semantics through clipping/bounding or reparameterization.

## Decision outcome

Chosen option: **qualify and resume with the existing mixed FLA execution path on `fla-core==0.5.2`**, because it passes both the corrected deterministic synthetic gate and the real step-4000/full-next-block gate while retaining the best measured throughput.

The corrected oracle disables CUDA autocast only inside the adaptive recurrence reference. The surrounding layer still uses FP32 master parameters plus CUDA FP16 autocast. Layer initialization is deterministic across every forced-decay row.

Corrected synthetic v0.5.2 result:

```text
tested decay:
[-0.25, -0.5, -0.75, -1.0, -1.25, -1.5, -2.0, -3.0, -4.0, -5.0, -6.0]

mixed FLA passing: all 11
mixed FLA failing: []
full-FP32 FLA passing: all 11
full-FP32 FLA failing: []
invalid reference rows: []
```

The verified real checkpoint was restored with cursor 3999, and the exact next training block 4000 was exercised over all 16 x 2048 sequences using microbatch 4 and checkpoint GradScaler scale 256. The gate stopped before clipping or optimizer/scheduler/data mutation.

Real result:

```text
REAL_STEP_4000_PARITY: PASS

mixed FLA:
  forward parity: PASS
  all gradients finite: PASS
  all parameter gradient parity: PASS

full-FP32 FLA:
  forward parity: PASS
  all gradients finite: PASS
  all parameter gradient parity: PASS
```

Warmed real-block throughput:

```text
adaptive FP32 recurrence: 1964.75 target tok/s
mixed FLA:               22765.80 target tok/s   (11.587x adaptive)
full-FP32 FLA:           21244.76 target tok/s   (10.813x adaptive)
```

Therefore full-FP32 FLA remains a qualified diagnostic/fallback mode, but mixed FLA is selected for production because it is also correct under the tested contract and is faster.

Production runtime is aligned to:

```text
fla-core==0.5.2
```

The implementation commit qualified on the live T4 and selected for the production launcher is:

```text
c0214d00047c61a290d9a138a6bd94ed5701337c
```

Production continuation from `step-00004000` is authorized. This ADR does **not** claim update 4001 has already happened; no qualification diagnostic executed an optimizer step.

## Consequences

### Positive

- Preserves exact GDN-2 recurrence semantics and existing learned parameters.
- Preserves checkpoint/state-dict compatibility and saved `gdn_chunk_size=32`.
- Removes the false production block caused by an invalid comparison oracle.
- Uses a real checkpoint/full-next-block gradient gate rather than synthetic evidence alone.
- Provides approximately 11.59x warmed throughput versus the adaptive backend on the actual next block.
- Keeps full-FP32 FLA available as a qualified fallback/diagnostic mode.
- Aligns production dependency metadata and the adapter version declaration to the tested `fla-core==0.5.2` release.

### Negative or limiting

- The final corrected gate qualifies v0.5.2 specifically; it does not retroactively qualify v0.5.1.
- First-run Triton autotuning/compilation on SM75 is expensive and should not be confused with steady-state throughput.
- Passing block 4000 cannot prove that no unrelated numerical or runtime failure will occur later in the trajectory; existing fail-closed trainer and checkpoint protections remain necessary.
- Historical failed sweep documents now require reading the later correction to understand their scientific status.

## Validation

The decision is validated by all of the following completed gates:

1. Tesla T4 / SM75 runtime with PyTorch `2.10.0+cu128`, CUDA 12.8, Triton 3.6.0, `fla-core==0.5.2`.
2. Corrected finite deterministic FP32 adaptive reference across the full requested synthetic decay sweep.
3. Mixed FLA output and all-gradient parity across all 11 synthetic decay values.
4. Full-FP32 FLA output and all-gradient parity across all 11 synthetic decay values.
5. Verified restore of `step-00004000` with `last_consumed_block_id=3999` and the matching attached 500M dataset.
6. Full true block-4000 accumulated forward/backward parity with checkpoint loss scale 256 and no optimizer step.
7. Warmed real-block benchmark showing both FLA modes remain significantly faster than adaptive, with mixed FLA fastest.
8. Production dependency/adapter pins resolve to `fla-core==0.5.2` under the locked `model` extra.

Production must fail closed if the pinned implementation cannot be restored, if `fla-core==0.5.2` cannot be resolved/imported, if checkpoint/dataset identity differs, or if the trainer encounters its existing finite-loss/gradient/scale/checkpoint safety failures.

## Links

- [`../current/gdn2_fla_qualification.md`](../current/gdn2_fla_qualification.md)
- [`../current/gdn2_fla_fp32_qualification.md`](../current/gdn2_fla_fp32_qualification.md)
- [`../current/gdn2_fla_investigation_handoff.md`](../current/gdn2_fla_investigation_handoff.md)
- [`../evidence/gdn2_fla_corrected_oracle_and_step4000_qualification_2026-08-08.md`](../evidence/gdn2_fla_corrected_oracle_and_step4000_qualification_2026-08-08.md)
- [`../evidence/gdn2_fla_fp32_qualification_corrected_2026-08-08.json`](../evidence/gdn2_fla_fp32_qualification_corrected_2026-08-08.json)
- [`../evidence/gdn2_fla_step4000_parity_2026-08-08.json`](../evidence/gdn2_fla_step4000_parity_2026-08-08.json)
- [`../evidence/gdn2_fla_step4000_benchmark_2026-08-08.json`](../evidence/gdn2_fla_step4000_benchmark_2026-08-08.json)
- [`../runbooks/20m_500m_runbook.md`](../runbooks/20m_500m_runbook.md)