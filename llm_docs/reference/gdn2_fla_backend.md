# GDN-2 FLA production backend

_Last reviewed: 2026-08-10 Europe/Rome_

This document is the current technical contract for GDN-2 CUDA execution. It deliberately omits the investigation chronology; historical qualification handoffs are archived and raw measurements remain under `../evidence/`.

## Selected backend

The production CUDA path is mixed-precision FLA on:

```text
fla-core: 0.5.2
qualified GPU: Tesla T4 / SM75
PyTorch: 2.10.0+cu128
CUDA runtime: 12.8
Triton: 3.6.0
trainer contract: FP32 master parameters + CUDA FP16 autocast
saved/configured gdn_chunk_size: 32
FLA internal runtime chunk: 64
```

The adaptive PyTorch GDN-2 recurrence remains the correctness/reference fallback. Full-FP32 FLA remains a diagnostic/fallback mode rather than the selected production mode.

## Semantic boundary

FLA is an execution replacement, not a model change. Production must preserve:

- the GDN-2 recurrence equation;
- learned decay parameterization;
- state-dict/checkpoint keys;
- saved model configuration `gdn_chunk_size=32`;
- no decay clipping/bounding introduced solely for runtime behavior.

FLA's internal 64-token execution chunk is an implementation detail and is not written into historical checkpoint model geometry.

## AMP dtype contract

Under trainer AMP, ordinary FLA compute tensors are canonicalized consistently to low precision while decay/state remain FP32. The earlier real-resume mixed-dtype Triton failure was an adapter integration bug, not a checkpoint/model incompatibility.

## Qualification basis

The current selection is supported by the corrected deterministic qualification in which the adaptive recurrence oracle executes with CUDA autocast disabled while the surrounding Small-LLM layer retains the real trainer AMP contract.

On the final corrected T4 gate:

```text
constant-decay sweep: mixed FLA passed all requested rows
real step-4000 next-block forward/backward parity: passed
all compared parameter gradients: finite and within gate
optimizer step during parity qualification: none
```

Warmed true-block throughput measured:

```text
adaptive FP32 recurrence: 1,964.75 target tok/s
mixed FLA:               22,765.80 target tok/s
full-FP32 FLA:           21,244.76 target tok/s
mixed speedup:           11.587x vs adaptive
```

The historical interpretation that released FLA had a decay-dependent backward NaN problem was invalidated by the corrected oracle: the old comparison allowed FP16 autocast to contaminate the supposed FP32 adaptive reference and did not reset layer initialization for every decay row.

## Evidence and decisions

Canonical final evidence:

- [`../evidence/gdn2_fla_corrected_oracle_and_step4000_qualification_2026-08-08.md`](../evidence/gdn2_fla_corrected_oracle_and_step4000_qualification_2026-08-08.md)
- [`../evidence/gdn2_fla_fp32_qualification_corrected_2026-08-08.json`](../evidence/gdn2_fla_fp32_qualification_corrected_2026-08-08.json)
- [`../evidence/gdn2_fla_step4000_parity_2026-08-08.json`](../evidence/gdn2_fla_step4000_parity_2026-08-08.json)
- [`../evidence/gdn2_fla_step4000_benchmark_2026-08-08.json`](../evidence/gdn2_fla_step4000_benchmark_2026-08-08.json)

Durable decisions:

- ADR 0018 — checkpoint-compatible FLA integration.
- ADR 0021 — corrected-oracle `fla-core==0.5.2` qualification and production continuation.

Detailed August 8 investigation handoffs are preserved under `../archive/gdn2_fla_investigation/` and are not current authorization.