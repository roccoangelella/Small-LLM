---
status: accepted
date: 2026-09-10
supersedes: null
---

# 0170 — Batch Muon Newton–Schulz by matrix shape

## Context and problem statement

`HybridMuonAdamW._muon_step` orthogonalized one logical matrix at a time: ten Newton–Schulz iterations, three GEMMs each, per matrix, plus per-matrix momentum and weight kernels. The RTX 4090 profile of 2026-09-08 (MoE 404M stored / 101M active, BF16, microbatch 2) recorded 521,704 kernel launches per update and an optimizer step of ~2.0 s device time; useful matmul time was ~1.4 s of a 19.0 s profiled update. With the accepted 64-expert geometry every MoE layer holds 192 expert matrices of one shape, so the per-matrix loop would issue ~46,000 GEMM launches per update for Newton–Schulz alone. Newton–Schulz cost scales with stored matrices, not with active parameters: with 131,072 targets per update essentially every expert receives gradient every step, so sparsity does not reduce optimizer work.

Numbering note: `main` carries a different ADR 0170 (MoE tokenizer target) and 0169 (expert scaling sequence); this branch keeps its own sequence as recorded in the decisions README.

## Considered options

- Keep the per-matrix loop and accept the launch count.
- Batch same-shape matrices into one stacked Newton–Schulz (`bmm`) and update momentum/weights with horizontally fused `torch._foreach_*` kernels, preserving the per-matrix arithmetic contract.
- Replace Muon on expert matrices with a cheaper optimizer (a recipe change requiring a learning comparison; out of scope here).

## Decision outcome

Chosen option: batch by shape. `HybridMuonAdamW.step` now calls `_muon_group_step`, which buckets Muon parameters with gradients by `(shape, device)`, validates every gradient in a bucket before mutating any state, updates momentum buffers with `_foreach_mul_/_foreach_add_`, computes the Nesterov term with the same `add(alpha=)` expression as the reference path, runs `_newton_schulz_orthogonalize_batched` once per bucket, and applies decay and the update with `_foreach_mul_/_foreach_add_`. Per-matrix semantics are unchanged: each matrix keeps its own norm, zero-update short circuit, RMS rescale, momentum buffer entry and state-dict layout; parameters without gradient are skipped; `RECIPE` and `identity()` are unchanged because the arithmetic recipe is unchanged. `InstrumentedHybridMuonAdamW` records per-matrix statistics through the `_on_muon_update` hook with a pre-update snapshot, as before. The single-matrix `_muon_step` remains as the numerical reference.

## Consequences

- Positive: on the tiny CPU MoE (125 Muon matrices, 3 shape buckets) one optimizer step issues 90 `bmm` instead of 3,750 `mm` and 13,702 ATen calls instead of 63,475; parameters and momentum buffers are bit-identical to the per-matrix path over three steps on CPU. Wall-clock savings on GPU are not measured by this decision; they require the next authorized profile.
- Failure boundary: a non-finite gradient now aborts before any parameter of its bucket is touched; the step remains non-transactional across buckets and groups (ADR record of `d71c3fd` still applies).
- GPU batched GEMMs may round differently from strided GEMMs; the portable contract is closeness at FP32 rounding level, tested with `rtol=1e-5, atol=1e-6`, with bitwise equality asserted on CPU.
- Not covered: sparse gradients (rejected as before), mixed devices within a group (bucketed separately), and any change to the Newton–Schulz coefficients or iteration count.

## Validation

`tests/test_trainer_optimizer_batched_muon.py`: batched vs single Newton–Schulz for both orientations and square matrices, a zero matrix inside a batch, non-finite rejection, a three-step grouped-vs-reference equality including momentum buffers, skipped parameters without gradient, bucket-level failure boundary, and instrumented statistics coverage. Existing optimizer, telemetry and MoE model tests pass unchanged (22 tests).

## Links

- [Optimizer failure boundary](../../trainer/optimizer.py) — `HybridMuonAdamW` docstring.
- [Pilot contract](0168-moe-paired-pilot-controller-and-observation.md)
