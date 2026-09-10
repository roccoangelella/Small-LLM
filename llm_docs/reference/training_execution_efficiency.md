# Training execution efficiency

_Last reviewed: 2026-09-10_

Contracts and mechanisms that govern how fast one training update executes, as distinct from what
it computes. The recipe (model, data, optimizer arithmetic, schedule) is owned by the other
reference documents and by ADRs; this document owns the execution path and its measurement rules.
Baseline evidence: [`../evidence/moe_execution_profile_rtx4090_2026-09-08.md`](../evidence/moe_execution_profile_rtx4090_2026-09-08.md).

## Regime

On the measured RTX 4090 baseline the GPU executes useful matmul for ~7 % of the update and any
kernel for ~26 %; the remainder is host-side Python, kernel-launch overhead (~6.5 µs each, half a
million per update) and host–device synchronizations. Model FLOP utilization is ≈ 4 %. A geometry
with 10× fewer active FLOPs per token and 8× more experts (the accepted E64/Top-2 design) makes this
worse unless the number of launches and syncs per token falls. Consequently:

1. Launch and synchronization count per token is the first-order metric; FLOP count is not.
2. Microbatch size is the cheapest lever once memory allows it (fewer launches per token).
3. Precision (FP8) acts only on the matmul share and is evaluated last.
4. Never multiply speedups from overlapping mechanisms; measure each change alone (Amdahl).

## Implemented contracts (ADRs 0170–0172)

| mechanism | contract | equivalence proof | file |
|---|---|---|---|
| Muon batched by shape | `HybridMuonAdamW._muon_group_step` buckets Muon parameters with gradients by `(shape, device)`; one batched Newton–Schulz (`bmm`) per bucket; momentum and weight updates via `torch._foreach_*`; per-matrix norm, zero-update short circuit, RMS rescale, state-dict layout and `RECIPE` unchanged; a non-finite gradient aborts before its bucket mutates | bit-identical parameters and momentum on CPU over three steps; tolerance `rtol=1e-5, atol=1e-6` is the portable contract for CUDA batched GEMMs | `trainer/optimizer.py` (`_newton_schulz_orthogonalize_batched`), `trainer/optimizer_telemetry.py` (`_on_muon_update` hook, pre-update snapshot) |
| Fused attention | `GatedMultiheadAttention.forward` uses `scaled_dot_product_attention` (`is_causal=True` for full attention; boolean `allowed_mask` for the sliding window); `reference_mix` keeps the FP32-score unfused path for tests | outputs and all gradients within FP32 rounding on CPU for full and windowed attention; causality and window semantics unchanged | `model/components.py` |
| Chunked output loss | `model.fused_loss.chunked_cross_entropy_sum(hidden, weight, labels, semantic_vocab_size, ignore_index, chunk_tokens=4096)` scores `hidden @ weight[:semantic].T` in token chunks under non-reentrant activation checkpointing; live logits bounded by chunk × V; padded vocabulary rows never scored; `reduction="sum"` and explicit `ignore_index` | value and gradients (hidden, tied weight) equal the unchunked call within FP32 rounding for divisible and ragged chunks, with and without ignored positions | `model/fused_loss.py` |
| Hidden-state entry points | `SmallLLM.hidden_states(ids)` and `MoESmallLLM.hidden_states_with_aux(ids)` return final-normed hidden states; `forward` / `forward_with_aux` still return logits for evaluation and inference; the training steps use the chunked path when the entry point exists and the full-logits path otherwise | end-to-end dense and MoE loss/gradient equality tests | `model/model.py`, `MOE_model/model.py`, `trainer/step.py::_cross_entropy_sum`, `MOE_model/step.py::_moe_cross_entropy_sum` |
| Sorted MoE dispatch | one stable `argsort` of `expert_indices`, one host read of the E cumulative counts, contiguous slices per expert, one `cat`, one in-place `index_copy_` over the permutation; router, telemetry, controller and dtype boundary untouched | bit-identical outputs, z-loss and gradients versus the previous `nonzero`/out-of-place `index_copy` path; profiler asserts 0 `nonzero`, 1 `index_copy_`, 1 `cumsum` per layer | `MOE_model/model.py::DroplessTop1MoE.forward` |

Not changed by these contracts: routing semantics, Top-1 dropless guarantee, z-loss, checkpoint
identity, recipe identity, evaluation code paths.

## Remaining systems work, in priority order

1. **Profile the accepted many-expert geometry on the target GPU** with the largest microbatch that
   fits (the E64/Top-2 model with optimizer state is ≈ 1.7 GB; a 24 GB card leaves the rest to
   activations). Required by ADR 0168 (main) before any 100B sparse run. Report the four clocks
   separately (update, allocated, device, calendar) and never a single tokens/s.
2. **Batched expert GEMM**: pad each expert's sorted slice to the batch maximum and run one `bmm` per
   layer (composes with ADR 0172; wastes FLOPs proportional to load imbalance; needs the maximum
   count on host, which the single boundary read already provides). Alternative: a grouped-GEMM
   kernel where the library supports the target architecture.
3. **CUDA graphs / `torch.compile`** on the fixed-shape parts after the dispatch is static enough;
   dropless routing with variable expert counts recompiles unless shapes are bucketed.
4. **Telemetry cost**: per-parameter statistics remain on-device reductions; the `before` snapshot
   in the instrumented optimizer is one clone per Muon matrix per update — keep the telemetry-off
   lane for throughput measurements.
5. **FP8** on Ada tensor cores, only after MFU is in the tens of percent; requires a scaling recipe
   and a learning-quality comparison, not only a throughput number.

## Measurement rules

- Compare warm synchronized update seconds on identical workloads; six adjacent updates give no
  confidence interval for a months-long run.
- A profiler run inflates CPU time (19.0 s versus 11.8 s on the baseline); use it for proportions and
  counts, never for absolute projections.
- Equivalence is proven on CPU per change with the previous implementation kept as the oracle in the
  test file; GPU numerics may differ at rounding level and are covered by tolerance, not equality.
- Cost projections use `cost = rate × quantity` with the allocated-time clock, never the update
  clock alone.
