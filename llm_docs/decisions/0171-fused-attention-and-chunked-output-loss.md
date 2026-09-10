---
status: accepted
date: 2026-09-10
supersedes: null
---

# 0171 — Fused scaled-dot-product attention and chunked output loss

## Context and problem statement

Two training-step allocations grew with sequence length and vocabulary rather than with useful work. `GatedMultiheadAttention` materialized the full [batch, heads, T, T] FP32 score tensor, masked it, ran a separate softmax and a second contraction (`masked_fill_`, `_softmax`, `_softmax_backward_data` and `bmm` all appear in the 2026-09-08 RTX 4090 profile). The training steps computed `F.cross_entropy` on full [tokens, vocabulary] logits: with 131,072 targets per update that is ~13 GB of BF16 logits at V=50,304 and ~2 GB at V=8,000, kept alive for backward and reproduced as FP32 inside the loss. Both limit the microbatch size, which is the cheapest lever against the launch-bound regime measured on the 4090 (MFU ≈ 4%).

## Considered options

- Keep the unfused reference paths.
- Use `torch.nn.functional.scaled_dot_product_attention` (`is_causal` for full attention, an explicit boolean mask for the sliding window) and compute the summed cross-entropy in token chunks with non-reentrant activation checkpointing, so each chunk's logits are recomputed in backward instead of stored.
- Custom Triton kernels for the output loss (more speed, more surface; deferred until the many-expert path is profiled).

## Decision outcome

Chosen option: fused SDPA plus chunked loss, behavior-preserving.

- `GatedMultiheadAttention.forward` calls `scaled_dot_product_attention`; `allowed_mask` and `reference_mix` keep the previous FP32-score semantics available for tests. Causality and the opt-in window are unchanged.
- `model/fused_loss.py::chunked_cross_entropy_sum` scores `hidden @ weight[:semantic].T` in chunks of `DEFAULT_CHUNK_TOKENS = 4096` tokens, `reduction="sum"`, explicit `ignore_index`; padded vocabulary rows are never scored. Live logits are bounded by chunk × V; backward recomputes one output projection per chunk.
- `SmallLLM.hidden_states` and `MoESmallLLM.hidden_states_with_aux` expose final-normed hidden states; `forward` / `forward_with_aux` still return logits for evaluation and inference.
- `trainer/step.py::_cross_entropy_sum` and `MOE_model/step.py::_moe_cross_entropy_sum` use the chunked path when the model exposes the hidden-state entry point and keep the full-logits path otherwise (test stubs and foreign models). Loss normalization, z-loss weighting, overflow handling, telemetry and profiling regions are unchanged.

## Consequences

- Values match the reference paths at FP32 rounding level on CPU (tests: `rtol=1e-5`, `atol=1e-6` for attention outputs and gradients; `atol=1e-5` on summed losses). Under BF16 autocast on CUDA the fused kernels accumulate in FP32 but round differently from the previous explicit-FP32-score path; this is a numerics change within mixed-precision noise, not a recipe change, and no checkpoint identity field changes.
- Backward pays one extra output-projection matmul per chunk; forward attention no longer allocates T×T scores. Wall-clock and memory gains on GPU are not measured by this decision and require the next authorized profile with a larger microbatch.
- Evaluation code that calls `model(ids)` for logits is untouched.

## Validation

`tests/test_fused_attention_and_loss.py`: fused vs reference attention (full and window 4, outputs and all gradients), causality/window semantics, chunked vs full cross-entropy for divisible and ragged chunk sizes with and without `ignore_index`, padded-row gradients exactly zero, end-to-end dense and MoE loss and gradient equality. The existing model, MoE, controller, pilot protocol, evaluation, CLI observation and optimizer suites pass (71 tests, 1 CUDA skip).

## Links

- [Batched Muon](0170-batch-muon-newton-schulz-by-matrix-shape.md)
- [Pilot contract](0168-moe-paired-pilot-controller-and-observation.md)
