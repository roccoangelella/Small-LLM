# GDN-2 Chunkwise Training

_Last updated: 2026-08-06_

## Decision

On 2026-08-01 the user approved implementing a real chunkwise Gated DeltaNet-2 training path rather than leaving the repository with only the serial recurrent oracle.

The repository now contains both execution forms and one training stability wrapper:

- `gdn2_recurrent_reference`: readable token-by-token FP32 correctness oracle;
- `gdn2_chunkwise_reference`: differentiable PyTorch WY-style chunkwise implementation;
- `PyTorchGDN2Backend`: callable wrapper for the recurrent oracle;
- `PyTorchChunkwiseGDN2Backend`: direct callable wrapper for fixed-size chunkwise execution;
- `AdaptiveChunkwiseGDN2Backend`: maximum-size chunk execution with numerical subchunking;
- `StableGatedDeltaNet2`: the GDN-2 layer used by assembled `SmallLLM` models;
- `GatedDeltaNet2`: the underlying reference layer retained for direct backend testing.

The general default chunk size remains 64 tokens. The active 20M / 100M experiment explicitly configures a maximum chunk size of 32. A final shorter chunk is supported, so sequence length does not need to be divisible by the configured maximum.

## Mathematical contract

For each head, GDN-2 updates a matrix state `S_t` using channel-wise decay, erase, and write controls:

```text
e_t = b_t ⊙ k_t
z_t = w_t ⊙ v_t
S_bar_t = Diag(exp(g_t)) S_(t-1)
S_t = S_bar_t + k_t (z_t - S_bar_t^T e_t)^T
o_t = S_t^T q_t / sqrt(d_k)
```

The recurrent oracle evaluates these equations one token at a time.

The chunkwise path splits the sequence into fixed-size chunks. For one chunk, it computes cumulative log-decay `G`, absorbs the decay into asymmetric key and erase factors, builds the strictly lower-triangular intra-chunk interaction matrix, and solves the small unit-lower-triangular WY system. Token interactions inside a chunk are then evaluated with dense matrix products. Only the state transition between chunks remains sequential.

The direct chunk implementation uses an algebraically equivalent centered decay factorization. Instead of explicitly forming `exp(-G)`, it factors each ratio `exp(G_r - G_s)` around a per-channel midpoint.

## Numerical stability contract

The centered factors are reciprocal. Dense matrix products also materialize anti-causal entries before triangular masking. Strong but valid negative decay can therefore overflow an intermediate even when every required recurrent result is finite.

Assembled training models use `AdaptiveChunkwiseGDN2Backend` rather than changing or clamping the recurrence:

1. Propose the configured maximum chunk.
2. Compute its cumulative log-decay span.
3. Keep the chunk when the span is at most 60.
4. Otherwise bisect it until the span has conservative FP32 exponent and reduction headroom.
5. If a selected chunk still returns non-finite output or state, retry it at smaller sizes.
6. Fail closed only when the one-token execution remains non-finite.

The configured `gdn_chunk_size` is therefore a maximum, not a promise that every region uses that exact width. Ordinary low-span regions retain the configured size.

The adaptive wrapper does not alter:

- learned parameters or parameter names;
- model configuration serialization;
- optimizer routing or optimizer state;
- checkpoint state-dict keys;
- the mathematical recurrent update.

## Precision policy

The following operations run in FP32:

- cumulative log-decay;
- decay exponentials;
- triangular solve;
- chunk auxiliaries and matrix products;
- recurrent state and final state.

The token output is cast back to the query dtype after the chunk computation. Non-finite output or state triggers adaptive retry in assembled models and fails loudly at one-token granularity.

## Verification contract

Repository tests require the direct chunkwise backend to agree with the recurrent oracle for:

- every token output;
- final recurrent state;
- gradients with respect to Q, K, V, log-decay, erase gate, write gate, and initial state;
- sequences spanning multiple chunks;
- a partial final chunk;
- one-shot, segmented, and tokenwise cache execution;
- causal behavior and ordinary layer backward propagation.

Adaptive stability tests additionally require:

- output, state, and gradient parity for a 32-token constant-`-6` decay case that overflows the old centered execution;
- identical checkpoint parameter keys between reference and stable GDN-2 layers;
- assembled GDN layers to use the adaptive backend with the configured maximum chunk size.

The qualification helper `assert_gdn2_backend_parity(..., check_gradients=True)` exposes output, state, and gradient checks for future optimized backends.

## Current capability boundary

The repository has a genuine autograd-compatible chunkwise training algorithm with a correctness-preserving numerical fallback. It is no longer accurate to describe GDN-2 training as only a serial Python token loop or as requiring a fixed smaller chunk globally.

However, the implementation still uses ordinary PyTorch cumulative sums, triangular solves, device-synchronized span checks, and matrix multiplications. It is a correctness-first backend, not a claim of parity with an official fused Triton kernel's throughput or memory efficiency.

The following remain unqualified:

- corrected-run passage beyond the previous T4 failure boundary at update 1,138;
- frequency and throughput cost of adaptive subchunking over the remaining corpus;
- fused forward and gate-aware backward kernels;
- installation and compatibility of upstream optimized GDN-2 kernels;
- a T4-specific CUDA/CUTLASS implementation if upstream kernels are unavailable.

The step-1,138 incident and adaptive decision are recorded in:

- [`../evidence/20m_100m/gdn2_nonfinite_step_1138_2026-08-06.md`](../evidence/20m_100m/gdn2_nonfinite_step_1138_2026-08-06.md)
- [`../decisions/0005-adapt-gdn2-chunks-to-decay-span.md`](../decisions/0005-adapt-gdn2-chunks-to-decay-span.md)
