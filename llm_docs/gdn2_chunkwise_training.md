# GDN-2 Chunkwise Training

_Last updated: 2026-08-01_

## Decision

On 2026-08-01 the user approved implementing a real chunkwise Gated DeltaNet-2 training path rather than leaving the repository with only the serial recurrent oracle.

The repository now contains both execution forms:

- `gdn2_recurrent_reference`: readable token-by-token FP32 correctness oracle;
- `gdn2_chunkwise_reference`: differentiable PyTorch WY-style chunkwise training implementation;
- `PyTorchGDN2Backend`: callable wrapper for the recurrent oracle;
- `PyTorchChunkwiseGDN2Backend`: callable wrapper for chunkwise execution;
- `GatedDeltaNet2`: uses the chunkwise backend by default, with chunk size selected by `ModelConfig.gdn_chunk_size`.

The frozen default chunk size is 64 tokens. A final shorter chunk is supported, so sequence length does not need to be divisible by 64.

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

The implementation uses an algebraically equivalent centered decay factorization. Instead of explicitly forming potentially unstable `exp(-G)`, it factors each ratio `exp(G_r - G_s)` around a per-channel midpoint. This preserves the exact chunk equations while reducing overflow risk.

## Precision policy

The following operations run in FP32:

- cumulative log-decay;
- decay exponentials;
- triangular solve;
- chunk auxiliaries and matrix products;
- recurrent state and final state.

The token output is cast back to the query dtype after the chunk computation. Non-finite outputs or states fail loudly and recommend reducing the configured chunk size or using a qualified fused backend.

## Verification contract

Repository tests now require the chunkwise backend to agree with the recurrent oracle for:

- every token output;
- final recurrent state;
- gradients with respect to Q, K, V, log-decay, erase gate, write gate, and initial state;
- sequences spanning multiple chunks;
- a partial final chunk;
- one-shot, segmented, and tokenwise cache execution;
- causal behavior and ordinary layer backward propagation.

The qualification helper `assert_gdn2_backend_parity(..., check_gradients=True)` exposes the same output, state, and gradient checks for future optimized backends.

## Current capability boundary

The repository now has a genuine autograd-compatible chunkwise training algorithm. It is no longer accurate to describe GDN-2 training as only a serial Python token loop.

However, this first implementation uses ordinary PyTorch cumulative sums, triangular solves, and matrix multiplications. It is a correctness-first chunkwise backend, not a claim of parity with the official fused Triton kernel's throughput or memory efficiency.

The following remain unqualified:

- T4 FP16 numerical behavior;
- peak memory and tokens per second at context 2,048;
- whether chunk size 64 is optimal on Turing hardware;
- fused forward and gate-aware backward kernels;
- installation and compatibility of upstream optimized GDN-2 kernels;
- a T4-specific CUDA/CUTLASS implementation if upstream kernels are unavailable.

Substantive GDN-2 training is authorized only after the PyTorch chunkwise path or a replacement optimized backend passes target-T4 correctness, stability, memory, and throughput gates. Plan B and Plan C remain available so kernel optimization cannot block the project indefinitely.
