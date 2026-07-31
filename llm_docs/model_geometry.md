# Model Geometry

_Last updated: 2026-07-31_

## Geometry strategy

The project does not hard-code one final model size. It defines a scalable family and validates the same architecture at progressively larger parameter budgets.

The scale sequence is:

1. tiny smoke model for correctness and integration;
2. approximately 100M first substantive architecture trial;
3. optional intermediate models when a benchmark needs more signal;
4. later 300–350M and larger models only after measured justification;
5. a near-1B model remains a long-term goal, not the first training run.

All configurations preserve the same broad architecture unless an experiment explicitly varies it:

```text
[GDN-2, GDN-2, GDN-2, MHA] × N
```

## Frozen smoke configuration

Purpose: validate implementation, kernels, data flow, backward pass, generation, parameter counting, checkpoint/resume, and end-to-end integration. It is not intended for meaningful quality comparisons.

| Quantity | Value |
|---|---:|
| Target parameters | approximately 20M |
| Context | 2,048 |
| Residual width `d_model` | 256 |
| Decoder layers | 8 |
| GDN-2 layers | 6 |
| MHA layers | 2 |
| SwiGLU width `d_ff` | 704 |
| MHA heads | 4 |
| MHA head dimension | 64 |
| GDN key heads | 4 |
| GDN value heads | 4 |
| GDN key dimension per head | 64 |
| GDN value dimension per head | 64 |
| Tied embeddings | yes |
| Final RMSNorm | yes |

The implementation must calculate and report the exact parameter count rather than trusting the approximate label.

## Frozen first substantive configuration

Purpose: first real comparison of the hybrid architecture against a matched all-MHA baseline.

| Quantity | Value |
|---|---:|
| Target parameters | approximately 100M |
| Context | 2,048 |
| Residual width `d_model` | 512 |
| Decoder layers | 20 |
| GDN-2 layers | 15 |
| MHA layers | 5 |
| Layer pattern | `[GDN-2, GDN-2, GDN-2, MHA] × 5` |
| SwiGLU width `d_ff` | 1,408 |
| MHA heads | 8 |
| MHA head dimension | 64 |
| GDN key heads | 8 |
| GDN value heads | 8 |
| GDN key dimension per head | 64 |
| GDN value dimension per head | 64 |
| GDN short-convolution kernel | 4 |
| Tied embeddings | yes |
| Final RMSNorm | yes |
| Semantic vocabulary | 50,257 |
| Candidate padded vocabulary | 50,304 |

The expected total is approximately 100M parameters. The source of truth is the implemented model's exact parameter counter, split by embeddings, GDN-2 mixers, MHA mixers, FFNs, norms, and other parameters.

## Approximate 100M parameter decomposition

Before implementation-level verification, the working estimate is:

| Component | Approximate parameters |
|---|---:|
| Tied embedding/output matrix | 25.76M |
| 20 SwiGLU FFNs | 43.25M |
| 5 MHA mixers | 5.24M |
| 15 GDN-2 mixers | 25.67M |
| Norms and small parameters | small remainder |
| Total | approximately 99.9M |

The exact GDN-2 count depends on the final faithful implementation of gates, short convolutions, output normalization, and reference-required biases.

## Scale templates

These are planning templates, not yet authorized production models.

| Role | Approx. parameters | `d_model` | Layers | `d_ff` | Heads × head dimension |
|---|---:|---:|---:|---:|---:|
| Kernel smoke | 20M | 256 | 8 | 704 | 4 × 64 |
| Intermediate debug | 44M | 384 | 12 | 1,024 | 6 × 64 |
| First substantive | 100M | 512 | 20 | 1,408 | 8 × 64 |
| Medium trial | approximately 200M | 768 | 20 | 2,048 | 12 × 64 |
| Serious trial | approximately 344M | 1,024 | 20 | 2,816 | 16 × 64 |

Larger configurations should preserve a 64-dimensional head initially unless profiling or quality evidence supports 128-dimensional heads or expanded projected widths.

## Why these dimensions are hardware-friendly

GPU matrix kernels split dimensions into fixed-size tiles. Dimensions divisible by 8, 16, 32, or 64 usually avoid partially empty edge tiles and make optimized Tensor Core kernels easier to use.

Examples:

```text
512 = 8 × 64
1408 = 22 × 64
1024 = 16 × 64
2816 = 44 × 64
```

For ordinary MHA:

```text
d_model = n_heads × d_head
```

The frozen 100M choice is exactly:

```text
512 = 8 × 64
```

Arbitrary dimensions such as 515 or a 70-dimensional head can waste padded work, complicate reshaping, or force slower kernel paths.

## SwiGLU geometry

For residual width `d_model` and intermediate width `d_ff`:

```text
W_gate: d_model × d_ff
W_up:   d_model × d_ff
W_down: d_ff × d_model
```

Ignoring biases:

```text
parameters per SwiGLU FFN = 3 × d_model × d_ff
```

At the 100M geometry:

```text
3 × 512 × 1408 = 2,162,688 parameters per FFN
```

The two 1,408-wide branches do not make a 2,816-wide hidden vector because they are multiplied elementwise.

## GDN-2 recurrent state geometry

With 8 heads and 64-dimensional keys and values, each GDN-2 layer maintains a recurrent matrix state with approximately:

```text
8 × 64 × 64 = 32,768 state values per active sequence
```

This state is independent of context length during recurrent decoding, although training uses chunkwise kernels and stores additional activations for backpropagation.

## Benchmark contract

Before accepting a larger scale, measure on the target T4:

- exact parameters by component;
- peak training memory at context 2,048;
- maximum stable microbatch;
- tokens per second;
- forward and backward kernel time;
- GDN recurrent-state memory;
- MHA activation memory;
- checkpoint size and save/load time;
- matched loss curves against the all-MHA baseline.

Scale decisions must follow measurements rather than parameter labels alone.
