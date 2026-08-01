# Model Geometry

_Last updated: 2026-08-01_

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
| Hybrid SwiGLU width `d_ff` | 704 |
| Matched Plan B / Plan C `d_ff` | 835 |
| MHA heads | 4 |
| MHA head dimension | 64 |
| GDN key heads | 4 |
| GDN value heads | 4 |
| GDN key dimension per head | 64 |
| GDN value dimension per head | 64 |
| Tied embeddings | yes |
| Final RMSNorm | yes |

Implemented exact counts:

| Smoke schedule | Exact parameters | Difference from hybrid |
|---|---:|---:|
| Primary GDN-2 hybrid, `d_ff=704` | 20,637,592 | — |
| Plan B SWA/full-attention transformer, `d_ff=835` | 20,634,880 | -2,712 (-0.013%) |
| Plan C all-MHA transformer, `d_ff=835` | 20,634,880 | -2,712 (-0.013%) |

Plan B and Plan C have equal learned parameter counts because sliding-window attention changes only the mask, not the projections or normalization parameters.

## Frozen first substantive configuration

Purpose: first real comparison of the hybrid architecture against matched transformer references.

| Quantity | Value |
|---|---:|
| Target parameters | approximately 100M |
| Context | 2,048 |
| Residual width `d_model` | 512 |
| Decoder layers | 20 |
| GDN-2 layers | 15 |
| MHA layers | 5 |
| Layer pattern | `[GDN-2, GDN-2, GDN-2, MHA] × 5` |
| Hybrid SwiGLU width `d_ff` | 1,408 |
| Matched Plan B / Plan C `d_ff` | 1,603 |
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
| Padded vocabulary | 50,304 |

The source of truth is the implemented model's exact parameter counter, split by embeddings, GDN-2 mixers, MHA mixers, FFNs, norms, and other parameters.

## Implementation-verified substantive parameter counts

The former pre-implementation estimate of approximately 99.9M understated the gated-MHA parameters because the implemented attention mixer has five full-width projections: Q, K, V, output gate, and output projection.

### Primary GDN-2 hybrid

| Component | Exact parameters |
|---|---:|
| Tied embedding/output matrix | 25,755,648 |
| 20 SwiGLU FFNs at `d_ff=1408` | 43,253,760 |
| 5 gated MHA mixers | 6,554,240 |
| 15 GDN-2 mixers | 25,667,640 |
| Block and final RMSNorms outside mixer counts | 20,992 |
| **Total** | **101,252,280** |

Mixer-local QK norms, GDN output norms, and GDN reference-required offsets are included in their corresponding mixer totals.

### Parameter-matched transformer schedules

Both Plan B and Plan C replace the same 15 GDN-2 mixers with gated attention mixers. Since SWA-512 and full MHA differ only in their causal mask, both use the same closest integral compensating FFN width:

```text
d_ff = 1603
```

| Transformer schedule | Exact parameters | Difference from hybrid |
|---|---:|---:|
| Plan B `[SWA-512, SWA-512, SWA-512, full MHA] × 5` | 101,237,760 | -14,520 (-0.014%) |
| Plan C `[full MHA] × 20` | 101,237,760 | -14,520 (-0.014%) |

The 14,520-parameter remainder is caused by integral FFN width. The primary hybrid remains frozen at `d_ff=1408`; widening is used only to compensate transformer replacements for controlled comparisons.

## Scale templates

These are planning templates, not yet authorized production models. Transformer replacements should derive their matched FFN width from the implemented parameter counter rather than copying the hybrid `d_ff` unchanged.

| Role | Approx. parameters | `d_model` | Layers | Hybrid `d_ff` | Heads × head dimension |
|---|---:|---:|---:|---:|---:|
| Kernel smoke | 20M | 256 | 8 | 704 | 4 × 64 |
| Intermediate debug | 44M | 384 | 12 | 1,024 | 6 × 64 |
| First substantive | 100M | 512 | 20 | 1,408 | 8 × 64 |
| Medium trial | approximately 200M | 768 | 20 | 2,048 | 12 × 64 |
| Serious trial | approximately 344M | 1,024 | 20 | 2,816 | 16 × 64 |

Larger configurations should preserve a 64-dimensional head initially unless profiling or quality evidence supports 128-dimensional heads or expanded projected widths.

## Why the primary dimensions are hardware-friendly

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

The frozen substantive residual geometry is exactly:

```text
512 = 8 × 64
```

The matched transformer width `1603` is selected for parameter equality rather than tile alignment. Before a large transformer-reference run, benchmark it against a nearby aligned width such as 1,600; changing to the aligned width would be a documented compute-efficiency variant, not the default matched comparison.

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

At the substantive hybrid geometry:

```text
3 × 512 × 1408 = 2,162,688 parameters per FFN
```

The two expanded branches are multiplied elementwise; they are not concatenated.

## GDN-2 recurrent state geometry

With 8 heads and 64-dimensional keys and values, each GDN-2 layer maintains a recurrent matrix state with approximately:

```text
8 × 64 × 64 = 32,768 state values per active sequence
```

This state is independent of context length during recurrent decoding. Efficient training still requires a chunkwise or otherwise parallel backend that stores the activations needed for backpropagation.

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
- matched loss curves against Plan B and Plan C.

Scale decisions must follow measurements rather than parameter labels alone.
