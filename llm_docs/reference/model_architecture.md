# Model architecture

_Last reviewed: 2026-08-13_

## Production family

Small-LLM is a dense decoder-only family below 1B parameters. The selected hybrid block rhythm is:

```text
[GDN-2, GDN-2, GDN-2, gated full MHA] × N
```

Each decoder block is sequential pre-norm:

```text
x = x + Mixer(RMSNorm(x))
x = x + SwiGLU(RMSNorm(x))
```

A final RMSNorm precedes the tied language-model head. Dropout is zero in the current pretraining family.

## Completed primary geometries

- 20M: 8 layers = 6 GDN-2 + 2 gated full-MHA, `d_model=256`, `d_ff=704`.
- 100M: 20 layers = 15 GDN-2 + 5 gated full-MHA, `d_model=512`, `d_ff=1408`.

Exact parameter counts live in [`model_geometry.md`](model_geometry.md).

## GDN-2 execution boundary

The mathematical GDN-2 recurrence, learned decay, state-dict keys, and serialized model geometry are independent of the execution backend.

Production CUDA execution is the qualified mixed FLA adapter on `fla-core==0.5.2`:

```text
FP32 master parameters + CUDA FP16 autocast
saved/configured gdn_chunk_size: 32
FLA internal runtime chunk: 64
```

The readable tokenwise PyTorch recurrence and PyTorch chunkwise/adaptive implementations remain reference/correctness fallbacks. They are not the selected production CUDA training backend. See [`gdn2_fla_backend.md`](gdn2_fla_backend.md) and [`gdn2_chunkwise_training.md`](gdn2_chunkwise_training.md).

## GDN-2 layer contract

GDN-2 supplies causal recurrent token mixing with independent Q/K/V projections, channel-wise decay/erase/write controls, a recurrent matrix state, short depthwise causal convolutions on Q/K/V, output normalization/gating, and output projection. The current short-convolution kernel is 4. GDN-2 does not use a separate RoPE path.

Backend changes must preserve checkpoint keys and recurrence semantics. Runtime fixes must not clamp or redefine learned decay merely to avoid kernel behavior.

## Full attention layers

Every fourth layer is full causal gated MHA. The current attention contract uses:

- independent Q/K/V heads;
- per-head QK RMSNorm before RoPE;
- full-head RoPE on Q/K only;
- sigmoid elementwise output gate before output projection;
- no attention dropout;
- bias-free ordinary Q/K/V/gate/output projections.

The 20M geometry uses 4 heads × 64 dimensions; the 100M geometry uses 8 × 64.

## Feed-forward network

Every block uses dense SwiGLU:

```text
g = W_gate x
u = W_up x
h = SiLU(g) * u
y = W_down h
```

The three ordinary FFN projections are bias-free and layer-local.

## Embedding/output contract

The semantic GPT-2 vocabulary is 50,257 IDs. Input embedding and LM output weights are tied. The internal matrix is padded to 50,304 rows for hardware alignment, but logits are cropped to the semantic vocabulary before loss/evaluation/sampling. Padding rows are not semantic tokens.

There are no learned absolute position embeddings.

## Context and state isolation

Current training/evaluation context is 2,048 input tokens with 2,049 stored context+1 IDs. Independent training records do not share GDN recurrent/convolution state.

## Controlled alternatives

Parameter-matched local/global transformer and all-gated-MHA schedules remain scientific baselines/fallback architectures. They are not silently substituted for the completed hybrid trajectories. Any architecture comparison must explicitly match data, training targets, optimizer contract, and evaluation identity.

## Initialization

Completed scaling trajectories use their frozen run configuration and seed policy; initialization is no longer an unresolved prerequisite for the 100M model. New geometry changes require a new explicit qualification rather than editing the completed-run contract retrospectively.
