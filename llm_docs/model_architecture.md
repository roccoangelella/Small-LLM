# Model Architecture

_Last updated: 2026-07-31_

## Scope

The base model is a dense decoder-only language model below 1B parameters. The implementation must be geometry-scalable: the same block code and configuration system must support tiny smoke models, controlled intermediate experiments, the first approximately 100M substantive model, and later larger models.

## Decoder macroarchitecture

The dominant token mixer is Gated DeltaNet-2. Periodic full causal softmax-attention layers provide unrestricted content-based token-to-token retrieval.

The frozen initial pattern is:

```text
[GDN-2, GDN-2, GDN-2, MHA] × N
```

This is a 3:1 GDN-2-to-MHA ratio. The first substantive model uses five repetitions, for 20 decoder blocks total: 15 GDN-2 blocks and 5 MHA blocks.

Every decoder block has two sequential residual branches:

```text
x = x + Mixer(RMSNorm(x))
x = x + SwiGLU(RMSNorm(x))
```

This is sequential pre-norm, not parallel residual execution.

## Normalization

Use RMSNorm rather than LayerNorm.

- Apply pre-RMSNorm before every mixer branch.
- Apply a separate pre-RMSNorm before every FFN branch.
- Start with `eps = 1e-6`.
- Apply one final RMSNorm after the final decoder block and immediately before the tied language-model head.

The final RMSNorm is required because the last residual update would otherwise reach the output projection without a subsequent pre-norm operation.

## Full-attention layers

Use ordinary multi-head causal self-attention, not GQA, in the first implementation.

For each MHA layer:

```text
Q = X W_q
K = X W_k
V = X W_v
Q_rot = RoPE(Q, positions)
K_rot = RoPE(K, positions)
A = softmax((Q_rot K_rotᵀ) / sqrt(d_head) + causal_mask)
Y = A V W_o
```

Initial rules:

- independent Q, K, and V heads;
- full causal attention;
- fixed RoPE on Q and K only;
- full-head RoPE;
- conventional RoPE base near 10,000 for the initial 2,048-token context;
- no RoPE modification of V;
- no learned absolute position embeddings.

GQA remains a later serving or long-context optimization, not an initial capacity choice.

## Gated DeltaNet-2 layers

GDN-2 supplies causal, order-sensitive recurrent token mixing and therefore receives no separate RoPE in the initial implementation.

The initial layer follows the reference GDN-2 structure:

- independent Q, K, and V projections;
- channel-wise erase and write gates;
- causal recurrent matrix state;
- short depthwise convolutions on Q, K, and V;
- initial convolution kernel size 4;
- gated output normalization and output projection;
- chunkwise training kernel and recurrent inference path.

For the first implementation:

- key and value widths match the residual width;
- key and value head counts match;
- key and value head dimensions match;
- grouped value attention is not used;
- negative-eigenvalue support is not enabled unless separately benchmarked.

RoPE inside GDN-2, grouped value geometry, disabling short convolution, and negative-eigenvalue variants are controlled later ablations.

## Feed-forward network

Every decoder block uses a dense SwiGLU FFN, regardless of mixer type:

```text
g = W_gate x
u = W_up x
h = SiLU(g) ⊙ u
y = W_down h
```

The two expanded vectors are multiplied elementwise; they are not concatenated. Therefore `d_ff` is the width of each branch and of the combined intermediate result.

Each decoder layer owns independent `W_gate`, `W_up`, and `W_down` matrices. The matrices are shared across token positions within a layer but never shared across decoder layers.

No activation follows `W_down`; its output enters the residual addition.

## Embeddings and output

Use the GPT-2 token-ID vocabulary already present in the corpus.

- semantic vocabulary size: 50,257;
- input embeddings and output LM projection are tied;
- the same matrix `E` is used for lookup and for `logits = x Eᵀ`;
- the implementation may pad the internal matrix to 50,304 rows for hardware alignment;
- padded token IDs are never emitted as valid training targets and their logits must be masked or excluded from sampling and loss.

No learned positional embedding table is used.

## Initial context

The development and initial architecture-trial context is fixed at 2,048 input tokens. Dataset records store 2,049 IDs so every 2,048-token input has a one-token-shifted target.

Longer contexts are deferred until the base architecture, recurrence, trainer, checkpointing, and throughput are validated.

## Biases, dropout, and initialization

These are not fully frozen yet.

The likely starting policy is bias-free ordinary linear projections, with only reference-required GDN-2 gate biases retained, and zero dropout because of the intended data scale. Exact initialization, residual scaling, and gate initialization require a separate decision and implementation test.
