# Model Architecture

_Last updated: 2026-07-31_

## Scope

The base model is a dense decoder-only language model below 1B parameters. The implementation must be geometry-scalable: the same block code and configuration system must support tiny smoke models, controlled intermediate experiments, the first approximately 100M substantive model, and later larger models.

## Framework and kernel boundary

PyTorch is the canonical framework for the model, trainer, autograd, checkpointing, and correctness tests.

Optimized Triton, CUDA, or library kernels are optional backend implementations behind stable PyTorch module and autograd interfaces. They do not replace PyTorch or define the mathematical contract. Every optimized GDN-2 path must be checked against a readable PyTorch reference recurrence.

The initial GDN-2 backend ladder is:

1. readable PyTorch recurrent reference for correctness;
2. available optimized chunkwise and recurrent kernels when compatible with the target GPU;
3. a T4-specific CUDA or other compatible kernel only if profiling shows that the existing path is unsupported or inadequate.

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

Use full multi-head causal self-attention, not GQA, in the first implementation. The full-attention layers include per-head QK-RMSNorm and an elementwise sigmoid output gate.

For each MHA layer:

```text
Q = X W_q
K = X W_k
V = X W_v
Q_norm = RMSNorm_per_head(Q)
K_norm = RMSNorm_per_head(K)
Q_rot = RoPE(Q_norm, positions)
K_rot = RoPE(K_norm, positions)
A = softmax((Q_rot K_rotᵀ) / sqrt(d_head) + causal_mask)
H = A V
G = sigmoid(X W_gate)
Y = (H ⊙ G) W_o
```

Initial rules:

- independent Q, K, and V heads;
- full causal attention;
- per-head QK-RMSNorm before RoPE;
- fixed RoPE on Q and K only;
- full-head RoPE;
- conventional RoPE base near 10,000 for the initial 2,048-token context;
- no RoPE modification of V;
- elementwise sigmoid output gate before `W_o`;
- no learned absolute position embeddings;
- zero attention dropout;
- bias-free Q, K, V, gate, and output projections.

GQA remains a later serving or long-context optimization, not an initial capacity choice.

The parameter-matched all-MHA baseline uses the same QK-RMSNorm and attention output gate so that the comparison isolates the sequence mixer rather than unrelated attention details.

## Gated DeltaNet-2 layers

GDN-2 supplies causal, order-sensitive recurrent token mixing and therefore receives no separate RoPE in the initial implementation.

The initial layer follows the reference GDN-2 structure:

- independent Q, K, and V projections;
- channel-wise erase and write gates;
- causal recurrent matrix state;
- short depthwise causal 1D convolutions on Q, K, and V;
- initial convolution kernel size 4;
- gated output normalization and output projection;
- chunkwise training kernel and recurrent inference path.

The short convolutions are not a CNN backbone. Each projected channel is mixed only with its own recent token history, initially the current token and the previous three positions. This supplies a small local receptive field before the recurrent state update.

For the first implementation:

- key and value widths match the residual width;
- key and value head counts match;
- key and value head dimensions match;
- grouped value attention is not used;
- negative-eigenvalue support is not enabled unless separately benchmarked;
- Q, K, V, decay, erase, write, and output projections follow the reference-required bias policy;
- `A_log`, `dt_bias`, and decay arithmetic remain FP32 where required for stability;
- independent 2,048-token training records do not share recurrent or convolution state;
- chunkwise and recurrent paths must pass numerical-parity tests against the PyTorch reference.

RoPE inside GDN-2, grouped value geometry, disabling short convolution, resizing the convolution, and negative-eigenvalue variants are controlled later ablations.

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

No activation follows `W_down`; its output enters the residual addition. The three FFN projections are bias-free.

## Embeddings and output

Use the GPT-2 token-ID vocabulary already present in the corpus.

- semantic vocabulary size: 50,257;
- input embeddings and output LM projection are tied;
- the same matrix `E` is used for lookup and for `logits = x Eᵀ`;
- the internal tied matrix is padded to 50,304 rows for hardware alignment;
- token IDs 50,257 through 50,303 are internal padding rows, never semantic tokens;
- no additive embedding bias and no separate LM-head bias are used;
- padded token IDs are never accepted as normal inputs or emitted as training targets;
- compute the aligned 50,304-wide projection, then crop logits to `logits[..., :50257]` before cross-entropy, evaluation probabilities, or sampling;
- padded rows are initialized to zero and receive no semantic loss or sampling probability.

Cropping is preferred to treating the padded rows as real classes. It preserves the aligned embedding/output matrix while keeping the language-model distribution defined over exactly the semantic vocabulary.

No learned positional embedding table is used.

## Initial context

The development and initial architecture-trial context is fixed at 2,048 input tokens. Dataset records store 2,049 IDs so every 2,048-token input has a one-token-shifted target.

Longer contexts are deferred until the base architecture, recurrence, trainer, checkpointing, and throughput are validated.

## Bias and dropout policy

The initial model is bias-free in ordinary embedding, attention, FFN, and projection paths.

Retain only biases or bias-like learned offsets required by the faithful GDN-2 formulation, including its decay-step offset and any reference output-gating bias. These exceptions must be explicitly named in parameter accounting and optimizer exclusions.

Use zero dropout throughout the initial pretraining and architecture-comparison runs. Nonzero dropout remains a controlled ablation if overfitting or instability evidence later justifies it.

## Initialization

The exact global initialization and residual-scaling policy remains to be frozen after a small implementation test. The candidate contract is:

- embedding rows initialized from a zero-mean normal distribution;
- RMSNorm scales initialized to one;
- ordinary biases, where reference-required, initialized to zero;
- ordinary linear weights initialized with a variance-aware Xavier-style rule;
- residual output projections scaled with depth, approximately by `1 / sqrt(2L)` relative to their base initialization;
- GDN-2 `A_log` initialized from log-rates sampled in the reference range;
- GDN-2 `dt_bias` initialized through inverse softplus so initial time steps occupy the reference small positive range;
- padded vocabulary rows initialized to zero.

The final choice must be tested for forward variance, early loss, gradient norms, FP16 stability, and agreement with the reference GDN-2 implementation before the approximately 100M run.
