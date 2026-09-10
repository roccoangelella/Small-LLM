---
status: accepted
last_reviewed: 2026-09-10
---

# ADR 0169 — Record initial MoE router numerical contract

## Context

The active MoE architecture investigation lives on branch `moe-8e-top1`. The detailed experiment decisions are maintained there, but the project-level memory must retain the accepted numerical contract for the initial ultra-small MoE routing experiment.

## Decision

For the first 20M-backbone MoE experiment:

- use dropless Top-2 routing with exact Quantile Balancing;
- aggregate QB routing statistics across the entire logical optimizer step, keeping one fixed balancing bias for all gradient-accumulation microbatches; compute the new exact quantile after the complete step and apply the new mean-centered bias only on the next optimizer step;
- use router affinity `s = sqrt(softplus(z))`, where `z = W_r x`;
- select experts with `TopK(s + b)`, but compute normalized selected-expert mixture weights from the unbiased `s`, never from `s + b`;
- initialize the trainable router matrix as `Normal(0, 0.02^2)` and initialize the non-gradient QB bias `b` to zero;
- keep all router/QB arithmetic in FP32;
- optimize the router matrix with AdamW, initial `weight_decay=0`, and no router-specific LR multiplier initially;
- do not use the classical softmax-router z-loss initially;
- use the original 20M training recipe as the conservative optimization baseline: hybrid Muon+AdamW for the common model parameters, peak LR `3e-4`, WSD, max-grad-norm `1.0`, minimum-LR ratio `0.1`; the eventual long-run warmup/stable/decay token horizons remain deliberately unfrozen;
- defer production sparse-backend selection to the later systems work with Edo;
- keep an approximately 8k-token vocabulary as the current tokenizer target, while deferring tokenizer implementation details.

## Initialization rationale

The chosen router normal standard deviation is `0.02`. DeepSeek-V4 is the closest current score-function reference because it uses Sqrt(Softplus) MoE affinity scoring and its released Transformers configuration/implementation uses `initializer_range=0.02` with normal initialization for the router matrix. Kimi K3 is not evidence that its router itself uses the global 0.02 normal initializer: the released Kimi K3 text implementation explicitly resets the custom MoE gate with Kaiming-uniform. Because Small-LLM has selected Sqrt(Softplus), and because its existing dense normal initializer also uses `std=0.02`, the DeepSeek-V4-aligned value is the cleaner first experiment. With `d_model=256` and a unit-RMS normalized router input, this gives an estimated initial router-logit standard deviation of about `0.32`, providing symmetry breaking without deliberately starting with wide logits.

## Implementation gate

No implementation is authorized by this decision alone. Before wiring the accepted design, the user must explain the trainable-router-versus-QB-bias distinction, the selection-only role of the QB bias, the next-step semantics of step-level QB under gradient accumulation, and the behavior of Sqrt(Softplus) relative to sigmoid.

## Detailed experiment memory

See branch `moe-8e-top1`, especially ADR 0173 `finalize-initial-moe-router-numerics.md`.
