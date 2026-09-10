---
status: accepted
date: 2026-09-10
supersedes: null
owners: [Small-LLM]
---

# ADR 0170: freeze initial ultra-small MoE placement, expert form, and router-balancing contract

## Context and problem statement

ADR 0168 authorized investigation of an ultra-small many-expert MoE targeted at 100B tokens, and the subsequent architecture discussion selected the completed approximately-20M dense Small-LLM geometry as the comparison backbone. The next design step is to constrain the routed FFN and router behavior while leaving expert hidden width and exact Top-K open for a focused decision.

The primary scientific goal remains interpretability against the dense 20M baseline: changes should be limited to the sparse FFN system wherever possible rather than simultaneously introducing a new backbone or expert family.

## Considered options

- Use MoE only in a subset of blocks versus replacing every dense FFN.
- Introduce a latent/bottleneck expert architecture versus retaining the dense model's ordinary bias-free SwiGLU form.
- Use softmax routing versus sigmoid expert scores.
- Train load balance through an auxiliary loss, use a heuristic loss-free bias controller, or use non-gradient Quantile Balancing to determine selection biases.

## Decision outcome

Chosen initial contract:

1. **Every one of the eight decoder blocks uses an MoE FFN.** The mixer, norm, residual, depth, and other 20M-backbone semantics remain unchanged.
2. **Each routed expert is an ordinary bias-free SwiGLU of the same architectural form as the dense 20M FFN.** No LatentMoE bottleneck or alternate expert family is introduced in the initial experiment.
3. **Router expert scores use sigmoid, not softmax.** For token representation `x`, the learned router produces logits `z = W_r x` and base expert scores `s = sigmoid(z)`.
4. **Expert selection uses Top-K over bias-adjusted scores.** A per-expert balancing bias affects selection, i.e. selection is based on `s + b`. The exact value of `K` remains open at this ADR.
5. **The balancing bias does not directly scale expert outputs.** After Top-K selection, the selected experts are combined using the underlying unbiased sigmoid scores, normalized across the selected set. Thus balancing controls which experts are selected while task-loss gradients still train the router through the mixture weights.
6. **The initial load-balancing mechanism is Quantile Balancing (QB), following the Kimi K3 design principle.** The balancing signal is a non-gradient routing controller derived from router-score / selection-margin quantiles rather than an auxiliary load-balancing loss. The exact small-single-GPU implementation, update cadence, batch scope, and numerical estimator are still to be specified before code is wired.
7. **Expert hidden width remains open.** The design will explicitly evaluate the widths proposed in the attached architecture note (`h=256`, `352`, `448`) under the assumptions of that note before selecting one. Because those parameter counts assume Top-2, any change to `K` requires fresh active/total accounting.
8. Quantile Balancing is the selected first experiment, not a claim of universal optimality. If routing telemetry or quality evidence is poor, later ablations may replace it with another balancing mechanism under an explicit new decision.

## Consequences

### Positive

- Sparse placement is maximally clean: all eight dense FFNs are replaced and no other backbone component is changed.
- Keeping the original SwiGLU expert form isolates sparsity/routing from latent-bottleneck changes.
- Sigmoid scores permit independent pre-selection expert affinities; normalization is applied only after selecting the routed set.
- Separating selection bias from combination weights prevents the non-gradient balancing controller from directly becoming an expert-output coefficient.
- Quantile Balancing supplies a deterministic utilization constraint without adding an auxiliary load-balancing gradient to the language-model objective.

### Negative or limiting

- Quantile Balancing constrains allocation; it does not add representational capacity and may over-constrain naturally non-uniform specialization if applied too rigidly.
- Kimi K3's QB was designed at vastly larger expert count and distributed expert-parallel scale, so its exact engineering choices are not automatically appropriate for a 64-ish-expert single-GPU tiny MoE.
- Expert width and Top-K remain coupled unresolved choices; the attached `h=256/352/448` accounting is valid only under its fixed Top-2 setup.

## Validation

Before implementation is wired, the design review must specify exact Top-K, expert count, expert hidden width, shared-expert policy, dropless/capacity semantics, QB estimator/update semantics, router initialization/stabilization, and required telemetry. The user must demonstrate understanding of the selected implementation before code changes are made.

During qualification, record per-layer expert counts, max/mean load, load CV or equivalent dispersion, selection-bias distribution, router-score/entropy statistics, selected-score mass, dead/underused experts, token-drop count, and training/validation quality so QB can be compared with a later balancing ablation if needed.

## Links

- [`0168-investigate-ultra-small-sparse-moe-for-100b-tokens.md`](0168-investigate-ultra-small-sparse-moe-for-100b-tokens.md)
- [`0154-freeze-first-8-expert-top1-moe-experiment-contract.md`](0154-freeze-first-8-expert-top1-moe-experiment-contract.md)
- [`../reference/model_architecture.md`](../reference/model_architecture.md)
- [`../reference/model_geometry.md`](../reference/model_geometry.md)
