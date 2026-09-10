---
status: accepted
date: 2026-09-10
supersedes: null
owners: [Small-LLM]
---

# ADR 0168: investigate an ultra-small sparse MoE targeted at 100B tokens

## Context and problem statement

The existing dedicated MoE branch implements the first 8-expert Top-1 experiment contract from ADR 0154. A new architectural direction is now worth investigating: make the MoE itself much smaller in active compute and shared-backbone width while increasing the number of small routed experts, and train a custom tokenizer with a substantially smaller vocabulary than the current GPT-2 50,257-token semantic vocabulary.

The purpose of this experiment is not merely to reduce total parameter count. The scientific target is the smallest sparse language model that remains meaningfully functional while being inexpensive to pretrain on the project's constrained GPUs. The intended full-training data budget for this MoE experiment is 100B tokens, subject to a tokenizer-aware definition of what that token budget means.

The attached design proposal considered, among other points, a 64-expert Top-2/no-shared configuration with an 8-block hybrid backbone, an 8K vocabulary, `d_model=256`, and expert hidden width 352, yielding approximately 9.94M active and 144.03M total parameters by its accounting. Those exact geometry values are inputs to the feasibility study, not accepted architecture choices in this ADR.

## Considered options

- Continue directly with the already-wired 8-expert Top-1 MoE as the only sparse experiment.
- Freeze the proposed 64-expert/Top-2/8K/`d=256` geometry immediately and begin production wiring.
- First establish whether an ultra-small, many-expert MoE is scientifically and operationally viable, including tokenizer, GPU-kernel, routing, memory, throughput, and quality-floor constraints, before selecting its exact geometry.

## Decision outcome

Chosen option: **investigate the ultra-small, many-expert sparse MoE direction before freezing or wiring the final production architecture.**

The durable constraints of this investigation are:

1. The target experiment is deliberately much smaller in active compute than the current dense 100M/200M family and uses dozens of routed experts rather than only a handful.
2. A custom, substantially smaller vocabulary/tokenizer is in scope and must be evaluated as part of the architecture rather than treated as a free parameter reduction.
3. The intended long-run budget is 100B tokens. Because a different tokenizer changes the number of tokens representing the same text, the study must define and report both custom-token count and a tokenizer-independent corpus quantity (bytes/characters/documents, or an explicitly measured GPT-2-equivalent token count) before claiming data-budget comparability.
4. The final model should prioritize practical trainability on the project's economically constrained GPUs, not just theoretical active-parameter sparsity.
5. No exact expert count, Top-K, shared-expert choice, `d_model`, expert hidden width, layer count, vocabulary size, routing policy, or GPU kernel is frozen by this ADR.
6. No 100B MoE production run is authorized until the many-expert execution path is benchmarked on the actual target GPU(s). In particular, a Python loop issuing one small GEMM per expert is not sufficient evidence of economical sparse execution.
7. The existing 8-expert Top-1 branch remains a useful implementation prototype/comparison point. This ADR does not erase ADR 0154 or silently redefine its checkpoint identity; a later architecture decision may explicitly supersede it for the next MoE production target.
8. Before any proposed implementation change from this investigation is wired, the design must be explained and the user must demonstrate understanding through open-ended questions, per the project protocol.

## Consequences

### Positive

- The project can test a scientifically interesting regime close to the lower end of recent MoE scaling studies rather than assuming that sparse models only become useful at hundreds of millions of active parameters.
- Tokenizer size, total expert capacity, active compute, shared-backbone width, and real GPU efficiency are treated as coupled constraints.
- The 100B data target creates a strongly overtrained regime for a roughly-10M-active model, which is directly relevant to recent evidence that tiny MoEs need substantially more data than compute-optimal dense baselines before sparse capacity pays off.

### Negative or limiting

- The existing pretokenized 100B GPT-2 corpus cannot be treated as byte-identical training data for a custom tokenizer. A new tokenizer-aware production path and new dataset identity will be required if this direction is selected.
- Dozens of very small experts can be slower than their FLOP count suggests without grouped/fused expert GEMMs and efficient token dispatch.
- Reducing `d_model` shrinks the shared residual representation, attention/GDN mixers, embeddings, and every expert interface simultaneously; MoE capacity cannot automatically compensate for an excessively narrow shared backbone.
- A 100B-token run is too expensive to use as the first systems qualification. Kernel throughput, routing stability, tokenizer compression, and early capability checkpoints must be measured before committing to the full trajectory.

## Validation required before a final architecture ADR

- Recompute exact active/total parameter accounting for the candidate geometry.
- Train candidate tokenizer(s) on a representative corpus sample and measure bytes/character per token, sequence inflation versus GPT-2, special-token behavior, and downstream evaluation compatibility.
- Prove deterministic document-boundary-preserving detokenization/retokenization or select an authoritative raw-text source path.
- Benchmark grouped/fused many-expert forward/backward execution on the actual target GPUs, recording safe microbatch, target tokens/s, peak reserved memory, and cost per billion target tokens.
- Verify Top-K routing, router gradient flow, load distribution, z-loss/balancing behavior, and dropless/no-token-loss semantics under the selected kernel.
- Establish explicit early quality gates on one continuous training trajectory before allowing it to consume all 100B tokens.

## Links

- [`0154-freeze-first-8-expert-top1-moe-experiment-contract.md`](0154-freeze-first-8-expert-top1-moe-experiment-contract.md)
- [`../reference/dataset_and_tokenization.md`](../reference/dataset_and_tokenization.md)
- [`../current/status.md`](../current/status.md)
- [`../current/roadmap.md`](../current/roadmap.md)
