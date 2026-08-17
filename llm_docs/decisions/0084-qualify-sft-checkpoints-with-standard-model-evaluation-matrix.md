---
status: accepted
date: 2026-08-15
supersedes: null
---

# 0084 — Qualify SFT checkpoints with the standard model evaluation matrix

## Context and problem statement

The completed 100M/2B S0 SFT checkpoint has a comprehensive post-SFT qualification bundle, but that bundle's qualitative generation settings are not the same as the project's frozen model-level comparison protocols. In particular, the SFT behavior suite uses its own 48/64-token generation budgets and the base qualitative section uses native per-prompt generation budgets. Those results are useful SFT-specific evidence, but they do not replace the canonical ADR 0025 greedy 32-token comparison or the supplementary ADR 0059 sampled comparison.

The user wants the 100M/2B SFT checkpoint, and future comparable SFT checkpoints, to be evaluated with the same standard model-level tests normally used for pretrained checkpoints, while retaining the SFT-specific qualification.

## Decision outcome

For the completed 100M/2B SFT checkpoint, run and retain the following evaluation matrix:

1. **Canonical ADR 0025 qualitative comparison** over the full 18-prompt set with `temperature=0`, `top_p=1`, `top_k=0`, `seed=17`, one sample per prompt, `questions_only=false`, a global `max_new_tokens=32`, `trace_top_tokens=0`, and CUDA FP16.
2. **Supplementary ADR 0059 sampled comparison** over the full prompt set with `temperature=1.0`, `top_k=20`, `top_p=0.9`, `seed=17`, one sample per prompt, `questions_only=false`, and each prompt's native full-suite generation budget rather than the 32-token cap.
3. **Full frozen `eval_core_v1` intrinsic evaluation**, requiring the canonical manifest identity and recording the ordinary loss/perplexity/BPB/top-k/calibration/cluster/position/bootstrap/runtime metrics.
4. **Teacher-forced held-out confidence diagnostic** over the frozen validation sample.
5. **Existing comprehensive post-SFT qualification**, including masked SFT validation/test loss, the deterministic chat-template instruction-behavior suite, and its base qualitative outputs.

The exact greedy-32 result remains the canonical cross-checkpoint qualitative comparison. The wider sampled run is supplementary and must never overwrite or be merged into the greedy score. The SFT-specific instruction-behavior suite is a separate chat-template measurement and must also remain separate from the base-model-style qualitative prompts.

Where the same exact `eval_core_v1` checkpoint/manifest result has already been produced and verified as part of the comprehensive post-SFT qualification, tooling should avoid unnecessary recomputation when it can safely reuse the verified result. Any reuse must preserve checkpoint identity, manifest SHA-256, precision, suite, and metric provenance in the output.

## Implementation guidance

Prefer a thin orchestration layer over duplicating metric or sampler implementations. Reuse the existing canonical evaluator, prompt-suite, teacher-forced diagnostic, and SFT qualification entrypoints. The orchestrator should fail closed on checkpoint identity or eval-manifest mismatch, write separate machine-readable artifacts for each protocol, and produce a small summary/index that makes the separation between canonical greedy, supplementary sampled, intrinsic, teacher-forced, and SFT-specific results explicit.

## Consequences

- SFT checkpoints remain directly comparable with the project's pretrained model endpoints under ADR 0025 and ADR 0059.
- SFT instruction acquisition remains measured with the dedicated chat-template behavior suite rather than inferred from base-model prompt continuations.
- Re-running a qualification cannot silently substitute native 48–128-token generation for the frozen 32-token greedy comparison.
- Full qualification is more comprehensive, but orchestration should reuse verified intrinsic results when safe to avoid redundant GPU work.

## Links

- [`0025-freeze-canonical-full-post-pretraining-prompt-suite.md`](0025-freeze-canonical-full-post-pretraining-prompt-suite.md)
- [`0033-use-comprehensive-post-sft-qualification-and-pretraining-cadence.md`](0033-use-comprehensive-post-sft-qualification-and-pretraining-cadence.md)
- [`0059-run-supplementary-sampled-three-way-full-evaluation.md`](0059-run-supplementary-sampled-three-way-full-evaluation.md)
- [`../runbooks/post_pretraining_prompt_suite.md`](../runbooks/post_pretraining_prompt_suite.md)
- [`../reference/training_and_evaluation.md`](../reference/training_and_evaluation.md)
