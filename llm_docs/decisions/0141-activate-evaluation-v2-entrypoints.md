---
status: accepted
date: 2026-09-03
owners: [Small-LLM]
implements:
  - 0140-wire-evaluation-v2-and-retire-fixed-length-qualitative-protocol
---

# ADR 0141: activate evaluation v2 entrypoints for pretraining and SFT

## Decision

Wire the staged evaluation-v2 implementation into the active pretrained and SFT
qualification entrypoints.

The normal pretrained checkpoint path now routes `trainer.eval_entrypoint` to
`trainer.eval_suite_v2`. The normal SFT qualification path keeps the historical
module name `post_training.sft.eval_suite`, but its report schema is now
`small-llm-post-sft-qualification-v2`.

## Active pretrained evaluation

The active pretrained JSON begins with `read_me_first` and `headline_summary`,
then reports:

1. frozen `eval_core_v1`;
2. L20-Edu-style zero-shot conditional-likelihood results for ARC-Challenge,
   ARC-Easy, HellaSwag, LAMBADA OpenAI, PIQA and WinoGrande;
3. the expanded base prompt suite with 100 mechanically scored prompts and 20
   qualitative continuations.

Legacy prompt-runner CLI flags are accepted only for command compatibility and
are not used by evaluation v2. The fixed-length qualitative protocol is not
emitted by the active pretrained entrypoint.

## Active SFT evaluation

The active SFT JSON begins with `read_me_first` and `headline_summary`, then
reports:

1. parent and tuned checkpoint metadata;
2. frozen `eval_core_v1`;
3. masked SFT validation/test loss;
4. `instruction_behavior_v2` as the primary instruction-following benchmark;
5. `instruction_behavior_v1_legacy` for longitudinal continuity;
6. `base_prompt_suite_v2` for native-budget qualitative/base-prompt evidence;
7. paired parent-vs-SFT behavior-v2 wins, losses, ties and exact McNemar/binomial
   statistics.

The active SFT evaluator no longer emits `qualitative_greedy_32`.

## Dependency boundary

The L20 external benchmark remains evaluation-only. The harness dependency stays
in `requirements-eval.txt`; it is not added to the training dependency lock.

## Validation

Before pushing, the new Python entrypoint modules were syntax-checked locally.
The active code is wired by import/module path rather than by changing training
logic, so training and checkpointing code remain untouched.

## Consequences

- `small-llm-eval` now produces the pretraining v2 JSON contract.
- `python -m trainer.eval_entrypoint ...` now produces the pretraining v2 JSON
  contract.
- `small-llm-sft-eval` and existing SFT launchers that call
  `post_training.sft.eval_suite` now produce the SFT v2 JSON contract.
- Legacy behavior-v1 numbers remain present but are no longer the primary SFT
  behavioral score.
