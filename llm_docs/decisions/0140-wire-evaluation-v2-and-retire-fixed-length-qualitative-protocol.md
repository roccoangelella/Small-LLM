---
status: accepted
date: 2026-09-03
owners: [Small-LLM]
supersedes:
  - 0024-freeze-canonical-questions-only-prompt-test-settings
  - 0025-freeze-canonical-full-post-pretraining-prompt-suite
implements:
  - 0138-approve-sft-behavior-v2-design-without-wiring
---

# ADR 0140: stage evaluation v2 and retire the fixed-length qualitative protocol

## Decision

Adopt evaluation v2 as the next project evaluation target and stage its first
implementation modules in-repository, while keeping the final active evaluator
entrypoint wiring as a follow-up patch that must receive a local test pass
before it replaces the current production paths.

This ADR records the durable protocol decision and the repository staging work
completed on 2026-09-03. It intentionally avoids pretending that the large
in-place evaluator rewrite was pushed if it has not yet been safely landed.

### Pretraining target

Canonical full pretraining qualification should contain three primary evidence
layers:

1. frozen `eval_core_v1` intrinsic language-model metrics;
2. the six-task L20-Edu-style zero-shot benchmark — ARC-Challenge, ARC-Easy,
   HellaSwag, LAMBADA OpenAI, PIQA and WinoGrande — scored strictly by
   conditional log likelihood through `lm-evaluation-harness==0.4.12`;
3. an expanded native-budget base-prompt suite with 100 mechanically scored
   prompts and 20 readable qualitative continuations.

The staged L20 adapter rejects rolling-likelihood and generation requests. Full
qualification should use complete task datasets; `fast` mode may cap each
external task at 100 examples and is diagnostic only.

EOS termination is not a pretraining metric. Teacher-forced confidence remains
a separate diagnostic and is not part of the headline qualification.

Greedy qualitative decoding is `temperature=0`, `top_p=1`, `top_k=0`.
Sampled qualitative decoding is `temperature=1`, `top_p=1`, `top_k=0`.
Prompt generation uses each case's native budget.

### SFT target

SFT Behavior v2 is the accepted next primary instruction-following benchmark.

It contains 180 semantic tasks across six balanced families. Every semantic
task is evaluated at four paired levels:

- L0: underlying capability without formatting constraints;
- L1: answer-only constraint;
- L2: exact one-line `Answer: <answer>` format;
- L3: exact two-line multi-constraint format.

This produces 720 cases: 480 diagnostic and 240 held-out qualification cases.
Conditional L1/L2/L3 compliance is calculated only for semantic tasks whose L0
answer was correct, so capability failure is not misreported as
instruction-following failure.

Greedy decoding is primary. Sampled robustness uses
`temperature=1`, `top_p=1`, `top_k=0` with seeds 17, 18 and 19. Parent and SFT
results are paired by exact case ID and report wins, losses, ties and an exact
two-sided McNemar/binomial p-value.

The legacy 30-case SFT behavior suite remains only for longitudinal comparison.
Masked SFT validation/test losses remain teacher-forced diagnostics.

### Fixed-length qualitative protocol

The old global fixed-length qualitative cap is retired as a future active
protocol target. Historical evidence may still describe runs that used it, but
new active qualification work should use native per-case budgets.

R-SFT qualitative regressions should also use native prompt budgets. Their
normal sampled qualitative view should use the same project-wide
`temperature=1`, `top_p=1`, `top_k=0` contract. Reasoning-specific pass@1
sampling remains a separate task-specific protocol.

## Human-readable JSON contract

Canonical pretraining and SFT JSON reports should be understandable without
reading evaluator source code.

They should therefore begin with:

- `read_me_first`: purpose, interpretation and metric directions;
- `headline_summary`: the main values and parent/SFT deltas;
- `protocol`: exact scoring/decoding contracts.

Detailed per-task/per-case evidence should follow those sections. Reports
should not use alphabetical key sorting when semantic reading order matters.

## Dependency boundary

`lm-evaluation-harness` is evaluation-only and pinned in
`requirements-eval.txt`. It is not added to the training dependency lock, so
benchmark tooling cannot silently alter training environments.

## Landed in this staging commit

- `post_training/sft/behavior_v2.py`: additive Behavior v2 suite/scoring module.
- `trainer/pretraining_eval_v2.py`: additive L20/base-prompt v2 module.
- `requirements-eval.txt`: isolated evaluation dependency pin.
- Current status/reference/runbook updates describing the v2 target and the
  retirement of the fixed-length qualitative protocol as an active target.

## Not landed yet

- Replacing `trainer.eval_suite`, `post_training.sft.eval_suite`, and the R-SFT
  evaluator entrypoints with the full v2 report contract.
- Updating future SFT decontamination to include all Behavior v2 prompts.
- Running GPU/local end-to-end qualification over real checkpoints.

## Consequences

- External capability comparisons gain a staged path toward substantially larger
  sample counts than the old qualitative probe.
- SFT evaluation gains a staged way to distinguish “cannot solve the task” from
  “can solve it but fails the instruction”.
- Pretraining should not be penalized for not behaving like a chat model.
- Existing `eval_core_v1` continuity is preserved.
- Legacy SFT v1 numbers remain comparable until the v2 entrypoint wiring is
  safely landed.
