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

# ADR 0140: wire evaluation v2 and retire the fixed-length qualitative protocol

## Decision

The project now uses evaluation v2 for pretrained and SFT checkpoints.

### Pretraining

Canonical full pretraining qualification contains three primary evidence layers:

1. frozen `eval_core_v1` intrinsic language-model metrics;
2. the six-task L20-Edu-style zero-shot benchmark — ARC-Challenge, ARC-Easy,
   HellaSwag, LAMBADA OpenAI, PIQA and WinoGrande — scored strictly by
   conditional log likelihood through `lm-evaluation-harness==0.4.12`;
3. an expanded native-budget base-prompt suite with 100 mechanically scored
   prompts and 20 readable qualitative continuations.

The L20 adapter rejects rolling-likelihood and generation requests. Full
qualification uses complete task datasets; the `fast` mode may cap each
external task at 100 examples and is diagnostic only.

EOS termination is not a pretraining metric. Teacher-forced confidence remains
a separate diagnostic and is not part of the headline qualification.

Greedy qualitative decoding is `temperature=0`, `top_p=1`, `top_k=0`.
Sampled qualitative decoding is `temperature=1`, `top_p=1`, `top_k=0`.
Prompt generation uses each case's native budget.

### SFT

SFT Behavior v2 is now wired as the primary instruction-following benchmark.

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

The old global fixed-length qualitative cap is retired from all active
pretraining, SFT and R-SFT evaluation paths and from project documentation.
Historical evidence is described only as coming from the retired fixed-length
protocol; it is not a current reproducibility target.

R-SFT qualitative regressions also use native prompt budgets. Their normal
sampled qualitative view uses the same project-wide
`temperature=1`, `top_p=1`, `top_k=0` contract. Reasoning-specific pass@1
sampling remains a separate task-specific protocol.

## Human-readable JSON contract

Both canonical pretraining and SFT JSON reports must be understandable without
reading evaluator source code.

They therefore begin with:

- `read_me_first`: purpose, interpretation and metric directions;
- `headline_summary`: the main values and parent/SFT deltas;
- `protocol`: exact scoring/decoding contracts.

Detailed per-task/per-case evidence follows those sections. Reports do not use
alphabetical key sorting because semantic reading order is intentional.

## Dependency boundary

`lm-evaluation-harness` is evaluation-only and pinned in
`requirements-eval.txt`. It is not added to the training dependency lock, so
benchmark tooling cannot silently alter training environments.

## Consequences

- External capability comparisons gain substantially larger sample counts than
  the old qualitative probe.
- SFT evaluation can distinguish “cannot solve the task” from “can solve it but
  fails the instruction”.
- Pretraining is no longer penalized for not behaving like a chat model.
- Existing `eval_core_v1` continuity is preserved.
- Legacy SFT v1 numbers remain comparable, but v2 held-out conditional
  compliance becomes the primary instruction-following evidence.
