# SFT Curriculum Sequence Decision

_Last updated: 2026-08-06 Europe/Rome_

## Status

**Frozen curriculum direction.** Exact datasets, token budgets, mixture weights, teacher identity, optimizer values, response-length distributions, and numeric evaluation gates remain to be frozen separately.

## Decision

The first post-training qualification will use three increasingly demanding supervised stages.

### S0 — Open-data chat and instruction SFT

Train on a comparatively broad but capacity-appropriate mixture of question-answering, instruction-following, rewriting, summarization, extraction, formatting, and ordinary conversational examples, together with a separately measured slice of original-distribution pretraining replay.

The purpose is to teach the interaction protocol and basic helpful behavior, not advanced reasoning.

### S1 — Concise teacher-response distillation

Use an approved teacher whose model license, provider terms, and output terms explicitly permit training and distillation.

For the first controlled experiment, reuse the frozen S0 prompt identities, data split, task mixture, token budget, replay policy, serialization, optimizer-selection protocol, and evaluation. Replace selected open-data assistant responses with teacher responses that provide:

- the answer directly;
- at most a minimal, useful justification when the task benefits from one;
- no long chain-of-thought monologue;
- no unnecessary sections or verbosity;
- language and difficulty appropriate for the student model.

S1 must initially branch from the same immutable base checkpoint as S0. It must not silently continue training the S0 checkpoint, because the first scientific question is whether teacher-produced labels outperform the open-data labels under otherwise matched conditions.

After the S0-versus-S1 comparison, a separate cumulative S0-to-S1 run may be evaluated as a curriculum ablation.

### S2 — Concise verified reasoning distillation

After S1 passes, introduce a smaller and more demanding subset whose responses contain an explicit, concise reasoning path that supports the final answer.

The reasoning must be:

- correct and verifiable where programmatic verification is possible;
- materially useful rather than filler;
- capacity-aligned to the student;
- short enough to avoid training the model to imitate verbose reasoning style instead of solving the task;
- followed by a clear final answer.

The implementation should enforce response limits in tokenizer tokens rather than words. The current candidate ceiling is **128 generated response tokens**, not 128 words. This cap remains proposed until the exact tokenizer-length distribution is measured on the selected dataset.

S2 is not permission to train long private-style chain-of-thought traces. It teaches short visible explanations or rationales. Long-CoT training remains deferred to a later, stronger base model and a separate controlled experiment.

## Scientific comparison contract

The minimum clean comparison is:

```text
immutable base checkpoint
  |-- S0: open-data responses
  |-- S1: teacher responses on matched prompts
  `-- later ablation: S0 checkpoint -> additional teacher-response training
```

Only after S1 is selected should the project build S2 as a more advanced reasoning curriculum. This prevents chat acquisition, response-quality distillation, and reasoning transfer from being changed simultaneously.

## Terminology

Teacher-generated prompt-response training with ordinary cross-entropy is already **offline response distillation through SFT**. It does not require a separate logit-distillation objective.

The visible explanations used in S1 and S2 are concise rationales. They should not be treated as guaranteed representations of the teacher's internal reasoning.

## Still open

The following are not frozen by this decision:

1. Exact S0 source datasets and pinned revisions.
2. Exact S0 target-token budget.
3. Exact pretraining-replay ratio and ablations.
4. Exact teacher model and execution provider.
5. Teacher prompt, decoding parameters, candidate count, and selection rule.
6. Exact S1 response-length target.
7. Whether the S2 ceiling remains 128 response tokens after measurement.
8. Exact S2 task taxonomy and verifier set.
9. Whether the final adopted production path is direct S1, cumulative S0-to-S1, or another measured variant.
