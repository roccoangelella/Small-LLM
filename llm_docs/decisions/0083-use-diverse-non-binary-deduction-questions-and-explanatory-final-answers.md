---
status: accepted
last_reviewed: 2026-08-14
---

# ADR 0083 — Use diverse non-binary deduction questions and explanatory final answers

## Decision

The R-SFT reasoning dataset will not collapse deductive supervision into mostly yes/no questions or one-token final answers.

For each reasoning family, generated problems should use a meaningful variety of answer forms appropriate to the underlying task. For deduction this includes, when logically suitable, selecting or naming the entity/conclusion that follows, stating what must be true, stating what cannot be true, identifying a classification or consequence, determining whether a conclusion follows, identifying a contradiction, and other natural forms beyond binary yes/no.

Binary questions remain allowed when they are the natural form of the logical problem, but they must not dominate the dataset.

The student-facing final answer should be a concise explanatory conclusion rather than merely `Yes`, `No`, or another bare label whenever a short explanation is useful. The full `reasoning` field remains the concise derivation; the `answer` field should give a natural user-facing conclusion that makes the result intelligible without needlessly repeating the entire reasoning trace.

For example, instead of:

```json
{"answer": "Yes."}
```

prefer a final answer such as:

```json
{"answer": "Yes. The backup generator must be active because the closed skylight rules out natural light, so the emergency lamps must be in use."}
```

The exact amount of repetition between `reasoning` and `answer` should remain concise and natural; the answer is not required to restate every intermediate inference.

## Prompt implication

Gemini teacher prompts should explicitly ask for diversity in the *form of the question and answer*, not only diversity in subject matter and logical structure. Prompts should avoid leading examples that make every generated item look like a yes/no classification task.

The structured teacher output remains conceptually:

```json
{
  "problem": "...",
  "reasoning": "...",
  "answer": "..."
}
```

where `reasoning` contains the concise complete derivation and `answer` contains the concise user-facing conclusion.

If later deterministic verification requires a canonical machine-readable target distinct from the natural-language answer, that verifier target should be stored as separate metadata rather than forcing the student-facing answer back into a bare label.

## Rationale

The goal of R-SFT is to teach reasoning behavior and explanatory conclusions, not merely map prompts to binary labels. Rationale-based distillation and concise-CoT work in 2026 explicitly train compact models on generated rationales rather than only final outcomes, while emphasizing that those rationales should retain essential reasoning rather than unnecessary verbosity.

A bare final label can remain useful for automated evaluation, but it should not dictate the language that the student is trained to produce for users.

An explanatory answer is useful behavioral evidence that the model can express the inference it performed, but it must not be treated as proof that the visible explanation is a faithful readout of hidden internal cognition.

## Relationship to previous decisions

ADR 0081 still keeps internal difficulty labels and step counts out of Gemini prompts. ADR 0082 still uses batched generation, one positive example, and lightweight diversity guidance. This ADR adds answer-form diversity and explanatory final answers to that prompting contract.
