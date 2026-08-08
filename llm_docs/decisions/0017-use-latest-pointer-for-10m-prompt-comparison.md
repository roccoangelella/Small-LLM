---
status: accepted
date: 2026-08-08
supersedes: null
---

# 0017 — Use latest pointer for 10M prompt comparison

## Context and problem statement

The historical approximately-10M-token qualification run (`20m-qualification-dataset-001`) predates the later validation-selected `best.json` publication convention. Its Hugging Face checkpoint tree exposes a `latest.json` pointer but no `best.json`, so the post-pretraining prompt suite fails when left at its default `--pointer best` for that run.

For this completed one-pass qualification, the terminal published checkpoint is the intended checkpoint for qualitative comparison against the later approximately-100M-token run.

## Considered options

- Reconstruct or publish a synthetic `best.json` pointer for the historical run.
- Use the existing `latest.json` pointer for the 10M run.
- Exclude the 10M run from prompt-level comparisons.

## Decision outcome

Chosen option: **use `--pointer latest` when running post-pretraining prompt diagnostics on `20m-qualification-dataset-001`**. Keep the normal validation-selected `best` pointer convention for newer runs that actually publish `best.json`.

## Consequences

### Positive

- The historical 10M checkpoint can be evaluated without mutating archived checkpoint metadata.
- The 10M and 100M runs can be compared with the same prompt/decoding configuration.
- The comparison remains explicit about the different pointer semantics of the old run.

### Negative or limiting

- `latest` and `best` are not generally interchangeable; this exception is specific to the historical 10M run.
- Future comparisons must preserve the pointer used in their evaluation artifacts.

## Validation

The post-pretraining prompt suite should successfully resolve `run/20m-qualification-dataset-001/latest.json`, verify and load the referenced checkpoint, and complete the canonical six-case short greedy diagnostic.

## Links

- `../runbooks/post_pretraining_prompt_suite.md`
- `../current/status.md`
- `../archive/20m_qualification/20m_kaggle_runbook.md`
