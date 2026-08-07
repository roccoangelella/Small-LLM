---
status: accepted
date: 2026-08-07
supersedes: null
---

# 0010 — Self-provision eval_core_v1 from the main evaluator

## Context and problem statement

The complete post-pretraining evaluation required users to run separate dependency, eval-corpus build, eval-corpus verification, and model-evaluation commands. The frozen `eval_core_v1` corpus is generated data rather than a tracked repository directory, so exposing its lifecycle as a prerequisite made the normal final-model workflow unnecessarily confusing and easy to mis-invoke with placeholder paths.

The corpus build and verification remain important reproducibility boundaries, but they do not need to be separate user-facing steps in the normal final evaluation path.

## Considered options

- Keep `small-llm-eval-data build`, `small-llm-eval-data verify`, and `small-llm-eval` as mandatory sequential commands.
- Track the generated evaluation corpus in the source repository.
- Keep the standalone build/verify tools for debugging, while making `small-llm-eval` ensure the frozen corpus exists, build it if absent, verify it, and then evaluate.

## Decision outcome

Chosen option: **make `small-llm-eval` self-provision `eval_core_v1`**.

The user-facing evaluator now:

1. selects a deterministic eval cache directory;
2. builds the frozen corpus only when that directory is absent;
3. verifies the complete corpus before every evaluation;
4. fails closed if an existing corpus is invalid rather than silently replacing it;
5. runs the existing checkpoint evaluator unchanged after verification.

Default cache location:

- Kaggle: `/kaggle/working/eval_core_v1`;
- other environments: `artifacts/eval_core_v1`;
- override: `SMALL_LLM_EVAL_DIR` or explicit `--eval-dir`.

The standalone `small-llm-eval-data build|verify` commands remain available for debugging, corpus publication, and explicit reproducibility checks, but they are no longer mandatory prerequisites for the normal `small-llm-eval` workflow.

## Consequences

### Positive

- A final model evaluation becomes one user-facing command.
- Users no longer need to know or invent an eval-corpus path before the first run.
- Repeated evaluations reuse the verified frozen corpus instead of rebuilding it.
- The verification boundary is preserved on every run.
- Explicit `--eval-dir` remains available for shared or prebuilt corpora.

### Negative or limiting

- The first evaluation in an environment may spend substantial time building the frozen corpus before model scoring begins.
- Building still requires network access to the pinned source when no cached corpus exists.
- An invalid existing cache fails closed and requires explicit operator cleanup rather than being overwritten automatically.

## Validation

Offline tests must verify that a missing eval directory triggers exactly one build followed by verification, while an existing directory is verified without rebuilding. The console script `small-llm-eval` must route through the self-provisioning entry point.

## Links

- [`../runbooks/eval_core_v1_runbook.md`](../runbooks/eval_core_v1_runbook.md)
- [`../../trainer/eval_entrypoint.py`](../../trainer/eval_entrypoint.py)
- [`../../trainer/eval_suite.py`](../../trainer/eval_suite.py)
