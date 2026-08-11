---
status: accepted
date: 2026-08-11
---

# 0036 — Build eval_core_v1 once and reuse it across compatible budgets

## Context

The permanent `eval_core_v1` corpus is defined entirely by the frozen ClimbMix source revision, split identity, retained cluster set, per-cluster quotas, tokenizer geometry, and selection order. It does not depend on model weights, checkpoint identity, pretraining token budget, or SFT state. Building it by rescanning the pinned source during a GPU evaluation wastes accelerator time.

The project currently needs the same frozen base-distribution qualification suite for the 20M model trained to 500M tokens and for the 20M model trained to 2B tokens.

## Decision

Build and verify one production `eval_core_v1` artifact on a CPU/network machine, persist that immutable artifact remotely, and reuse the exact same verified corpus for both the 500M and 2B parent/SFT qualification runs.

The canonical standalone command is:

```bash
uv run --python 3.13 small-llm-eval-data build --output-dir ./eval_core_v1
```

The command uses the accelerated deterministic scanner and immediately runs the frozen verifier. No model checkpoint or GPU is required for the build.

A second build is required only if a frozen corpus-defining input changes, such as the source revision, split seed/hash version, accepted cluster policy, context geometry, tokenizer/vocabulary contract, or eval-core quotas/schema. A different model checkpoint or pretraining budget alone does not justify rebuilding the corpus.

## Consequences

- The 500M and 2B qualification runs are directly comparable on the same base-distribution examples.
- GPU evaluation sessions only verify/load the persisted corpus and score checkpoints.
- The one-time source scan can be performed on an inexpensive CPU host with good network connectivity.
- The verified artifact should be published once and treated as immutable.

## Links

- `dataset/eval_core.py`
- `dataset/eval_core_accelerated.py`
- `dataset/eval_core_cli.py`
- `trainer/eval_entrypoint.py`
- `0035-accelerate-and-persist-eval-core-v1.md`
