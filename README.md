# Small-LLM

Small-LLM is a learning and research project to build a modern English decoder-only language model below 1B parameters from random initialization.

The current model family is a dense hybrid:

```text
[GDN-2, GDN-2, GDN-2, gated full attention] x N
```

## Current stage

The approximately-20M-parameter model has completed the 100M- and 500M-token scaling points. The active pretraining experiment is the fresh approximately-2B-token point, using the same model geometry with the qualified mixed-FLA GDN-2 CUDA backend from update 1. SFT qualification on the completed 500M parent is authorized in parallel.

Current state and next gates live in:

- [`llm_docs/current/status.md`](llm_docs/current/status.md)
- [`llm_docs/current/roadmap.md`](llm_docs/current/roadmap.md)

## Repository map

```text
dataset/      consolidated deterministic corpus/profile/verification/eval tooling
model/        geometry-scalable GDN-2 hybrid decoder
trainer/      optimization, checkpointing, evaluation, and generation
kaggle/       current T4 launch and publication entry points
tests/        offline correctness and repository-contract tests
journals/     informal study notes
llm_docs/     authoritative project memory
```

The old decoded-text/Gemini curation pipeline, per-budget dataset qualification wrappers, and completed one-off 20M qualification launchers have been removed. Their accepted results remain under `llm_docs/evidence/` or `llm_docs/archive/`; Git history preserves the old implementations.

## Main commands

Install the complete supported project runtime, including model execution, Hugging Face transport, W&B, post-training utilities, and the Beam SDK:

```bash
uv sync
```

Use `uv sync --locked` when you want a fail-closed check that `pyproject.toml` and the committed `uv.lock` already agree.

Inspect the frozen finite-dataset profiles:

```bash
python -m dataset.qualification profiles
```

Launch or exactly resume the active 20M-model/2B-token Kaggle run:

```bash
python kaggle/launch.py train --model 20M --tokens 2B
```

Build/verify/publish a fixed finite dataset through the same profile registry:

```bash
python kaggle/launch.py publish --model 20M --tokens 2B
```

Build and run the frozen evaluation suite:

```bash
small-llm-eval-data build --output-dir /data/eval_core_v1
small-llm-eval fast --eval-dir /data/eval_core_v1 --output-json artifacts/eval-fast.json
small-llm-eval full --eval-dir /data/eval_core_v1 --output-json artifacts/eval-full.json
```

Run the offline repository tests:

```bash
uv run python -m unittest discover -v
```

## Documentation

[`llm_docs/README.md`](llm_docs/README.md) is the documentation map. It separates current state, decisions, reference material, runbooks, research, completed evidence, and superseded material so agents and humans do not have to load one large, stale manual.
