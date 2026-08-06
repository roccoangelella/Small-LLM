# Small-LLM

Small-LLM is a learning and research project to build a modern English decoder-only language model below 1B parameters from random initialization.

The current model family is a dense hybrid:

```text
[GDN-2, GDN-2, GDN-2, gated full attention] x N
```

## Current stage

The accepted approximately-20M-parameter model completed its 10M-token engineering run and is now being trained on a fixed approximately-100M-token dataset. The project is staying with the main GDN-2 hybrid during this data-scaling stage; architecture baselines are deferred until larger model versions.

Current state and next gates live in:

- [`llm_docs/current/status.md`](llm_docs/current/status.md)
- [`llm_docs/current/roadmap.md`](llm_docs/current/roadmap.md)

## Repository map

```text
dataset/      deterministic corpus, cache, durability, and eval-set builders
model/        geometry-scalable GDN-2 hybrid decoder
trainer/      optimization, checkpointing, evaluation, and generation
kaggle/       current T4 launch and publication entry points
tests/        offline correctness and repository-contract tests
journals/     informal study notes
llm_docs/     authoritative project memory
```

The old decoded-text/Gemini curation pipeline and completed one-off 20M qualification launchers have been removed. Their accepted results remain under `llm_docs/evidence/`; Git history preserves the old implementations.

## Main commands

Install model and post-training dependencies:

```bash
uv sync --locked --extra model --extra post-training
uv pip install -r dataset/requirements-remote.txt
```

Run the current segmented 20M-model/100M-token experiment on Kaggle:

```bash
python kaggle/run_20m_100m.py
```

Build and run the frozen evaluation suite:

```bash
small-llm-eval-data build --output-dir /data/eval_core_v1
small-llm-eval fast --eval-dir /data/eval_core_v1 --output-json artifacts/eval-fast.json
small-llm-eval full --eval-dir /data/eval_core_v1 --output-json artifacts/eval-full.json
```

Run the offline repository tests:

```bash
uv run --extra model python -m unittest discover -v
```

## Documentation

[`llm_docs/README.md`](llm_docs/README.md) is the documentation map. It separates current state, decisions, reference material, runbooks, research, completed evidence, and superseded material so agents and humans do not have to load one large, stale manual.
