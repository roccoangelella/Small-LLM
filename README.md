# Small-LLM

Small-LLM is a learning and research project to build a modern English decoder-only language model below 1B parameters from random initialization.

The current model family is a dense hybrid:

```text
[GDN-2, GDN-2, GDN-2, gated full attention] x N
```

## Current stage

The 20M scaling series through 2B target tokens is complete, as are the 100M/2B and 100M/10B pretraining points. The current 100M/10B pretrained endpoint is `step-00076294` at 10,000,007,168 consumed target tokens.

Active work is now post-pretraining: the 100M/10B S0 SFT trajectory is using the frozen same-data recipe recorded in the project decisions, and ADR 0144 defines two short post-completion 100M/10B pretraining probes that hold LR at `1e-5` and `2e-5` to test whether the tail plateau came from terminal decay rather than model/data saturation.

Current state and next gates live in:

- [`llm_docs/current/status.md`](llm_docs/current/status.md)
- [`llm_docs/current/roadmap.md`](llm_docs/current/roadmap.md)

## Repository map

```text
dataset/      consolidated deterministic corpus/profile/verification/eval tooling
model/        geometry-scalable GDN-2 hybrid decoder
trainer/      optimization, checkpointing, evaluation, and generation
kaggle/       T4 launch, SFT, diagnostic, evaluation, and publication entry points
beam/         alternate Beam provider adapter and reproduction procedures
modal/        Modal provider adapter and reproduction procedures
tests/        offline correctness and repository-contract tests
journals/     informal study notes
llm_docs/     authoritative project memory
```

The old decoded-text/Gemini curation pipeline, per-budget dataset qualification wrappers, and completed one-off 20M qualification launchers have been removed. Their accepted results remain under `llm_docs/evidence/` or `llm_docs/archive/`; Git history preserves the old implementations.

## Main commands

Install the complete supported project runtime:

```bash
uv sync
```

Use `uv sync --locked` when you want a fail-closed check that `pyproject.toml` and the committed `uv.lock` already agree.

Inspect the frozen finite-dataset profiles:

```bash
python -m dataset.qualification profiles
```

Inspect the stable Kaggle launcher surfaces:

```bash
python kaggle/launch.py --help
python kaggle/launch_sft.py --help
python kaggle/launch_r_sft.py --help
```

Inspect the current 100M/10B post-completion pretraining probes without allocating a GPU:

```bash
python kaggle/src/probes_100m_10b.py --dry-run
```

Build the frozen evaluation corpus and inspect the active evaluation-v2 entrypoints:

```bash
small-llm-eval-data build --output-dir /data/eval_core_v1
small-llm-eval --help
small-llm-sft-eval --help
```

Full pretraining evaluation v2 uses the separately pinned `lm-evaluation-harness==0.4.12` dependency from `requirements-eval.txt`; evaluation dependencies are intentionally kept outside the training lock.

Run the offline repository tests:

```bash
uv run python -m unittest discover -v
```

## Documentation

[`llm_docs/README.md`](llm_docs/README.md) is the documentation map. It separates current state, decisions, reference material, runbooks, research, completed evidence, and superseded material so agents and humans do not have to load one large, stale manual.
