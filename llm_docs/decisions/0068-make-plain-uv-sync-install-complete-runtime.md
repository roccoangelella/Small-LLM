---
status: accepted
date: 2026-08-13
---

# ADR 0068: Make plain `uv sync` install the complete project runtime

## Context

The repository split ordinary runtime dependencies across optional extras and standalone setup commands. That allowed `beam/launch.py` to exist on `main` while a normal project environment created by plain `uv sync` omitted the Beam SDK and failed immediately with `ModuleNotFoundError: No module named 'beam'`. The committed `uv.lock` had also drifted behind the dependency declarations.

## Decision

Plain `uv sync` is the canonical local and VPS environment setup for Small-LLM.

The default uv environment must include the dependencies required for model execution, FLA GDN-2, Beam launching, Hugging Face transport, tokenization/post-training utilities, and W&B telemetry. Beam remains pinned to `beam-client==0.2.201`. Existing optional extras remain available as compatibility aliases for older commands, but they are not required to obtain the complete canonical runtime.

The project uses uv's default `runtime` dependency group for remote/post-training/logging packages that need not be exported as base package requirements. Because the group is listed in `tool.uv.default-groups`, a plain `uv sync` installs it automatically together with the base project dependencies.

Every dependency-contract change must regenerate and commit `uv.lock`. `uv sync --locked` is the fail-closed reproducibility check; it must succeed on a clean checkout when the lock and project metadata are current.

## Consequences

A fresh checkout no longer needs one-off `uv pip install`, `pip install`, or explicit project extras before invoking the supported launchers. The canonical operator setup is one command: `uv sync`.

Dataset-only environments may install more packages than before. This is an accepted tradeoff for a reproducible project environment and for preventing provider launchers from failing because their SDK was omitted from the default sync.
