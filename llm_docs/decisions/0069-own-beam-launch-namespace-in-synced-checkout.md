---
status: accepted
date: 2026-08-13
---

# ADR 0069: Own the `beam.launch` namespace in the synced checkout

## Context

Beam derives a remote Function handler module from the defining source path. Running the canonical launcher as `python beam/launch.py ...` therefore registers handlers such as `beam.launch:remote_import_preflight`.

The Beam worker prepends the synced user-code directory and then imports that handler module with Python import machinery. Before this decision, the repository's `beam/` directory had no `__init__.py`, so `beam.launch` resolved through Beam's installed top-level SDK package instead of the Small-LLM adapter. The installed package has no `launch` submodule, causing `ModuleNotFoundError: No module named 'beam.launch'` before the preflight function body could run. The controller then received no result and produced a secondary `NoneType` unpacking error.

## Decision

Keep `python beam/launch.py ...` as the canonical Beam operator command and keep the existing launcher/training implementation in place.

Add `beam/__init__.py` so the synced checkout is an explicit Python package for the `beam` namespace. That package forwards the launcher SDK symbols `Image`, `Volume`, and `function` to Beam's underlying `beta9` package. Consequently, when the remote worker imports `beam.launch`, the repository package resolves first and `beam/launch.py` becomes importable without shadowing away the SDK objects used by the launcher.

Add a regression test that uses `PathFinder` against the repository root to prove that `beam` resolves to the project bridge and `beam.launch` resolves to the project launcher. The test also guards the required `beta9` symbol forwarding.

Do not rename the launcher, mutate Beam's private handler metadata, or introduce a second canonical CLI solely to avoid the namespace collision.

## Consequences

The existing Beam command remains stable for operators and remote handlers can import before any CPU or GPU work begins. The fix is isolated to Python package resolution and does not change model geometry, dataset staging, checkpointing, W&B identity, GPU selection, or the scientific launch gate.

The project now intentionally owns the `beam` namespace whenever the synced repository root is first on `sys.path`. If future adapter code imports additional top-level Beam SDK symbols through `from beam import ...`, `beam/__init__.py` must forward those symbols as well.
