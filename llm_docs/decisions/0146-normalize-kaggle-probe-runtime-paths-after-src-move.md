---
status: accepted
date: 2026-09-04
owners: [Small-LLM]
---

# ADR 0146: normalize Kaggle probe runtime paths after the src move

## Context

The consolidated 100M/10B probe launcher from ADR 0144 failed on Kaggle before checkpoint restore with:

`RuntimeError: expected provider-neutral runtime .../kaggle/beam/runtime.py, got .../kaggle/src/runtime.py`.

The `kaggle/src/` reorganization changed the filesystem depth of `deep_decay_10b_from_15500_impl.py`, while that historical implementation still derives repository paths from its old location. Its internal `ROOT`, `KAGGLE`, and `BEAM` values therefore point one directory too low. Python can additionally retain a previously imported module named `runtime`, so changing `sys.path` alone is insufficient when `kaggle/src/runtime.py` is already cached.

The pip dependency-conflict warnings printed immediately beforehand are not the cause of this failure: the isolated Hugging Face Hub installation completed and the probe process successfully re-executed before the runtime-path exception.

## Decision

- `kaggle/probes_100m_10b.py` is the stable operator entrypoint for the ADR-0144 probe family.
- The scientific implementation remains consolidated in `kaggle/src/probes_100m_10b.py`; no new scientific probe implementation is introduced.
- Before delegating, the public wrapper normalizes the imported deep-decay implementation to:
  - repository root = actual repository root;
  - Kaggle implementation directory = `kaggle/src`;
  - provider-neutral runtime directory = repository `beam/`.
- If the generic module name `runtime` is already cached from a path other than `beam/runtime.py`, the wrapper removes that cache entry before the deep-decay helper resolves the provider-neutral runtime.
- Operators should run `python kaggle/probes_100m_10b.py ...`, not the lower-level `kaggle/src/probes_100m_10b.py` directly.
- The consolidated static test is named `tests/test_kaggle_probes_100m_10b.py` and must assert this path-normalization contract.

## Consequences

The 100M/10B probes keep one scientific implementation while gaining a stable Kaggle-facing launcher that is insulated from the earlier source-tree move. The existing low-LR experiment semantics, source-checkpoint policy, W&B identities, and no-HF-publication policy are unchanged.
