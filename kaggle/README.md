# Kaggle workspace

This directory contains the Kaggle-side launchers, training/evaluation runtimes, publication helpers, and environment files for the Small-LLM project.

## Layout

- `src/` contains the Python implementation files and runnable experiment scripts. It is intentionally flat for now because several modules use peer imports.
- `env/` contains Kaggle/Hugging Face publication environment examples and requirements.
- root-level `launch.py`, `launch_sft.py`, `launch_r_sft.py`, and `probes_100m_10b.py` are stable wrappers that forward to `src/`.

## Stable operator surface

Use the root wrappers for normal launcher discovery and operation:

```bash
python kaggle/launch.py --help
python kaggle/launch_sft.py --help
python kaggle/launch_r_sft.py --help
python kaggle/probes_100m_10b.py --help
```

Lower-level files under `src/` are implementation, diagnostics, or reproduction entrypoints and should be called directly only when their current documentation says to do so. The active 100M/10B post-completion pretraining diagnostic is consolidated behind the stable probe launcher:

```bash
python kaggle/probes_100m_10b.py --dry-run
python kaggle/probes_100m_10b.py --probe active
```

The wrapper normalizes repository paths after the `kaggle/src/` reorganization before delegating to `src/probes_100m_10b.py`, ensuring the continuation code resolves the provider-neutral `beam/runtime.py` and the canonical Kaggle dual-T4 wrapper. The launcher owns the current constant-`1e-5` and constant-`2e-5` hold probes. The earlier Probe A files were merged under ADR 0144 and should not be reintroduced as parallel scientific probe implementations.

See [`src/README.md`](src/README.md) for the current source-file groups.

When adding new Kaggle code, prefer `kaggle/src/` for Python, `kaggle/env/` for dependency or environment files, and keep the root reserved for stable launch wrappers plus this index.
