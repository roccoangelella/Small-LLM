# Kaggle workspace

This directory contains the Kaggle-side launchers, training/evaluation runtimes, publication helpers, and environment files for the Small-LLM project.

## Layout

- `src/` contains the Python implementation files and runnable experiment scripts. It is intentionally flat for now because several modules use peer imports.
- `env/` contains Kaggle/Hugging Face publication environment examples and requirements.
- root-level `launch.py`, `launch_sft.py`, and `launch_r_sft.py` are compatibility wrappers that forward to `src/`.

## Common commands

Existing launcher commands are preserved:

```bash
python kaggle/launch.py --help
python kaggle/launch_sft.py --help
python kaggle/launch_r_sft.py --help
```

For lower-level scripts, call the moved file directly:

```bash
python kaggle/src/qualify_dual_t4.py --help
python kaggle/src/build_and_push_100m.py --help
python kaggle/src/run_20m_one_click.py --help
```

When adding new Kaggle code, prefer `kaggle/src/` for Python, `kaggle/env/` for dependency or environment files, and keep the root reserved for stable launch wrappers plus this index.
