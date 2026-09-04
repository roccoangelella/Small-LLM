# Kaggle source files

Python files in this directory are kept in one flat namespace to avoid breaking peer imports such as `from sft_cli import ...`, `from runtime import ...`, or direct script execution from Kaggle cells.

Logical groups:

- launch/orchestration: `launch.py`, `launch_sft.py`, `launch_r_sft.py`
- pretraining and data-scaling runs: `dual_t4_train*.py`, `run_20m*.py`, `deep_decay_10b_from_15500*.py`, `probes_100m_10b.py`
- SFT and R-SFT: `sft_*.py`, `rsft_*.py`, `dual_t4_sft*.py`, `dual_t4_rsft.py`
- evaluation/qualification: `qualify_dual_t4*.py`, `rsft_eval_runtime.py`
- publication and infrastructure: `build_and_push_100m.py`, `sft_publish.py`, `wandb_preflight.py`, `triton_cache.py`

`probes_100m_10b.py` is the single public home for short 100M/10B pretraining probes. New probes should be added there rather than as new one-off launch files.

A deeper package split should be done only together with import rewrites and smoke tests.
