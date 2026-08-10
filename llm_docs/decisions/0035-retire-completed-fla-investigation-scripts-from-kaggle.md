---
status: accepted
date: 2026-08-10
supersedes: 0026
---

# 0035 — Retire completed FLA investigation scripts from Kaggle

## Context and problem statement

The `kaggle/` directory was re-audited after the finite-data launch surface and the SFT lane had both been consolidated behind `kaggle/launch.py` / `kaggle/runtime.py` and `kaggle/launch_sft.py` / `kaggle/sft_runtime.py` respectively.

ADR 0026 had deliberately kept several GDN-2/FLA qualification executables while backend qualification was still an active investigation. That investigation is now complete. The current production backend contract is recorded in `llm_docs/reference/gdn2_fla_backend.md`, and the canonical measured outputs are preserved under `llm_docs/evidence/` plus the archived investigation handoff. Git history preserves the exact source used to generate those historical results.

The cleanup audit distinguished old-looking files from dead files. In particular, the 100M-named training/publication engines are still loaded by `kaggle/runtime.py`, their helper modules remain live dependencies, the publication requirements file is still used by the unified launcher, and the per-profile publication environment templates remain referenced by reproducibility runbooks. Those files are not cleanup candidates merely because their names reflect earlier stages.

## Decision outcome

Remove the following completed investigation executables from `main` because they have no live code or test callers and are referenced only by historical decision/evidence/archive material:

```text
kaggle/run_gdn2_fla_layer_probe.py
kaggle/run_gdn2_fla_t4_probe.py
kaggle/run_gdn2_fla_step4000_parity.py
kaggle/run_gdn2_fla_step4000_benchmark.py
kaggle/run_gdn2_fla_fp32.py
```

Retain `kaggle/run_gdn2_fla_fp32_qualification.py`. Unlike the removed wrappers/probes, it is still exercised by `tests/test_gdn2_fla_fp32_qualification.py`, so deleting it would remove active regression coverage rather than merely reduce historical clutter.

Retain the active finite-data implementation chain:

```text
kaggle/launch.py
kaggle/runtime.py
kaggle/run_20m_100m_data_scaling.py
kaggle/run_20m_one_click.py
kaggle/run_20m_100m_console.py
kaggle/build_and_push_100m.py
kaggle/wandb_preflight.py
kaggle/requirements-100m-publish.txt
```

Retain the active SFT implementation chain:

```text
kaggle/launch_sft.py
kaggle/sft_runtime.py
kaggle/sft_publish.py
```

Retain the 100M/500M/2B publication environment examples while their corresponding reproducibility runbooks reference them.

Historical evidence and accepted technical results are not rewritten to remove the names of scripts that produced them. Those documents remain immutable evidence/history; the script source remains recoverable from Git.

## Consequences

### Positive

- `kaggle/` contains fewer completed-investigation executables that can be mistaken for current operational commands.
- The supported human launch surfaces remain unchanged.
- The active profile runtime, publication engines, SFT runtime, and regression-tested corrected FP32 qualification logic remain intact.
- Backend qualification evidence remains available without keeping every historical launcher/probe on `main`.

### Negative or limiting

- Re-running the removed August-8 probes requires checking out the historical revision that contains them.
- Historical evidence and ADRs may mention executable paths that no longer exist on `main`.

## Validation

- Exact-filename searches must show no live code/test caller for each removed script.
- `kaggle/runtime.py` must retain all files it imports or dynamically loads.
- `kaggle/sft_runtime.py` and `kaggle/launch_sft.py` must retain their publication/training dependencies.
- `tests/test_gdn2_fla_fp32_qualification.py` must continue to resolve `kaggle/run_gdn2_fla_fp32_qualification.py`.
- Current backend/reference documents must continue to point to preserved evidence rather than require a removed executable.

## Links

- [`0026-prune-superseded-one-off-kaggle-diagnostics.md`](0026-prune-superseded-one-off-kaggle-diagnostics.md)
- [`0030-consolidate-kaggle-profile-wrappers-behind-one-runtime.md`](0030-consolidate-kaggle-profile-wrappers-behind-one-runtime.md)
- [`../reference/gdn2_fla_backend.md`](../reference/gdn2_fla_backend.md)
- [`../runbooks/unified_kaggle_launcher.md`](../runbooks/unified_kaggle_launcher.md)
- [`../runbooks/sft_s0_runbook.md`](../runbooks/sft_s0_runbook.md)
