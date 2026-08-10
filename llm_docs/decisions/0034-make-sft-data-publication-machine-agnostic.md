---
status: accepted
date: 2026-08-10
supersedes: null
---

# 0034 — Make SFT data preparation and publication machine-agnostic

## Context and problem statement

The first operational SFT launcher inherited unconditional Kaggle filesystem assumptions from the GPU training environment: `kaggle/sft_runtime.py` used `/kaggle/working` as its global work root and `/kaggle/input` as its global input root. That made `prepare` and `publish` fail on an ordinary VPS before any SFT work began, even though those actions are CPU/data/publication operations and have no technical dependency on a Kaggle notebook or GPU.

The project normally keeps large finite pretraining datasets on the VPS before private Kaggle publication. Running SFT bundle construction and private publication on that VPS avoids consuming Kaggle GPU session time merely to perform data preparation and upload work.

## Considered options

- Keep SFT preparation/publication Kaggle-only and always run those actions from a Kaggle notebook.
- Emulate Kaggle on other machines by creating root-level `/kaggle/working` and `/kaggle/input` directories.
- Make the SFT runtime machine-agnostic, retaining Kaggle path conventions only as conveniences when the launcher is actually running in a Kaggle environment.

## Decision outcome

Chosen option: **make SFT preparation/publication machine-agnostic and prefer the VPS when the source dataset already lives there**.

The runtime contract is:

- `prepare` and `publish` must work from a normal repository clone without requiring root-level `/kaggle` directories.
- `SMALL_LLM_WORK_DIR` may explicitly select the work root.
- When `/kaggle/working` actually exists, Kaggle keeps that conventional work root.
- Otherwise the default work root is a writable `small-llm-work` directory beside the controlling repository clone.
- `SMALL_LLM_INPUT_DIR` may override implicit input discovery.
- `/kaggle/input` remains the default implicit attached-dataset discovery location for Kaggle `train`/`eval`, but callers on any machine may instead pass `--dataset-dir` explicitly.
- `--replay-root` means the pretraining dataset directory containing `manifest.json`; passing the manifest file itself is invalid and must fail early with a clear error.
- When the verified replay dataset is already present on the VPS, use the VPS for SFT bundle preparation and private Kaggle publication so Kaggle accelerator sessions are reserved for training and evaluation.

## Consequences

### Positive

- The same canonical `kaggle/launch_sft.py` command surface works on ordinary Linux hosts and Kaggle.
- SFT bundle construction/publication no longer consumes Kaggle GPU session time unnecessarily.
- The accepted VPS-held pretraining dataset can be consumed directly without copying it into a Kaggle session first.
- Kaggle keeps convenient automatic `/kaggle/input` discovery for attached SFT bundles during training/evaluation.
- Incorrect `--replay-root .../manifest.json` invocations fail before worktree creation or source preparation.

### Negative or limiting

- A non-Kaggle host that wants implicit train/eval bundle discovery must configure `SMALL_LLM_INPUT_DIR`; otherwise it should pass `--dataset-dir` explicitly.
- Machine-local work products now live outside the repository by default, so operators should know the selected work root when cleaning or inspecting intermediate artifacts.
- Publication still requires valid Kaggle credentials/network access even when it is launched from the VPS.

## Validation

- Unit-test workspace resolution with Kaggle paths absent and with an explicit work-root override.
- Unit-test that a replay dataset directory containing `manifest.json` is accepted while passing `manifest.json` itself is rejected.
- Run the canonical 500M-parent `publish` command from the VPS using `/data/small-llm/20m-500m-ops/kaggle-dataset` as `--replay-root`.
- Require the complete prepare -> verify -> private upload -> fresh round-trip -> tree-hash comparison -> bundle re-verification path to succeed before treating the portability change as operationally qualified.

## Links

- [`../reference/post_training_sft.md`](../reference/post_training_sft.md)
- [`../runbooks/sft_s0_runbook.md`](../runbooks/sft_s0_runbook.md)
- [`0032-scale-sft-budget-with-pretraining-and-qualify-on-500m-first.md`](0032-scale-sft-budget-with-pretraining-and-qualify-on-500m-first.md)
- [`0033-use-comprehensive-post-sft-qualification-and-pretraining-cadence.md`](0033-use-comprehensive-post-sft-qualification-and-pretraining-cadence.md)
