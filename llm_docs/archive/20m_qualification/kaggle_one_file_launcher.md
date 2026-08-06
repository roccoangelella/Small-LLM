# One-File Kaggle Qualification Launcher

_Last updated: 2026-08-04_

## Decision

On 2026-08-04 the user chose to consolidate the immediate Kaggle qualification workflow into one executable workflow rather than manually running the runbook command by command.

The user subsequently clarified that Kaggle should first clone the repository and then run an entrypoint from the repository's own `kaggle/` directory. Uploading or invoking a separate external Python file is not part of the approved workflow.

The repository-native entrypoint is:

```text
kaggle/run_20m_from_clone.py
```

Expected notebook invocation:

```python
%cd /kaggle/working/Small-LLM
!python kaggle/run_20m_from_clone.py
```

The controlling clone remains intact. Because the frozen launch commit predates the launcher files, the entrypoint creates a separate detached Git worktree at the frozen launch commit and runs every evidence-producing command there. It does not silently test the controller clone's current `main` state.

The underlying fail-closed qualification implementation remains in:

```text
kaggle/run_20m_one_click.py
```

By default one execution performs:

1. NVIDIA T4 environment verification;
2. detached worktree creation at the frozen launch commit from the existing clone;
3. Python 3.13 and dependency setup through `uv`;
4. the complete offline test suite;
5. the corrected T4 parity, FP16, memory, and throughput harness;
6. automatic discovery of the attached qualification dataset by the accepted manifest and Drive-manifest SHA-256 values;
7. literal full dataset verification;
8. regeneration and exact validation of the 306-update qualification plan;
9. the 20-successful-update constant-LR trainer preflight with W&B;
10. durable logs, numeric exit-code files, reports, checkpoint evidence, and a single summary JSON under `/kaggle/working`.

The launcher may also be run with `--gates-only` to stop before the trainer preflight.

## Credentials

The repository-native entrypoint does not need `GITHUB_TOKEN` after the private repository has already been cloned. The default preflight still requires the Kaggle Secret `WANDB_API_KEY`; `WANDB_ENTITY` remains optional.

## Safety boundary

This consolidation changes orchestration only; it does not remove qualification gates or change the model, optimizer, data, or schedule decisions.

The launcher deliberately does **not** start the complete 306-update one-pass segment. A successful 20-update preflight authorizes post-preflight review only. Empirical threshold freezing, same-hardware A/A repeatability, actual-process interruption/resume, and remote publication plus empty-environment recovery must still pass before the one-pass segment is authorized.

## Frozen identities embedded by the launcher

```text
launch commit: 45d1da4a1ac3f18cf6ce02b8439672f10e2c8b4c
manifest SHA-256: 1e5ee8f372b77b6728288610dbe7cce74d833be21e53d1538bc5a890229b18bb
Drive manifest SHA-256: fbb29ee0d0102658e1274e39d6647cf56a6dcb685e0f566b1736847dcc4fbe84
```

A future launch-commit change must be explicit and recorded; it must not silently follow `main`.
