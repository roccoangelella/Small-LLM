# One-File Kaggle Qualification Launcher

_Last updated: 2026-08-04_

## Decision

On 2026-08-04 the user chose to consolidate the immediate Kaggle qualification workflow into one executable file rather than manually running the runbook command by command.

The authoritative launcher is:

```text
kaggle/run_20m_one_click.py
```

It is a fail-closed operational wrapper. By default one execution performs:

1. NVIDIA T4 environment verification;
2. private GitHub clone and detached checkout of the frozen launch commit;
3. Python 3.13 and dependency setup through `uv`;
4. the complete offline test suite;
5. the corrected T4 parity, FP16, memory, and throughput harness;
6. automatic discovery of the attached qualification dataset by the accepted manifest and Drive-manifest SHA-256 values;
7. literal full dataset verification;
8. regeneration and exact validation of the 306-update qualification plan;
9. the 20-successful-update constant-LR trainer preflight with W&B;
10. durable logs, numeric exit-code files, reports, checkpoint evidence, and a single summary JSON under `/kaggle/working`.

The launcher may also be run with `--gates-only` to stop before the trainer preflight.

## Safety boundary

This consolidation changes orchestration only; it does not remove qualification gates or change the model, optimizer, data, or schedule decisions.

The launcher deliberately does **not** start the complete 306-update one-pass segment. A successful 20-update preflight authorizes post-preflight review only. Empirical threshold freezing, same-hardware A/A repeatability, actual-process interruption/resume, and remote publication plus empty-environment recovery must still pass before the one-pass segment is authorized.

## Frozen identities embedded by the initial launcher

```text
launch commit: 45d1da4a1ac3f18cf6ce02b8439672f10e2c8b4c
manifest SHA-256: 1e5ee8f372b77b6728288610dbe7cce74d833be21e53d1538bc5a890229b18bb
Drive manifest SHA-256: fbb29ee0d0102658e1274e39d6647cf56a6dcb685e0f566b1736847dcc4fbe84
```

A future launch-commit change must be explicit and recorded; it must not silently follow `main`.
