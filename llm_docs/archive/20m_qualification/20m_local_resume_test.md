# Approximately-20M Local Interruption and Resume Test

_Last updated: 2026-08-04_

## Decisions

On 2026-08-04 the user accepted the observed universal gradient clipping for the frozen approximately-20M qualification recipe.

The accepted interpretation is:

```text
max_grad_norm: 1.0
observed clipping frequency: 100% in both 50-update A/A runs
first-10 gradient-norm median: 1.3997
last-10 gradient-norm median: 1.3850
final gradient norm: 1.3443
maximum observed gradient norm: 2.6810
FP16 overflow events: 0
A/A non-runtime numerical differences: 0 / 10,650
```

Universal clipping is therefore not a blocker for this engineering qualification. The decision applies only to the frozen 20M recipe and evidence set. It is not a general claim that clipping every update is optimal, and it does not silently change the learning rate, clipping threshold, optimizer, or later approximately-100M experiments.

The next approved qualification gate is an actual trainer-process interruption at the verified update-25 checkpoint boundary followed by exact local resume through update 50.

## Authoritative Kaggle entrypoint

```text
kaggle/run_20m_local_resume_from_clone.py
```

Run from the already-cloned repository:

```python
%cd /kaggle/working/Small-LLM
!git pull --ff-only
!python kaggle/run_20m_local_resume_from_clone.py
```

Requirements remain the same as the prior Kaggle tests:

```text
accelerator: NVIDIA Tesla T4
internet: enabled
accepted private qualification dataset: attached
Kaggle Secret: WANDB_API_KEY
optional Kaggle Secret: WANDB_ENTITY
```

## Scope

One invocation performs the following fail-closed sequence:

1. Keep the controlling clone on `main` and create a clean detached worktree at frozen launch commit `45d1da4a1ac3f18cf6ce02b8439672f10e2c8b4c`.
2. Select the accepted mounted dataset by both frozen manifest hashes.
3. Run another literal full dataset scan and regenerate the exact 306-update plan.
4. Run an uninterrupted 50-update WSD-prefix reference with checkpoints at updates 25 and 50.
5. Start a separate 50-update trainer process from the same initial state.
6. Wait until the trainer has emitted and the controller has independently verified the complete `step-00000025` checkpoint.
7. Send `SIGTERM` to the actual trainer process group, require a non-zero interrupted exit, and verify that no process in that group remains alive.
8. Verify that the update-25 checkpoint remains complete, hash-valid, at an optimizer boundary, and points to next block 25.
9. Start a fresh process with `--resume step-00000025` and the same W&B run ID, then run exactly 25 additional successful updates through update 50.
10. Compare the uninterrupted reference against the combined interrupted/resumed trajectory.
11. Compare semantic checkpoint contents at updates 25 and 50, including model, optimizer, scheduler, scaler, RNG, counters, and pipeline state.
12. Preserve logs, exit-code files, checkpoints, W&B identities, comparison reports, and a final JSON summary under `/kaggle/working`.

## Pass requirements

The test passes only when:

```text
reference successful updates: 1 through 50 exactly
interrupted successful updates before signal: 1 through 25 exactly
resumed successful updates: 26 through 50 exactly
next block after checkpoint: 25
actual process-group termination: confirmed
combined discrete trajectory: exact
combined non-runtime numerical trajectory: exact
validation result: exact
step-25 semantic checkpoint state: exact
step-50 semantic checkpoint state: exact
non-finite values: none
```

Byte-for-byte checkpoint-tree equality is recorded but is not itself required when semantic state is exact. This distinction addresses the earlier A/A finding where checkpoint tree hashes differed despite exactly identical training telemetry.

## Interpretation boundary

A successful result is written as:

```text
status: passed_local_interruption_resume
authorization: remote_recovery_only
```

It authorizes preparation and execution of the private remote-publication and empty-environment recovery test. It does not authorize the complete 306-update run by itself.

The authoritative final summary path is:

```text
/kaggle/working/small_llm_local_resume_summary.json
```

The full one-pass training run remains unauthorized until this local-resume gate and the subsequent remote empty-environment recovery gate pass.
