# Approximately-20M Remote Recovery Attempt — 2026-08-05

## Outcome

The corrected Hugging Face repository configuration was accepted:

```text
roccoangelella/small-llm-20m-qualification
```

The publisher segment trained through update 25, created `step-00000025`, uploaded the complete approximately-217 MB trainer state to the private Hugging Face repository, and emitted a successful `remote_publication` event.

The overall gate did not proceed to empty-environment restore because the local five-step continuation restarted from initialization instead of resuming the update-25 checkpoint.

## Root cause

`kaggle/run_20m_remote_recovery_from_clone.py` built resume commands through the proven local-resume helper, then replaced the command tail beginning at `--wandb-tags`. The helper appends `--resume step-00000025` after that tag section, so the replacement unintentionally deleted the resume argument.

Evidence in the executed command:

```text
--steps 5
--wandb-resume allow
--wandb-tags 20m t4 remote-recovery empty-environment
```

The required argument was absent:

```text
--resume step-00000025
```

The resulting run reported blocks 0-4, local steps 1-5, warmup learning rates, and final checkpoint `step-00000005`. The controller correctly failed closed because it expected resumed updates 26-30.

This is a Kaggle qualification-controller bug. It is not evidence of a trainer resume failure, model instability, dataset failure, Hugging Face publication failure, or Google Drive restore failure.

## Fix

A fail-closed hotfix entrypoint was added:

```text
kaggle/run_20m_remote_recovery_resume_fix_from_clone.py
```

It imports the original controller, restores `--resume <checkpoint>` after tag rewriting, and verifies that every resume command contains exactly one resume flag with the expected checkpoint ID.

## Next invocation

```python
%cd /kaggle/working/Small-LLM
!git pull --ff-only
!python kaggle/run_20m_remote_recovery_resume_fix_from_clone.py
```

The full 306-update training run remains unauthorized until this corrected remote empty-environment recovery gate passes.
