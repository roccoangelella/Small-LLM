# 20M / 500M qualification-report dispatch incident — 2026-08-07

## Classification

Launcher profile-dispatch defect. The fixed 500M dataset passed the attached-dataset full scan, but the inherited 100M launch loop invoked `dataset.qualification_100m_report` when deriving the finite one-pass schedule.

This is not a dataset-production failure, dataset-identity mismatch, trainer numerical failure, or checkpoint/resume failure.

## Observed failure

The Kaggle launch reached:

```text
[dataset] verify every attached shard
[ok] verify every attached shard
[dataset] derive the exact one-pass schedule
qualification report error: ValueError: qualification production target_source_tokens mismatch: expected 100000000, got 500000000
LAUNCH FAILED CLOSED: qualification-plan failed with exit code 1
```

The error is expected behavior from the 100M qualification report: it rejects a manifest whose production target is 500M. The correct report module already exists as `dataset.qualification_500m_report` and binds profile `20m-500m-data-scaling-v1` with the 500M source-token envelope.

## Impact

- The 500M attached dataset had already passed its literal full shard scan.
- Failure occurred before remote checkpoint restore and before the real trainer command.
- No 500M optimizer update was consumed.
- No partial 500M model checkpoint needs cleanup.
- The completed and verified 500M dataset does not need to be rebuilt or republished.

## Root cause

`kaggle/run_20m_500m_data_scaling.py` intentionally reuses the proven 100M launch loop. The loop contains one literal subprocess module name:

```text
dataset.qualification_100m_report
```

The 500M overlay replaced dataset/profile identities and trainer behavior but did not replace this literal report-dispatch string.

## Repair

Commit `b766cdf31c9f1222d5bca50bac2a0521bdb92a2e` updates the 500M one-click entry point to rewrite only the exact inherited report module:

```text
dataset.qualification_100m_report
→ dataset.qualification_500m_report
```

All other setup, verification, restore, trainer, validation, checkpoint, and publication commands continue through the existing launcher unchanged.

Commit `d2df3aa9118f4e0755b38ababaaa39ee362a5ecf` adds an isolated regression test verifying that the qualification-report command is rewritten and an unrelated dataset-verification command is left byte-for-byte unchanged.

The frozen training worktree remains `01d562ea1845d0dd128a0458e613c9e677b7381d`; that worktree already contains the correct 500M qualification-report module. The operational fix therefore requires only pulling current `main` before rerunning the normal one-click command.

## Recovery

On Kaggle:

```bash
cd /kaggle/working/Small-LLM
git switch main
git pull --ff-only
python kaggle/run_20m_500m.py
```

Expected corrected sequence:

```text
[dataset] verify every attached shard
[ok] verify every attached shard
[dataset] derive the exact one-pass schedule
[ok] derive the exact one-pass schedule
...
[train] ...
```

Fresh training must still begin directly at microbatch 4 with zero microbatch probe steps, and held-out validation/local checkpoint/verified remote publication remain every 250 successful optimizer updates.
