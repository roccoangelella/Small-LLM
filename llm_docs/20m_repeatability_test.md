# Approximately-20M Same-T4 Repeatability Test

_Last updated: 2026-08-04_

## Decision

On 2026-08-04 the user chose to run the post-preflight reference and A/A repeatability stage using the same repository-native Kaggle modality as the successful 20-update preflight.

The authoritative entrypoint is:

```text
kaggle/run_20m_repeatability_from_clone.py
```

It is run from an already-cloned repository on Kaggle. The controlling clone remains on `main`, while evidence-producing commands execute in a clean detached Git worktree at the frozen launch commit:

```text
45d1da4a1ac3f18cf6ce02b8439672f10e2c8b4c
```

## Scope

One invocation:

1. requires an NVIDIA T4 and the existing `WANDB_API_KEY` Kaggle Secret;
2. selects the accepted attached dataset by its frozen manifest identities;
3. performs another literal full dataset scan and regenerates the exact 306-update plan;
4. runs an uninterrupted 50-successful-update reference segment;
5. runs a second independent 50-successful-update A/A segment from the same seed, initialization, model, optimizer, WSD horizons, data order, and hardware;
6. validates local checkpoints at updates 25 and 50;
7. compares discrete trajectories, non-runtime numerical telemetry, validation results, and complete checkpoint-tree hashes;
8. records clipping, gradient-norm, FP16 scaler/overflow, throughput, memory, optimizer, loss, and validation summaries;
9. writes complete logs, exit-code files, checkpoints, and a final summary JSON under `/kaggle/working`.

The exact training prefix is:

```text
steps: 50
schedule: WSD using the full 306-update token horizons
warmup tokens: 524,288
stable tokens: 7,471,104
decay tokens: 2,011,136
minimum LR ratio: 0.1
checkpoint cadence: updates 25 and 50
validation: update 50
seed: 17
architecture: GDN-2 hybrid
chunk size: 32
precision: FP16
optimizer: hybrid Muon + AdamW
```

## Interpretation boundary

This test measures the same-T4 nondeterministic floor and whether the clipping and gradient-norm pattern observed in the 20-update preflight remains repeatable and bounded over a longer prefix.

A successful execution produces `passed_repeatability_measurement` with authorization limited to `threshold_review_only`. It does not automatically authorize the complete 306-update segment. The resulting distributions must be reviewed and empirical warning/failure thresholds frozen before the process-kill/resume and remote-recovery stages.

Runtime-dependent values such as elapsed time, throughput, data wait, and peak-memory counters are summarized but excluded from exact numerical trajectory comparison. Dataset/block/counter/clipping/overflow identities must match exactly. Non-runtime numerical differences and checkpoint-tree differences are reported as measured nondeterminism rather than hidden or retroactively tolerated.

## Kaggle invocation

```python
%cd /kaggle/working/Small-LLM
!git pull --ff-only
!python kaggle/run_20m_repeatability_from_clone.py
```

The authoritative summary path is:

```text
/kaggle/working/small_llm_repeatability_summary.json
```

The full 306-update run remains unauthorized until repeatability interpretation, local interruption/resume, and remote empty-environment recovery pass.