# Approximately-20M Kaggle Launch Authorization

_Last updated: 2026-08-05_

## Decision

At 2026-08-05 10:04 Europe/Rome, the user explicitly authorized launching the frozen complete approximately-20M one-pass pretraining segment.

```text
launch authorization: granted
execution venue: Kaggle
accelerator: NVIDIA Tesla T4
run state at decision time: authorized_not_yet_started
```

This resolves the explicit launch gate recorded in `project_status.md` and `20m_remote_recovery_results.md`.

## Authorized run

```text
model parameters: 20,637,592
architecture: gdn2_hybrid
launch commit: 45d1da4a1ac3f18cf6ce02b8439672f10e2c8b4c
dataset run ID: 20m-qualification-dataset-001
manifest SHA-256: 1e5ee8f372b77b6728288610dbe7cce74d833be21e53d1538bc5a890229b18bb
Drive manifest SHA-256: fbb29ee0d0102658e1274e39d6647cf56a6dcb685e0f566b1736847dcc4fbe84
steps: 306
passes: 1
train target tokens: 10,006,528
schedule: WSD
warmup: 16 updates / 524,288 target tokens
stable: 228 updates / 7,471,104 target tokens
decay: 62 updates / 2,011,136 target tokens
minimum LR ratio: 0.1
seed: 17
precision: FP16
```

## Frozen-recipe constraint

The authorization preserves the qualified recipe unchanged. It does not authorize changes to:

- architecture or model geometry;
- initialization or seed;
- optimizer routing or optimizer hyperparameters;
- learning rate, clipping, or schedule;
- dataset identity, order, pass count, or token horizons;
- checkpoint, validation, and remote-publication policy;
- frozen launch commit.

Any such change would define a different run and would require a new explicit decision and appropriate requalification.

## Execution interpretation

Kaggle is the selected venue for this run. The run becomes `running` only after the trainer process starts and produces verifiable evidence, such as its W&B identity and local evidence directory. The authorization decision alone must not be recorded as a completed launch.

The current manual launch procedure remains the exact complete one-pass construction in `llm_docs/20m_kaggle_runbook.md`, using the accepted private dataset and required Kaggle secrets. After startup, record the actual W&B run ID, evidence path, Kaggle environment identity, and first successful checkpoint boundary in the project documentation.
