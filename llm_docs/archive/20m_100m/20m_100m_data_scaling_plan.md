# 20M Model / 100M-Token Data-Scaling Plan — Historical

_Last updated: 2026-08-10 Europe/Rome_

This file records the **historical planning stage** that led to the completed 20M/100M experiment. Several operational details in the original plan were changed during execution (including W&B identity, session/cadence policy, and the final launcher surface), so the old document is no longer an executable runbook.

Git history preserves the full original plan and its intermediate assumptions.

## Final experiment identity

```text
model parameters: 20,637,592
architecture: gdn2_hybrid
context length: 2,048
seed: 17
dataset profile: 20m-100m-data-scaling-v1
dataset run ID: 20m-100m-dataset-001
W&B run ID: 20m-100m-data-004
launch commit: 8e3cd9cb149facc5fa28e8108a70304c1f8c1c15
target accepted source tokens: 100,000,000
minimum: 90,000,000
maximum: 110,000,000
producer checkpoint cadence: 20,000,000 source tokens
training microbatch: 4
final durability / validation / remote publication cadence: 250 updates
```

## Current reproduction command surface

Dataset publication:

```bash
python kaggle/launch.py publish --model 20M --tokens 100M
```

Training / exact resume:

```bash
python kaggle/launch.py train --model 20M --tokens 100M
```

For the complete current procedure, fixed identities, completed result, and evidence locations, use:

- [`../../runbooks/20m_100m_runbook.md`](../../runbooks/20m_100m_runbook.md)
- [`../../runbooks/unified_kaggle_launcher.md`](../../runbooks/unified_kaggle_launcher.md)

## Scientific intent retained from the historical plan

The experiment changed dataset scale while keeping the approximately-20M model family, tokenizer, source revision/policy, context length, seed, optimizer family, effective 16-sequence optimizer block, and one-pass WSD policy fixed. The finite dataset remained an immutable schema-v2 shard set consumed locally on Kaggle rather than a live training stream.

The completed run is now evidence, not an active authorization for further 100M execution-policy changes.
