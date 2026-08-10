# Supervised fine-tuning implementation reference

_Last reviewed: 2026-08-10 Europe/Rome_

This document describes the SFT implementation surface that currently exists in the repository. It is **not authorization to start a production SFT run** and it does not automatically carry the old 20M/100M S0 hyperparameters forward to a later base checkpoint.

The earlier August 6 S0 design/decision packet is preserved under `../archive/post_training_s0_2026-08-06/`. Those documents are historically important because they record user-approved choices for the then-planned 20M/100M S0 qualification, but their budgets, learning rates, mixtures, and milestones were scoped to that experiment.

## Code surface

The reusable implementation lives under:

```text
post_training/sft/
```

The public package surface currently exposes:

- logical conversation/schema types;
- `GPT2ChatTemplate` and GPT-2 encoding support;
- deterministic S0 record filtering;
- SFT data configuration and schedule planning;
- dataset building, storage, and shard reading;
- SFT training-plan support;
- base/SFT state-dict interpolation.

See `post_training/sft/__init__.py` for the current exported API.

## Implementation invariants inherited from the design work

The implementation was built to support a finite, auditable post-training pipeline rather than a one-off script. Important design properties include:

- immutable parent base-checkpoint identity;
- explicit dataset/manifest identity;
- deterministic data ordering and resume state;
- explicit supervision masks for chat targets;
- finite streams without implicit wraparound;
- target-token accounting rather than example-count accounting;
- checkpoint/resume state sufficient to identify the exact next training position;
- configuration-driven budgets and mixtures rather than a universal hard-coded 4M limit;
- no required change to decoder geometry or tokenizer vocabulary simply to enter SFT;
- a configurable interpolation utility for evaluating movement back toward the parent base checkpoint.

## Historical S0 baseline

The August 6 design work explored and in several cases approved a first S0 baseline for the approximately-20M model after the 100M pretraining stage. That historical packet included decisions around full-parameter fine-tuning, masked cross-entropy, replay, chat serialization, finite target-token horizons, checkpointing, and later distillation stages.

Those records are not rewritten into a present production recipe because the project subsequently continued pretraining/data scaling and the next post-training decision is intentionally deferred until the current base-model evidence is reviewed.

When a new SFT run is authorized, create a new ADR that explicitly freezes at least:

- parent base checkpoint;
- exact dataset sources and revisions;
- chat template and supervision-mask identity;
- train/validation/test split identity;
- instruction/replay mixture;
- finite loss-bearing target-token budget;
- optimizer family and hyperparameters;
- scheduler policy and exact derived horizons;
- optimizer target-token block and hardware microbatch;
- validation/checkpoint/publication cadence;
- model-selection and base-retention gates;
- teacher identity/terms if distillation is used.

Do not infer those values from the archived S0 packet merely because the implementation supports them.

## Tests

The repository contains unit/integration coverage for SFT configuration, schema, templates, filtering/builder behavior, mixtures, storage, and trainer integration under `tests/test_sft_*.py`.

## Related material

- Historical S0/SFT design packet: [`../archive/post_training_s0_2026-08-06/README.md`](../archive/post_training_s0_2026-08-06/README.md)
- Current roadmap: [`../current/roadmap.md`](../current/roadmap.md)
- General training contract: [`training_system.md`](training_system.md)
