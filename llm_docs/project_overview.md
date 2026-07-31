# Project Overview

_Last updated: 2026-07-31_

## Goal

Build a dense decoder-only language model with fewer than 1B parameters from random initialization that:

- speaks and writes good English;
- follows instructions and holds coherent conversations after post-training;
- develops useful basic and intermediate reasoning;
- uses modern small-model architecture and training techniques;
- serves primarily as a learning and research project for an AI MSc student.

The initial scope does **not** deliberately target coding capability. Coding may be added later as a separate extension. Because semantic clusters are imperfect, incidental code can remain even after excluding the explicit programming cluster.

## Development strategy

The implementation is geometry-scalable rather than tied to one final model size. Development proceeds through:

1. an approximately 20M smoke model for correctness and integration;
2. an approximately 100M first substantive architecture comparison;
3. progressively larger controlled trials only after measured justification;
4. a possible near-1B model as a long-term goal, not the first run.

The initial model and dataset context is 2,048 input tokens. Longer contexts are deferred until the architecture, recurrence, trainer, checkpointing, and throughput are validated.

## Resource assumptions

- Initial accelerator: one NVIDIA T4.
- Likely microbatch size for larger trials: 1, with gradient accumulation.
- Local VPS storage budget: approximately 400 GB for live cache, checkpoints, temporary files, and working space.
- Durable dataset storage: personal Google Drive/Google One with approximately 5 TB available.
- Unique first-pass corpus target: 90B accepted source tokens.
- Minimum acceptable completed corpus: 80B accepted source tokens.
- Hard maximum: 100B accepted source tokens.
- Potential repeated-presentation target: up to 2T tokens, subject to later validation.

The intended first pass overlaps source streaming, preparation, local caching, Drive mirroring, and model training. With sufficient buffering, the T4 should be the steady-state bottleneck; network and preprocessing must not starve it.

## High-level system

```text
pinned source corpus
      ↓
deterministic dataset preparation
      ↓
locally durable immutable shards
      ↓
bounded trainer consumer
      ↓
geometry-scalable hybrid decoder
      ↓
versioned joint model/data checkpoints
```

Google Drive is the durable dataset mirror, not a random-access training filesystem. Training must be pausable and safely resumable, including migration between VPS providers.

## Documentation policy

The files in `llm_docs/` are the source of truth for project decisions, status, technical contracts, and open questions. There is no separate project-memory file.

When a decision changes:

1. update the relevant topic document;
2. update `project_status.md` or `decisions_and_ablations.md` when the current status or decision ledger changes;
3. record what changed, why it changed, and which benchmark, operational result, or new requirement justified it;
4. do not silently erase superseded reasoning.

The documentation is expected to evolve continuously with the implementation and experiments.
