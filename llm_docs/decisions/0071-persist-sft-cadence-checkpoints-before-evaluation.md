---
status: accepted
date: 2026-08-14
---

# ADR 0071 — Persist SFT cadence checkpoints before evaluation

## Context

The live 100M/2B SFT run completed 250 finite optimizer updates on two T4s, then rank 0 was terminated by `SIGKILL` during the rank-zero-only evaluation phase. The step-250 update itself was healthy, but the previous cadence ordering ran validation and behavior evaluation before local checkpoint save and remote publication. A failure inside evaluation therefore discarded an otherwise valid exact-resume boundary.

The earlier NCCL cadence-wait failure had already been addressed by moving rank coordination to a long-timeout Gloo control group. This new failure is independent: rank 1 waited correctly while rank 0 disappeared during evaluation.

## Decision

At every SFT cadence boundary, durability precedes qualification side effects. The canonical order is:

1. save the exact local checkpoint;
2. publish the checkpoint remotely when remote publication is due;
3. run validation and behavioral evaluation.

A remote-publication boundary always implies a local save first, even when the configured local checkpoint cadence would not otherwise fire at that step.

The same checkpoint-first ordering applies to the final session boundary before final evaluation.

Cadence phases emit lightweight process/GPU memory telemetry around checkpointing, publication, validation, and behavior evaluation so an external kill can be localized without sacrificing the preceding optimizer work.

## Consequences

- Evaluation failure can no longer erase the optimizer progress at a checkpoint/publication boundary.
- Current-step validation/behavior metrics are not embedded in the checkpoint that was persisted immediately before those metrics were computed; they remain emitted through evaluation telemetry/W&B and the final training summary.
- If checkpoint serialization or remote publication itself fails, evaluation does not run; durability remains the first gate.
- The 100M/2B SFT profile must pin the checkpoint-first implementation commit before another live Kaggle run.
