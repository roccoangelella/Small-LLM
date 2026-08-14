---
status: accepted
date: 2026-08-14
---

# ADR 0081 — Track reasoning difficulty labels for telemetry only

## Context and problem statement

The three mixed difficulty bands need separate learning diagnostics without turning their labels into model-visible hints or an implicit curriculum.

## Considered options

1. Omit difficulty labels after shuffling.
2. Serialize difficulty labels into model inputs or targets.
3. Retain labels as telemetry-only metadata.

## Decision outcome

For the first reasoning-oriented SFT dataset, every example must carry an explicit difficulty label (`L1`, `L2`, or `L3`) as dataset metadata.

The difficulty label is **not** part of the serialized model input or target and must not influence sampling/order during training. Training examples remain globally shuffled across difficulty bands as already decided.

The label exists for analysis and qualification. Training/evaluation telemetry must support reporting supervised token loss by difficulty band so we can observe whether the student learns easy, medium, and harder reasoning at different rates.

## Measurement contract

Per-band loss should be normalized by the number of supervised target tokens in that band, not by example count, so different trace lengths do not bias the comparison.

Where the reasoning serialization makes span boundaries available, qualification should additionally support separate reasoning-span and answer-span metrics by difficulty. Difficulty metadata should remain available alongside a separate skill-family label so learning can later be inspected on both axes without exposing either label to the model.

## Rationale

The first reasoning SFT deliberately mixes three toughness bands rather than presenting an easy-to-hard curriculum. Keeping the bands as telemetry-only metadata preserves that training contract while making the learning dynamics observable.

## Consequences

- Dataset records must retain difficulty and skill-family metadata outside model-visible text.
- Training order remains globally shuffled and cannot use difficulty labels as a curriculum signal.
- Evaluation must normalize per-band loss by supervised target tokens and should expose span-level metrics when available.
