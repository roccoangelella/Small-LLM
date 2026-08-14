---
status: accepted
last_reviewed: 2026-08-14
---

# ADR 0077 — Track reasoning difficulty labels for telemetry only

## Decision

For the first reasoning-oriented SFT dataset, every example must carry an explicit difficulty label (`L1`, `L2`, or `L3`) as dataset metadata.

The difficulty label is **not** part of the serialized model input or target and must not influence sampling/order during training. Training examples remain globally shuffled across difficulty bands as already decided.

The label exists for analysis and qualification. Training/evaluation telemetry must support reporting supervised token loss by difficulty band so we can observe whether the student learns easy, medium, and harder reasoning at different rates.

## Measurement contract

Per-band loss should be normalized by the number of supervised target tokens in that band, not by example count, so different trace lengths do not bias the comparison.

Where the reasoning serialization makes span boundaries available, qualification should additionally support separate reasoning-span and answer-span metrics by difficulty. Difficulty metadata should remain available alongside a separate skill-family label so learning can later be inspected on both axes without exposing either label to the model.

## Rationale

The first reasoning SFT deliberately mixes three toughness bands rather than presenting an easy-to-hard curriculum. Keeping the bands as telemetry-only metadata preserves that training contract while making the learning dynamics observable.
