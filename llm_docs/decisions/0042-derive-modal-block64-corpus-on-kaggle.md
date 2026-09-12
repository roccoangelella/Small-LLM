---
status: superseded
date: 2026-08-11
supersedes: null
superseded_by: 0043
---

# Derive the Modal block-64 corpus directly on Kaggle

## Context and problem statement

No separate context was recorded at the time; section added for the template.

## Considered options

No alternative was recorded at the time; section added for the template.

## Decision outcome

This ADR originally selected a Kaggle notebook as the place to derive and upload the block-64 Modal corpus from the existing `small-llm-20m-2b-dataset-001` dataset.

That operator path is no longer active. ADR 0043 supersedes it with a VPS-only control plane: the VPS downloads the immutable Kaggle dataset, verifies it, runs the same byte-preserving `dataset.reblock` transformation, uploads the derived corpus to the Modal data Volume, and launches Modal training.

The scientific block-64 decision from ADR 0041 is unchanged; only the operational handoff location changed.
## Consequences

No consequences were recorded at the time; section added for the template.
