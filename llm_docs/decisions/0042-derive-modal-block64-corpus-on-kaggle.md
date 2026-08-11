---
status: superseded
date: 2026-08-11
superseded_by: 0043
---

# Derive the Modal block-64 corpus directly on Kaggle

## Historical decision

This ADR originally selected a Kaggle notebook as the place to derive and upload the block-64 Modal corpus from the existing `small-llm-20m-2b-dataset-001` dataset.

That operator path is no longer active. ADR 0043 supersedes it with a VPS-only control plane: the VPS downloads the immutable Kaggle dataset, verifies it, runs the same byte-preserving `dataset.reblock` transformation, uploads the derived corpus to the Modal data Volume, and launches Modal training.

The scientific block-64 decision from ADR 0041 is unchanged; only the operational handoff location changed.
