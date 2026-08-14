---
status: accepted
accepted: 2026-08-14
---

# Define the reasoning skill contract before R-SFT data construction

## Decision

Before building or training the first reasoning-oriented SFT dataset, define the target reasoning skill contract for the approximately-100M model.

The contract must specify at least:

- reasoning task/skill families;
- the three previously accepted difficulty bands;
- the intended concise trace shape for each band;
- what constitutes a correct final answer and a valid intermediate reasoning path;
- held-out qualification metrics and promotion gates;
- which tasks can be deterministically verified.

Dataset-source selection, teacher prompting, synthetic generation, and R-SFT implementation follow this skill definition rather than defining the curriculum implicitly through whichever dataset is easiest to obtain.

## Context

ADR 0073 already freezes the first R-SFT curriculum direction as concise reasoning split across three difficulty bands, shuffled rather than presented monotonically easy-to-hard. This decision makes the reasoning-skill taxonomy and evaluation contract the first design artifact for that stage.

## Deferred

This ADR does not yet choose whether reasoning/final-answer boundaries will be represented by new tokenizer vocabulary items or by textual delimiters encoded with the existing GPT-2 vocabulary. That remains a separate design choice.
