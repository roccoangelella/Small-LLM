---
status: accepted
date: 2026-09-08
owners: [Small-LLM]
---

# ADR 0152: include prompt and continuation in GemRouter judgment output

## Decision

`trainer.base_prompt_judge` judgment artifacts must preserve the readable model evidence needed to inspect each semantic verdict without reopening the source evaluation JSON.

For every judged Base Prompt objective case, the output case record will include the exact source:

- `prompt`;
- `continuation`.

These fields are added alongside the existing case identity, family, verdict, score, and concise judge reason. No other raw generation metadata is duplicated into the GemRouter judgment artifact: generated token IDs, reference answers, response-token counts, seeds, and related evaluator metadata remain only in the source evaluation JSON.

The judge request, semantic scoring contract, prompt set, summaries, provider-batch provenance, and greedy/sampled evaluation behavior are unchanged. The new fields are copied locally from the already-loaded source rows after a successful judgment, so they do not affect GemRouter traffic or judge decisions.

## Motivation

The GemRouter-aided JSON should be directly readable as evaluation evidence. A reviewer should be able to inspect the original prompt and the model continuation beside each semantic judgment without cross-referencing a second JSON file.
