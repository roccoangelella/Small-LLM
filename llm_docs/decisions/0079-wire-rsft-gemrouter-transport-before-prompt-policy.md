---
status: accepted
last_reviewed: 2026-08-14
---

# ADR 0079 — Wire the R-SFT GemRouter transport before prompt policy

## Decision

Create the first reasoning-distillation API transport at `post_training/R-SFT/dataset.py`.

The transport uses the user-provided OpenAI-compatible endpoint:

`https://gemr.84-8-255-231.nip.io/v1/chat/completions`

with default model `gemini-3.7-flash` and bearer authentication from `GEMR_API_KEY`. The client must accept ordinary chat `role`/`content` messages and normalize the first assistant response from `choices[0].message.content`.

For this base implementation, send only `model` and `messages`. Do not set temperature, top-p, top-k, token limits, seed, or other generation controls; provider defaults remain in force.

`GEMR_API_KEY` may be supplied through the process environment or the repository-root `.env`. The real secret must never be committed.

## Scope boundary

This change is transport only. Do not yet freeze or implement:

- R0 prompt templates or few-shot examples;
- skill/difficulty generation instructions;
- batching policy or call-budget allocation;
- deterministic logical verifiers;
- rejection/repair/retry policy;
- deduplication or decontamination;
- final frozen dataset serialization.

Those are separate decisions after the request/response plumbing is qualified.

## Verification

Keep the transport dependency-light and cover the request/response contract with local unit tests that do not make a live Gemini call or require a real API key.
