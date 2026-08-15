---
status: accepted
date: 2026-08-15
---

# ADR 0088 — Use minimal schema, uniform R0 generation, 10% retention, and configurable reasoning serialization

## Context

The Gemini prompt suite and transport are ready. After manually sampling multiple deductive batches, the project explicitly chose not to add a formal semantic problem generator or deterministic logic verifier for the first R0 dataset. The next requirement is a small, testable dataset pipeline rather than more semantic enforcement machinery.

## Decision

### Minimal teacher schema

Gemini output is accepted only as a top-level JSON array with the expected number of records. Every record contains exactly three non-empty string fields:

- `problem`
- `reasoning`
- `answer`

Do not add provenance/model/provider fields merely because the current teacher is Gemini. Once accepted, project-side records add only the metadata required downstream: `skill` and `difficulty`.

Malformed JSON, surrounding prose/Markdown, wrong batch counts, missing fields, extra fields, and empty/non-string values fail closed. No semantic re-judge is performed in R0.

### Uniform R0 generation

The first R0 corpus is generated uniformly across the existing seven skills and three difficulty bands: 21 skill x difficulty cells with the same requested record count in each cell.

`generate.py` takes `examples_per_cell` explicitly rather than silently choosing the final production dataset size. Teacher calls are batched (default 10 records per call), including a smaller final call when the per-cell count is not divisible by the batch size.

Internal labels `L1`/`L2`/`L3` remain hidden from Gemini. The generator converts each level into plain-language structural requirements and passes those requirements to `prompts.py`. Generated records are globally shuffled with the project seed after collection so level labels do not create an easy-to-hard curriculum.

The generator must be testable without live API access by injecting a fake teacher client. A dry-run planning mode must also work without endpoint credentials.

### Retention starting point

Reasoning SFT starts with a 90% reasoning / 10% instruction-retention mixture. The 10% retention share is enforced at the loss-bearing target-token level by reusing the existing SFT `TargetTokenMixer`, rather than by raw record count.

The top-level 90/10 split is frozen here. The retention helper accepts the chosen instruction-source sub-mixture as an input instead of silently deciding which individual S0 sources must supply retention.

### Student serialization

Accepted reasoning examples serialize semantically as:

1. user problem;
2. reasoning-start marker;
3. teacher reasoning;
4. reasoning-end marker;
5. answer-start marker;
6. natural final answer;
7. normal conversation EOS supplied by the existing SFT template/tokenization path.

The three marker strings/IDs remain configurable inputs. This ADR does not supersede the required textual-delimiter versus atomic-special-token ablation and does not silently freeze exact token spellings.

Stored conversation metadata retains `skill` and `difficulty` for telemetry. Conversation identity is derived deterministically from training content, so no provenance field is required.

## Consequences

- The R0 data path is intentionally lightweight: prompt -> Gemini -> strict JSON schema -> attach skill/difficulty -> global shuffle -> serialize/mix for SFT.
- Gemini remains trusted for semantic problem/solution quality in this first dataset.
- Uniform generation enables clean skill x difficulty qualification.
- Retention is comparable to the existing SFT mixture because it is measured in supervised target tokens.
- Final production dataset size, retention sub-source allocation, and exact reasoning marker strings remain separate decisions.
