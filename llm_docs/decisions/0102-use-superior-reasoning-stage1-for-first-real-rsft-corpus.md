---
status: superseded
date: 2026-08-19
superseded_by: 0104
---

# ADR 0102 — Use Superior Reasoning Stage 1 for the first real R-SFT corpus

## Context

The repeated-epoch R0 diagnostic established that the approximately-100M model can learn the atomic `<think>`, `</think>`, `<answer>` response protocol when the frozen 630-example Gemini corpus is replayed heavily, but the resulting behavior is strongly overfit: question-shaped prompts trigger reasoning-shaped text without reliable semantic reasoning, while generalization remains weak. More replay of the same tiny corpus is therefore not the next production direction.

Alibaba-Apsara's Superior Reasoning SFT release provides a much larger set of independently generated reasoning examples. Its Stage 1 is the lower-temperature, stability-oriented portion and exposes explicit domain labels including `science` and `instruction_following`, allowing the first expansion to avoid math and code rather than approximating domain from prompt text.

## Decision

For the first larger R-SFT corpus after the 630-example R0 diagnostic:

- use only `Alibaba-Apsara/Superior-Reasoning-SFT-gpt-oss-120b`, config `stage1`, split `train`;
- retain only rows whose domain is `science` or `instruction_following`; exclude `math` and `code` for this experiment;
- scan the complete Stage-1 stream and report the exact available row count for every source domain before selection;
- select 25,000 unique Superior examples ranked by the token length of the parsed reasoning body under the project's GPT-2 tokenizer;
- target a 30% science / 70% instruction-following allocation, i.e. 7,500 / 17,500 examples when both pools are large enough;
- if one eligible domain cannot satisfy its requested quota, deterministically backfill the deficit from the other eligible domain rather than reducing the overall 25,000-example target;
- reject malformed outputs that do not contain a non-empty `<think>...</think>` reasoning body followed by a non-empty final answer;
- drop exact duplicate prompts after whitespace normalization and case folding before shortest-trace ranking;
- merge the selected Superior examples with the existing approximately-630-example Gemini R0 corpus for this first experiment; do not generate another large Gemini-only corpus yet;
- shuffle the combined records deterministically and emit a manifest containing source/domain counts, usable counts, realized selection counts, selected reasoning-token range, seed, and final JSONL SHA-256;
- implement the source adapter at `post_training/R-SFT/dataset/superior_reasoning.py`.

The Superior records are serialized into the existing five-field reasoning record shape using source-specific telemetry labels (`SR_SCIENCE` and `SR_INSTRUCTION_FOLLOWING`). The existing Gemini records retain their original R0 skill/difficulty labels.

## Rationale

Twenty-five thousand unique external examples increase problem and language diversity by roughly two orders of magnitude relative to the original R0 corpus without immediately moving to a corpus so large that a failed training recipe becomes expensive to diagnose. Selecting the shortest valid reasoning traces biases the first approximately-100M experiment toward teacher behaviors it has a realistic chance to imitate, while keeping instruction-following as the majority domain directly attacks the narrow question-shaped routing observed after the repeated-epoch probe.

Stage 1 is preferred over Stage 2 for this first expansion because it is the lower-temperature/stability portion of the source release. Math and code are intentionally deferred so that the experiment remains focused on broader reasoning and instruction behavior rather than exact computation.

## Integration boundary

The current production `post_training/R-SFT/build_atomic.py` still enforces the historical uniform Gemini R0 skill-by-difficulty matrix. This decision does **not** silently weaken that invariant. Support for building/training a heterogeneous Superior-plus-Gemini corpus must be added explicitly before this new dataset becomes a trainable production bundle.

## Consequences

- The 10-epoch 630-example checkpoint remains diagnostic evidence, not the preferred R-SFT parent or production recipe.
- The next dataset construction step can determine exact Stage-1 science/instruction-following availability and produce the frozen 25k selection in one streaming pass.
- Reasoning-length selection is tokenizer-aware and reproducible rather than based on characters or raw bytes.
- The external corpus is additive to the existing Gemini R0 examples for now; the project can later revisit domain weights, add math/code, or use Stage 2 based on qualification evidence.
