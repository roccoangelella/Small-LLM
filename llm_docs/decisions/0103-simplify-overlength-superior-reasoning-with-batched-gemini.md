---
status: accepted
date: 2026-08-19
supersedes: null
---

# 0103 — Simplify overlength Superior Reasoning with batched Gemini

## Context and problem statement

The first real R-SFT corpus contains 25,000 selected Superior Reasoning examples plus the frozen 630-example Gemini R0 corpus. The 100M model and production R-SFT data path use a 2,048-token context window.

A direct serialization audit with the actual R-SFT GPT-2 + reasoning-token template found that 13,002 of the 25,630 combined reasoning records do not fit the 2,048-token training context: 3,654 science examples and 9,348 instruction-following examples. Only 12,628 records fit unchanged. Some failures are dominated by long reasoning traces, while some instruction-following examples have an inherently long required answer, so shortening reasoning alone is insufficient.

The existing private GemRouter transport can call `gemini-3.7-flash`. Because the endpoint is used under a limited/free-tier budget, overlength repair must be batched rather than performed one request per document.

## Considered options

- Drop every overlength example and train only on the 12,628 examples that fit unchanged.
- Truncate overlength examples at the 2,048-token boundary.
- Ask Gemini to compress only problem/reasoning while preserving answers verbatim.
- Ask Gemini to freely create smaller analogous curriculum examples.
- Ask Gemini for fidelity-first compression, with minimal scope reduction only when the original required output itself is too large.

## Decision outcome

Chosen option: **use batched Gemini fidelity-first curriculum compression for overlength Superior Reasoning examples, with at most four examples per GemRouter request and exact local tokenizer validation after every response**.

The selected system prompt is:

```text
You are a fidelity-first curriculum compressor for reasoning-SFT examples that must fit a ~100M parameter model with a 2,048-token context window.

Process every input item independently and return one rewritten item for each input id.

Priority order:
1. Preserve the original task type, domain, conclusion, named entities, factual relationships, constraints, and numerical givens whenever they can fit.
2. First shorten by removing repetition, digressions, verbose exposition, redundant examples, and unnecessary formatting. Make the reasoning explicit, linear, and easy to imitate.
3. If the ORIGINAL REQUIRED OUTPUT is itself too large for a compact example, make the smallest scope reduction necessary: shorten source material, reduce repeated list items/subparts, or narrow the requested deliverable while preserving the same instruction-following skill and core reasoning pattern. The answer must then be correct for the rewritten problem.
4. Do NOT change scientific constants, numerical values, entities, or conclusions merely to make an example look simpler. Change them only if unavoidable for a scope-reduced analogous task, and prefer deleting irrelevant context instead.
5. Never invent unsupported facts. Keep each problem self-contained and solvable.
6. Target <= 160 words for problem, <= 300 words for reasoning, <= 160 words for answer, preferably <= 600 words total. Concision is more important than filling the budget.
7. Output strict RFC 8259 JSON only: one JSON array, same item order, objects with exactly id, problem, reasoning, answer. Use valid JSON escaping. Prefer plain-text math notation instead of LaTeX/backslashes. No Markdown code fence and no commentary outside the JSON.
```

The implementation contract is exposed in `post_training/R-SFT/dataset/superior_reasoning.py` as `SIMPLIFICATION_SYSTEM_PROMPT`, `SIMPLIFICATION_MAX_BATCH_SIZE = 4`, `build_simplification_messages`, and `parse_simplification_response`.

Every accepted rewrite must pass strict JSON/schema/order validation and then pass the real R-SFT tokenizer/template at a 2,048-token context. A rewrite that remains too long, changes IDs/order, or produces malformed JSON is rejected rather than silently truncated. Failed items may be retried in later batches of at most four.

## Prompt bake-off evidence

Four representative overlength examples were selected for the live GemRouter test: near-threshold and substantially overlength examples from both science and instruction-following. Their original serialized lengths were 2,049, 3,618, 2,049, and 4,358 tokens.

Four prompt policies were exercised with four examples in each GemRouter request:

- **A — conservative compression:** problem/reasoning could be shortened but answer had to remain verbatim. It produced strict JSON and preserved semantics, but only 3/4 examples fit; the long-output instruction example still serialized to 3,150 tokens.
- **B — general semantic compression:** all fields could be shortened. It produced substantially shorter text but failed strict JSON because of invalid backslash escapes, demonstrating that machine-parseability and plain-text math need to be explicit.
- **C — free curriculum simplification:** all 4/4 examples fit, but an evaluator flagged material scope changes in 3/4 examples, including replacing original numerical givens and replacing Tencent/JD data with a fabricated generic company example.
- **D — fidelity-first compression with minimal scope reduction:** all 4/4 responses parsed as strict JSON and all 4/4 passed the actual 2,048-token R-SFT serialization. Serialized lengths became 302, 380, 726, and 455 tokens. A batched Gemini comparison judge scored all four rewrites 5/5 for task preservation, factual fidelity, reasoning-pattern preservation, answer correctness, and small-model trainability, with no scope-change or unsupported-addition flags on this sample.

The live prompt-test evidence is stored under `artifacts/rsft-superior-reasoning-25k/prompt-tests/` in the working checkout. It is experimental evidence rather than a source-of-truth dataset artifact.

## Consequences

### Positive

- The 25k Superior target does not need to be cut roughly in half solely because of the model context limit.
- Compression is teacher-assisted rather than lossy token truncation.
- Batching four examples per request reduces GemRouter call count and is the default free-tier-conscious unit.
- The prompt strongly prefers preserving the original instance and only permits scope reduction when the required output itself is the blocker.
- Strict local validation makes provider formatting mistakes and overlength rewrites fail closed.

### Negative or limiting

- Rewritten examples are no longer byte-identical to the upstream Superior Reasoning samples and require provenance in the final dataset manifest.
- Gemini-based repair introduces additional API cost and teacher-model dependence.
- The four-example bake-off is strong prompt-selection evidence but not enough to assume zero semantic drift across ~13k rewrites; production conversion still needs deterministic validation and useful telemetry.
- A semantic judge can supplement but must not replace deterministic schema/token-fit checks.

## Validation

Before the repaired corpus is used for R-SFT training:

- process overlength records in batches of at most four;
- require strict JSON with exact IDs in input order and exactly `id/problem/reasoning/answer` output fields;
- reconstruct each rewritten R-SFT record and require it to fit the actual 2,048-token training serialization without truncation;
- record original and rewritten serialized token lengths and whether scope reduction was requested/retried;
- preserve unchanged records exactly when they already fit;
- report counts of unchanged, rewritten, parse-rejected, overlength-rejected, retried, and finally accepted records in the final manifest;
- rerun the R-SFT dataset/unit tests before bundle construction.

## Links

- `post_training/R-SFT/dataset/superior_reasoning.py`
- `tests/test_reasoning_sft_superior_reasoning.py`
- ADR 0102 — Use Superior Reasoning Stage 1 for the first real R-SFT corpus
- `artifacts/rsft-superior-reasoning-25k/prompt-tests/`
