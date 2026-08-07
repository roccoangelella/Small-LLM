---
status: accepted
date: 2026-08-07
supersedes: null
---

# 0007 — Render teacher-forced examples as readable ground truth

## Context and problem statement

The teacher-forced held-out confidence diagnostic correctly measures model probabilities at GPT-2 BPE token positions, but its first terminal format displayed only the preceding context and an isolated target token. Byte-level BPE fragments such as `" ple"` or `"lect"` are not meaningful to a human without adjacent ground-truth tokens, making valid examples appear disconnected or corrupted.

For example, a source phrase such as `This is a huge crowd pleaser.` may use a target BPE piece `" ple"`. Printing only `context='This is a huge crowd' true=' ple'` obscures that the target belongs to the full word `pleaser`.

## Considered options

- Keep the isolated-token display and require manual BPE interpretation.
- Decode several future tokens but leave the target fragment unexpanded.
- Reconstruct surrounding ground-truth text, expand the target BPE piece to its containing lexical word when possible, bracket that readable span, and separately retain the exact target token metadata.

## Decision outcome

Chosen option: **reconstruct readable ground-truth context and bracket the full lexical span containing the target token**.

Representative teacher-forced examples should use this terminal shape:

```text
TEXT: "...but don't want to put a lot of time in. This is a huge crowd [pleaser]."
TARGET TOKEN: " ple"  (token 12345)
MODEL TOP-1: " of"  p=15.5%
TARGET: " ple"  p=0.00004%, rank=42,245
```

The bracketed text is for human interpretation only. The measured unit remains the exact GPT-2 BPE target token shown separately with its token ID, probability, and vocabulary rank.

## Consequences

### Positive

- Representative failures can be understood as normal text rather than isolated tokenizer fragments.
- The display separates the human-readable word/span from the exact token-level measurement.
- Existing probability, rank, loss, perplexity, and raw-token calculations remain unchanged.
- The full raw per-token JSON records remain token-level and are not replaced by word-level scoring.

### Negative or limiting

- A BPE target that is whitespace or punctuation may not have a containing lexical word; in that case the display brackets the target span itself.
- Word-span expansion is a visualization heuristic and must not be interpreted as a word-level model metric.

## Validation

Unit tests must verify that a target piece such as `" ple"` inside `pleaser` renders as `[pleaser]`, and that a token beginning inside a word such as `"lect"` inside `selection` expands to `[selection]`. The exact target token and probability/rank fields must remain unchanged.

## Links

- [`../runbooks/post_pretraining_prompt_suite.md`](../runbooks/post_pretraining_prompt_suite.md)
- [`../../trainer/teacher_forced_diagnostic.py`](../../trainer/teacher_forced_diagnostic.py)
