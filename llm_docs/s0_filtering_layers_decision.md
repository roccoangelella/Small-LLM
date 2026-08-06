# S0 Filtering Layers Decision

_Last updated: 2026-08-06 Europe/Rome_

## Frozen decisions

The user approved the first two S0 data-selection layers:

1. **Source-level allowlisting and rejection.** Use a pinned small-model-oriented SmolTalk source family, retain exact source identities, and reject source components that are outside the current English general-chat scope.
2. **Deterministic hard content filtering.** Reject malformed, overlength, code/tool, advanced-math, long-reasoning, unsupported-role, duplicate, contaminated, and otherwise out-of-scope records using auditable deterministic checks.

These two layers are required before token-budget construction and SFT training.

## Layer 3 status

Per-record capability classification into exact labels such as `general_qa`, `simple_explanation`, or `everyday_conversation` is **not frozen**. The user questioned whether this classification would be reliable or useful enough to justify its complexity.

The current recommendation is to avoid a TF-IDF or other learned capability classifier for S0 v1 unless an explicit experiment requires category-level balancing or diagnosis. Source identities that already correspond to a capability, such as constraints, rewriting, summarization, and everyday conversation, may be used directly. The broad Magpie component may remain a general instruction pool after hard filtering rather than being assigned pseudo-precise capability labels.

## Candidate simplified source-level mixture

Pending explicit approval, replace the earlier five-way semantic quota with a source-level mixture:

```text
55% filtered Smol-Magpie-Ultra-Short general pool
10% dedicated everyday-conversations pool
10% Smol-Constraints
 5% Smol-Rewrite
 5% Smol-Summarize
15% frozen ClimbMix replay
```

Shares are measured by loss-bearing target tokens. The exact source availability and retained-token counts must be audited before freezing the final quotas.

## Required audits without capability labeling

Even if Layer 3 is omitted, the builder must report:

- source-level row and target-token counts before and after every filter;
- turn-count and prompt/response-length distributions;
- system-message frequency;
- duplicate and rejection counts with reason codes;
- random manual samples from each retained source;
- benchmark-decontamination results;
- final source-level mixture by loss-bearing target tokens.

A small post-hoc manual audit may estimate broad content coverage, but it does not become a training label or quota unless a later decision requires it.
