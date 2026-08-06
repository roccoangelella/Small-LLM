---
status: accepted
date: 2026-08-06
supersedes: null
---

# 0002 — Freeze eval_core_v1 and the unified evaluation CLI

## Context and problem statement

The existing approximately-10k-target-token validation sample is enough to show learning but too small for stable per-cluster analysis. The project also needs intrinsic metrics and the existing human-readable prompt answers to remain part of one checkpoint evaluation flow.

## Considered options

- Continue using the tiny per-run validation sample.
- Hold out a global token quota without cluster or document floors.
- Build permanent fast and full stratified suites with both document and token floors, then run metrics and prompts through one CLI.

## Decision outcome

Chosen option: **permanent nested fast and full suites plus one unified CLI**.

For each of the 19 retained clusters:

```text
fast: at least 32 documents and 16,384 scored target tokens
full: at least 256 documents and 131,072 scored target tokens
```

Selection comes only from the already-frozen deterministic validation partition. The ordinary complete commands are:

```text
small-llm-eval fast
small-llm-eval full
```

Both run intrinsic metrics and the existing `PROMPT_CASES` by default.

## Consequences

### Positive

- Results are comparable across token budgets and later model sizes.
- Per-cluster reporting has independent-document coverage.
- Prompt outputs, metrics, checkpoint identity, and eval identity live in one result bundle.

### Negative or limiting

- The corpus must be built once from the remote source and stored durably.
- Full evaluation is materially more expensive than the old tiny validation pass.
- Runtime and confidence intervals still require T4 measurement before acceptance.

## Validation

Build and verify the immutable corpus, benchmark both suites on a T4, and evaluate the accepted 10M checkpoint before using the scorecard as a scale-decision gate.

## Links

- [`../reference/eval_core_v1_design.md`](../reference/eval_core_v1_design.md)
- [`../runbooks/eval_core_v1_runbook.md`](../runbooks/eval_core_v1_runbook.md)
