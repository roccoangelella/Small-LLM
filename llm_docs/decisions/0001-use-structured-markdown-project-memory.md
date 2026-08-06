---
status: accepted
date: 2026-08-06
supersedes: omnibus flat llm_docs layout
---

# 0001 — Use structured Markdown project memory

## Context and problem statement

`llm_docs/` had become a flat mixture of current status, technical contracts, runbooks, experiment plans, accepted results, superseded authorizations, and one-off agent tasks. Humans and agents had to inspect many similarly named files and could not reliably tell current truth from historical evidence.

Current agent-engineering guidance also warns against one large instruction manual: context is scarce, monolithic guidance rots quickly, and a short map pointing to structured sources of truth is easier to verify. Documentation frameworks distinguish task-oriented how-to material from reference and explanation, while ADR practice recommends one significant decision per numbered Markdown record.

## Considered options

- Keep the flat folder and maintain a larger index.
- Put all memory in one continuously updated file.
- Split memory by purpose and use a short `AGENTS.md` as the navigation map.

## Decision outcome

Chosen option: **split memory by purpose and use a short agent map**.

The repository uses:

```text
current/    present state and roadmap
decisions/  one ADR per durable choice
reference/  technical contracts
runbooks/   operational how-to documents
research/   investigations and external comparisons
evidence/   immutable completed results
archive/    superseded plans and historical scaffolding
```

`AGENTS.md` points to this structure rather than duplicating it.

## Consequences

### Positive

- Current state is cheap to load and difficult to confuse with historical plans.
- Decisions retain rationale and can be superseded explicitly.
- Runbooks, reference material, research, and evidence have distinct maintenance rules.
- Mechanical tests can enforce the layout and index links.

### Negative or limiting

- Existing links and habits must adapt to the new paths.
- Authors must classify new documents instead of dropping them into the folder root.
- The first migration preserves some verbose historical documents rather than rewriting them.

## Validation

The repository test suite checks that only `llm_docs/README.md` remains at the documentation root, required indexes exist, ADRs contain the required sections, and removed legacy paths do not return.

## Links

- [`../research/project_memory_research.md`](../research/project_memory_research.md)
- [`../README.md`](../README.md)
