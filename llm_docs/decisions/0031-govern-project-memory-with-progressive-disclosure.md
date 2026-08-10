---
status: accepted
date: 2026-08-10
supersedes: null
---

# 0031 — Govern project memory with progressive disclosure and lifecycle rules

## Context and problem statement

Small-LLM already had a structured project-memory taxonomy, but later work bypassed it: substantive Markdown files accumulated again at `llm_docs/` root, completed FLA investigation handoffs remained under `current/`, and multiple overlapping S0/SFT decision/design records looked more current than their original 20M/100M scope justified.

This creates a retrieval problem for both humans and LLM agents. Always-read context becomes larger, stale state competes with current state, and a document's location stops communicating whether it is authorization, current technical truth, observation, research, or history.

The user explicitly requested a full cleanup and heavy research on agent/project-memory structure. The research synthesis is recorded in `../research/agent_memory_and_documentation_2026-08-10.md`.

## Considered options

### Keep the existing taxonomy but tolerate misplaced files

Rejected. The taxonomy has little value if new files can bypass it. It also contradicts the repository's existing project-memory regression test that reserves the `llm_docs/` root for the map.

### Replace the repository memory with one large project-memory document

Rejected. A monolithic memory is easy to discover but expensive to load, difficult to keep fresh, and encourages duplication of decisions, measurements, commands, and technical contracts.

### Create a much deeper wiki-style hierarchy

Rejected. Extra hierarchy can reduce discoverability and create empty categories without solving lifecycle/freshness. Small-LLM should keep a shallow semantic taxonomy and rely on indexes plus targeted retrieval.

### Keep the existing semantic lanes, tighten lifecycle rules, and add a plan lane

Accepted. This preserves the useful existing model while making the always-read layer small and giving complex multi-session work a dedicated resumable artifact.

## Decision outcome

Project memory is governed as a progressive-disclosure repository-local system of record.

### Root map

`llm_docs/README.md` is the only Markdown file permitted directly under `llm_docs/`.

`AGENTS.md` remains a concise repository map rather than an encyclopedia. It points agents to current state, decisions, and only the relevant deeper documents.

### Working memory

`llm_docs/current/` contains only:

```text
status.md
roadmap.md
```

`status.md` records verified present state. `roadmap.md` records immediate gates, priorities, and open decisions. Detailed handoffs, qualification reports, research essays, and incident logs are forbidden from `current/` once their outcome can be summarized and linked.

### Durable decisions

`llm_docs/decisions/` contains one consequential durable choice per ADR. ADRs use explicit status and are not rewritten to hide changed reasoning; later choices supersede earlier records.

Unnumbered historical files whose names contain `decision` are not automatically current ADRs. Their lifecycle is determined by scope and present relevance.

### Current technical truth

`llm_docs/reference/` contains authoritative current technical contracts and implementation descriptions. Reference documents state the resulting system, not the full chronological investigation that produced it.

### Procedures

`llm_docs/runbooks/` contains executable or intentionally reproducible operating procedures. Superseded planning documents move to archive rather than remain beside active procedures.

### Research

`llm_docs/research/` contains source synthesis, comparisons, recommendations, assumptions, and unresolved analysis. Research is never authorization by itself.

### Execution plans

`llm_docs/plans/` is added for complex multi-step work:

```text
plans/active/
plans/completed/
```

A plan is a self-contained resumable execution artifact containing progress, discoveries, decisions encountered during execution, and validation. Plans do not replace ADRs, current state, reference, evidence, or runbooks. Completed plans are historical traces.

### Evidence

`llm_docs/evidence/` contains immutable measured or observed results. Later corrections to interpretation do not erase the original observation; the corrected interpretation is recorded separately and linked.

New evidence should normally live in a run- or topic-specific subdirectory.

### Archive

`llm_docs/archive/` contains superseded plans, completed investigation handoffs, historical operating decisions, and old design packets that remain useful for reconstruction but are not current authorization.

Git history may be used to retire redundant drafts when a canonical historical record plus an archive index preserves the useful state. We do not keep several near-identical working-tree copies solely because they once existed.

### Duplication and precedence

Prefer links over duplicating the same facts across layers.

Default precedence remains:

```text
current status -> accepted ADR -> current reference/runbook -> evidence -> research -> archive/journals
```

When a measured observation conflicts with an interpretation, evidence remains the observation while current/ADR/reference records the accepted interpretation.

### Documentation gardening

Perform memory gardening whenever a major state transition occurs, especially after:

- an experiment completes;
- an investigation is resolved;
- an architecture/runtime path is replaced;
- a complex active plan completes;
- a previously deferred stage becomes active.

The gardening pass should update current state, close/supersede decisions, move measurements to evidence, refresh reference/runbooks, archive stale handoffs/plans, remove duplicate current text, and run structural/link tests.

### Mechanical enforcement

Repository tests enforce at least:

- `llm_docs/*.md` contains only `README.md`;
- `llm_docs/current/*.md` contains only `status.md` and `roadmap.md`;
- required category indexes exist;
- index Markdown links stay inside `llm_docs/` and resolve;
- ADRs keep the required minimum structure;
- `AGENTS.md` stays small and points to the current memory map.

## Consequences

### Positive

- Agents load a small high-signal working set by default.
- A document's location communicates its epistemic role and lifecycle.
- Historical evidence/decisions remain available without competing with current state.
- Long-running work has a resumable plan surface without turning status into a scratchpad.
- Mechanical tests catch future root/current leakage immediately.
- Post-training design history is preserved without silently making old 100M-specific choices current for a later base checkpoint.

### Tradeoffs

- Moving/retiring legacy files changes historical paths; archive indexes and Git history provide reconstruction.
- Maintainers must perform explicit lifecycle transitions instead of leaving a document where it was first written.
- Some information is intentionally one link deeper, trading always-loaded convenience for context quality and freshness.

## Links

- [`../research/agent_memory_and_documentation_2026-08-10.md`](../research/agent_memory_and_documentation_2026-08-10.md)
- [`../README.md`](../README.md)
- [`../plans/README.md`](../plans/README.md)
- [`0001-use-structured-markdown-project-memory.md`](0001-use-structured-markdown-project-memory.md)
