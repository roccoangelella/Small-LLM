---
status: research
reviewed: 2026-08-10
---

# Agent memory and repository documentation research

This note records the research used to reorganize Small-LLM's project memory on 2026-08-10. It is analysis, not project authorization. Durable choices derived from it belong in an ADR.

## Research question

How should a repository be documented when LLM coding agents are expected to work on it repeatedly across long-running experiments, while humans still need the documentation to remain understandable and maintainable?

The recurring failure modes in this repository were familiar ones: long handoff documents becoming de facto current state, historical decisions being dropped into the documentation root, investigation notes remaining in the always-read path after the investigation ended, and the same facts appearing in status, runbooks, decisions, and evidence with different freshness.

## Primary-source findings

### OpenAI: repository memory should be a map with progressive disclosure

OpenAI's 2026 agent-first engineering guidance describes the root agent instruction file as a map rather than a large manual. The reasons are directly relevant here: context is scarce, overly broad guidance dilutes the important constraints, large instruction files rot, and prose without mechanical validation is weak enforcement.

The recommended pattern is a short root map that points into a structured repository-local system of record. Detailed design documents, product specifications, references, active/completed execution plans, and generated material live behind that map and are opened only when relevant. The same guidance recommends mechanically checking documentation structure and links and doing recurring documentation gardening.

OpenAI's Codex material also emphasizes that agent context is finite. Repository knowledge should therefore be discoverable without being loaded into every task.

Sources reviewed:

- OpenAI, *Harness engineering: leveraging Codex in an agent-first world* (2026-02-11).
- OpenAI, *Unrolling the Codex agent loop*.
- OpenAI Cookbook, *Using PLANS.md for multi-hour problem solving*.
- OpenAI Codex documentation on `AGENTS.md`.

### Anthropic: keep always-loaded memory concise and scope detail

Anthropic's Claude Code memory documentation makes a similar separation between persistent project instructions and details that should be loaded only for particular work. It recommends concise, specific project memory; modular rules for scoped concerns; and task-specific skills rather than continuously loading every instruction into context.

Claude Code's auto-memory design is also explicitly hierarchical: a concise index is loaded automatically while detailed topic files are read on demand. The important design lesson is not a Claude-specific file name; it is that the index and the stored detail should have different jobs.

Sources reviewed:

- Anthropic, Claude Code documentation: *How Claude remembers your project* / memory documentation.
- Anthropic, Claude Code documentation for `.claude/rules/`, imports, and scoped project instructions.

### GitHub Copilot: repository and path-scoped instructions are complementary

GitHub's Copilot coding-agent documentation supports repository-wide instructions and narrower path-specific instructions, including `AGENTS.md`-style files with nearest-scope precedence. This reinforces the idea that global agent memory should contain only invariants that truly apply globally; subsystem-specific operating details should live nearer the subsystem or in linked documentation.

Source reviewed:

- GitHub Docs, Copilot coding agent / adding repository custom instructions.

### Agent-memory research: memory must support revision, freshness, and bounded retrieval

Recent 2026 research on stateful coding/agent workloads treats long-horizon agent memory as an evolving system rather than a read-only vector store. Repeated themes include:

- memories are constructed and revised over time;
- retrieval quality and memory freshness matter as much as storage;
- unregulated growth degrades usefulness;
- stale or contradictory memories require semantic revision or retirement;
- work traces can be useful, but only when shared memory supports read/write/verify rather than blindly accumulating every trace;
- long-term and short-term memory have different lifecycle requirements.

For repository documentation, this argues against putting every investigation handoff into `current/`. Current state should be aggressively refreshed; detailed observations should become immutable evidence; durable decisions should be explicitly versioned/superseded; and obsolete handoffs should be archived.

Research reviewed:

- *Agent Memory: Characterization and System Implications of Stateful Long-Horizon Workloads* (2026).
- *Remember Your Trace: ...* work on repository-level agent memory and work traces (2026).
- *Agentic Memory: ...* work on long-term/short-term memory management (2026).
- *Is Agent Memory a Database?* (2026).

### Diátaxis: separate procedural, reference, and explanatory purposes

The Diátaxis documentation framework separates tutorials, how-to guides, reference, and explanation because each serves a different user need. Small-LLM does not need to reproduce the framework literally, but the separation is useful:

- `runbooks/` are goal-oriented how-to procedures;
- `reference/` is authoritative description of the current system;
- `research/` is explanatory/investigative material and unresolved comparison;
- plans and tutorials should not be disguised as reference contracts.

Source reviewed:

- Diátaxis documentation framework, official site.

### ADR practice: one decision, short record, explicit supersession

Architecture Decision Record practice consistently favors small repository-local records containing one consequential decision, its context, considered alternatives, and consequences. Accepted decisions should not be rewritten to make history look cleaner; a later decision supersedes the earlier one.

This is particularly important for an agent memory because it distinguishes "what somebody once considered" from "what the project authorized" without requiring an agent to reconstruct intent from a chronological chat log.

Sources reviewed:

- Architecture Decision Records project / MADR documentation.
- Martin Fowler, *Architecture Decision Records* (2026 update).

## Derived design principles for Small-LLM

### 1. Separate working memory from long-term memory

`current/` is working memory. It should be tiny, high-freshness, and overwritten as reality changes. For Small-LLM this means only:

```text
current/status.md
current/roadmap.md
```

Long investigation handoffs do not belong there once their conclusion is known.

### 2. Keep the root map tiny

`llm_docs/README.md` is the navigation map. No substantive Markdown document belongs beside it. `AGENTS.md` is the repository-level agent map and should remain short enough to read on every task.

### 3. Store facts by semantic type, not chronology

A filename containing "decision" is not automatically an ADR, and a file created during an active investigation is not permanently "current". Placement is determined by what the document *does*:

- present verified state -> `current/`
- approved durable choice -> `decisions/`
- current technical contract -> `reference/`
- executable/reproducible procedure -> `runbooks/`
- unresolved analysis or source synthesis -> `research/`
- self-contained multi-step execution plan -> `plans/`
- measured/observed result -> `evidence/`
- superseded, abandoned, historical, or completed handoff -> `archive/`

### 4. Prefer links over duplicated facts

Status should state the outcome and point to the ADR/evidence/reference. An ADR should state the decision and point to measurements rather than copying the full benchmark. A reference should describe the current implementation contract without retelling the entire investigation chronology.

This reduces contradictory copies and makes freshness manageable.

### 5. Treat plans as living resumable artifacts

Complex multi-step work benefits from a self-contained execution plan with progress, discoveries, decisions, and validation. Plans should live under:

```text
plans/active/
plans/completed/
```

Only work complex enough to need resumability gets a plan. Completed plans become historical execution traces, not current project truth.

### 6. Preserve evidence; revise conclusions elsewhere

Measured observations are immutable except for explicit transcription corrections. If later investigation shows the interpretation was wrong, retain the original evidence and record the corrected interpretation in a later evidence note, ADR, or reference. This is exactly how the FLA false-positive diagnostic history should be handled.

### 7. Make freshness explicit

The always-read documents should carry review dates. Research should state whether it is open or concluded. ADRs have accepted/superseded/rejected status. Archive material must visibly state that it is not current authorization.

### 8. Enforce structure mechanically

Agent instructions are advisory; repository tests are enforceable. Useful invariants include:

- only `README.md` at `llm_docs/*.md`;
- only `status.md` and `roadmap.md` under `current/`;
- all category indexes resolve their local links;
- ADRs follow a minimum structural schema;
- the root `AGENTS.md` remains small and points to the current memory map.

### 9. Garden after state transitions

The best time to clean memory is when the project's state changes: an experiment completes, an investigation closes, an implementation supersedes wrappers, or a future plan becomes active. At that point:

1. update current state;
2. record/close the decision;
3. preserve measurements as evidence;
4. update the stable reference;
5. archive the old plan/handoff;
6. remove duplicate current text;
7. run structure/link tests.

This is more reliable than occasional large cleanups after many stale documents accumulate.

## Recommended Small-LLM hierarchy

```text
AGENTS.md                       # <= small map, always read
llm_docs/
  README.md                     # project-memory map only
  current/
    status.md                   # verified present state
    roadmap.md                  # immediate gates/open questions
  decisions/
    README.md
    NNNN-*.md                   # one durable choice each
  reference/
    README.md
    *.md                        # current technical contracts
  runbooks/
    README.md
    *.md                        # executable/reproducible procedures
  research/
    README.md
    *.md                        # unresolved analysis/source syntheses
  plans/
    README.md
    active/
    completed/
  evidence/
    README.md
    <run-or-topic>/...          # immutable observations
  archive/
    README.md
    <topic-or-campaign>/...     # superseded/historical records
journals/                       # informal study notes, non-authoritative
```

The hierarchy is intentionally shallow. The point is progressive disclosure, not taxonomy for its own sake.

## Retrieval order for an LLM agent

Default task startup should be:

1. `AGENTS.md`.
2. `llm_docs/current/status.md`.
3. `llm_docs/current/roadmap.md`.
4. `llm_docs/decisions/README.md` and only the relevant ADRs.
5. one relevant reference or runbook.
6. evidence/research/archive only when the task requires history, measurements, or unresolved rationale.
7. an active execution plan when the task names or creates one.

This keeps the common context small while preserving deep history for targeted retrieval.

## Conclusion

The existing Small-LLM taxonomy was fundamentally sound. The problem was lifecycle discipline: new documents bypassed the taxonomy, completed investigations stayed in working memory, and old SFT design notes looked more current than they were. The correct fix is therefore not a more elaborate wiki. It is stricter progressive disclosure, explicit lifecycle transitions, a dedicated plan lane for complex work, and mechanical enforcement of the map.