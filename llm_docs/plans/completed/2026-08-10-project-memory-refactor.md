---
status: completed
date: 2026-08-10
---

# Project-memory documentation refactor

## Objective

Restore `llm_docs/` to a high-signal progressive-disclosure memory system for humans and LLM agents, research current agent-memory/documentation practice, classify legacy files by semantic role, preserve useful history, remove duplicate/stale working-memory copies, repair maps, and mechanically enforce the resulting structure.

## Constraints

- Preserve accepted project decisions and measured evidence.
- Do not rewrite historical evidence to match later interpretation.
- Keep current 2B experiment state/authorization unchanged.
- Do not silently promote old 20M/100M S0 numeric choices to a future post-2B SFT run.
- Prefer a small map and targeted retrieval over a larger always-read instruction set.

## Work completed

- Researched 2026 OpenAI/Anthropic/GitHub agent-instruction and memory practices, recent agent-memory research, Diátaxis, and ADR practice.
- Added `research/agent_memory_and_documentation_2026-08-10.md`.
- Recorded ADR 0031 for progressive-disclosure memory governance.
- Added `plans/` with active/completed lifecycle rules.
- Removed all substantive Markdown leakage from `llm_docs/` root.
- Reduced `current/` to `status.md` and `roadmap.md` only.
- Added `reference/gdn2_fla_backend.md` for the selected production FLA contract.
- Archived the canonical completed FLA investigation handoff and retired redundant qualification/blocker snapshots.
- Moved 20M/100M W&B and remote-recovery incidents into evidence.
- Archived completed 20M/100M planning/operational decisions.
- Archived the useful historical S0/SFT design packet and retired redundant architecture drafts.
- Added `reference/post_training_sft.md` to separate current implementation capability from future run authorization.
- Moved the obsolete 100M data-scaling plan out of runbooks.
- Rebuilt root/category maps and updated `AGENTS.md`.
- Strengthened `tests/test_project_memory.py` to enforce the root/current/index contracts.

## Discoveries

- The existing taxonomy was fundamentally good; the primary failure was lifecycle discipline rather than lack of categories.
- Several files named `decision` mixed approvals with proposals and were not formal ADRs.
- Completed FLA qualification documents remained in `current/` even after the investigation had closed and the 500M trajectory had completed.
- The repository already had a test asserting that the documentation root should contain only `README.md`; later root files were regressions against the intended design.
- Historical S0 decisions were explicitly scoped to the 20M/100M base, so keeping them visible at root risked accidental reuse after later pretraining scaling.

## Resulting memory model

```text
AGENTS.md -> current -> decisions -> relevant detail

llm_docs/
  README.md
  current/{status,roadmap}.md
  decisions/
  reference/
  runbooks/
  research/
  plans/{active,completed}/
  evidence/
  archive/
```

## Validation

Required final checks:

- root contains only `README.md` Markdown;
- current contains only `status.md` and `roadmap.md`;
- all category index links resolve;
- ADR 0031 is indexed;
- project-memory unit tests reflect the new plan lane/current invariant;
- no active source-of-truth link points to deleted FLA/current or root SFT documents;
- current 2B status/runbook remains unchanged in scientific intent.

Any validation gap should be recorded explicitly rather than treated as passed by assumption.
