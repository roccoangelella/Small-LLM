---
status: accepted
date: 2026-09-04
owners: [Small-LLM]
---

# ADR 0143: remove IRE project state from the repository

## Decision

Remove the tracked `.ire/` directory from `main` and ignore `/.ire/` going
forward.

`llm_docs/` remains the canonical durable project-memory system for Small-LLM.
IRE-specific state, configuration, resources, caches, and agent memory are not
part of the model codebase or its reproducibility contract.

## Rationale

The repository already has a dedicated structured project-memory hierarchy under
`llm_docs/`. Keeping a second research-memory system under `.ire/` duplicates
state, introduces tool-specific metadata, and creates ambiguity about which
memory source is authoritative.

## Consequences

- Existing tracked `.ire/` files are deleted from `main`.
- `/.ire/` is ignored in `.gitignore`.
- Future local IRE state will not enter version control.
- Project decisions and durable status continue to be recorded under `llm_docs/`.
