---
status: accepted
date: 2026-08-24
supersedes: null
---

# ADR 0119 — Publish the frozen R-SFT 1% corpus on main

## Context and problem statement

The unified R-SFT Stage-2 source scan is complete and the 1% reasoning corpus has been assembled from the frozen 16,716-row Stage-1-expanded base plus 4,979 already-context-fit Stage-2 rows. The resulting artifact contains 21,695 rows and projects 18,009,290 reasoning train targets against the requested 18,009,004 target. It was assembled with `include_adapted: false`, so no GemRouter-compressed over-context examples are present or required for the 1% corpus.

The finalized JSONL is 91,356,196 bytes, below GitHub's 100 MB per-file hard limit, and has SHA-256 `acb6a029d641bcf661beb24ddb2a4e7c1deadca47da3e5bd3acb1c7090e58042`.

## Decision outcome

Publish the frozen 1% reasoning corpus and its manifest directly on the repository `main` branch under `artifacts/rsft-superior-1pct/`.

Do not commit the much larger `artifacts/rsft-1pct-work/` intermediate source/over-context work directory. It remains reproducible working state rather than a canonical training artifact.

## Consequences

- The exact 1% R-SFT reasoning dataset is versioned alongside the code and project memory.
- Downstream launchers can reference a stable repository path and verify the frozen manifest/hash.
- The repository gains roughly 91 MB of Git history from the JSONL artifact.
- Over-context Stage-2 candidates remain outside Git history and should only be adapted if a future larger target actually requires them.

## Validation

The committed artifact must retain 21,695 JSONL rows, byte size 91,356,196, and SHA-256 `acb6a029d641bcf661beb24ddb2a4e7c1deadca47da3e5bd3acb1c7090e58042`. The manifest must record `include_adapted: false` and projected reasoning train targets of 18,009,290.

## Links

- [`0113-use-superior-reasoning-stage2-for-rsft-scaling.md`](0113-use-superior-reasoning-stage2-for-rsft-scaling.md)
- [`0115-refactor-rsft-dataset-production-into-source-adapters-generic-context-repair-and-main-builder.md`](0115-refactor-rsft-dataset-production-into-source-adapters-generic-context-repair-and-main-builder.md)
- [`../runbooks/rsft_stage2_scaling.md`](../runbooks/rsft_stage2_scaling.md)
