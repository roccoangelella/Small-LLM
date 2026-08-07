---
status: accepted
date: 2026-08-07
supersedes: null
---

# 0014 — Simplify the public Gated Delta Rule-2 repository documentation

## Context

The standalone public repository initially included several release-process and provenance markdown files (`LEGAL.md`, `THIRD_PARTY.md`, `CONTRIBUTING.md`, `SECURITY.md`, `PUBLISHING.md`, `RELEASE_CHECKLIST.md`, and `docs/PROVENANCE.md`). Those files were useful during release preparation but made the public repository feel like an internal LLM project workspace rather than a focused reusable library.

## Decision

Keep the public repository focused on the implementation and consolidate the useful public-facing documentation into `README.md`.

The public repository should retain:

- `README.md`;
- `LICENSE`;
- `CITATION.cff`;
- package metadata;
- source code;
- tests;
- examples;
- CI configuration.

Remove the standalone process/checklist/provenance markdown files listed above. The README should contain only concise information needed by users and contributors: purpose, install/API examples, precision/correctness status, limitations, provenance/licensing/non-affiliation, contribution guidance, citation, and license.

Do not expose Small-LLM-specific project history or internal workflow narrative in the standalone public repository. Preserve detailed provenance and project decision history internally in Small-LLM.

## Consequences

- The public repository is substantially smaller and easier to understand.
- Legal/provenance hygiene remains visible without a collection of process documents.
- Detailed internal decision and evidence history stays in Small-LLM rather than leaking into the standalone library's public presentation.

## Link

- Public repository: https://github.com/roccoangelella/Gated-Delta-Rule-2
