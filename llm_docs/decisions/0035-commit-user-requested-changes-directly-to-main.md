# ADR 0035: Commit user-requested changes directly to main

Date: 2026-08-10
Status: Accepted

## Decision

For changes explicitly requested by the project owner in ChatGPT project work, commit and push the implementation directly to the repository `main` branch instead of opening a pull request, unless the owner explicitly asks for a PR or branch-based review flow.

## Context

During the SFT launcher log-alignment change, a pull request was opened after implementing the requested change. The project owner clarified that this repository workflow should use direct commits to `main` rather than PRs for such requests.

## Consequences

- Do not create a PR by default for owner-requested project changes.
- Commit and push the requested implementation directly to `main`.
- Continue to record durable project decisions under `llm_docs/`.
- Use a PR only when the owner explicitly requests one or when a separate review workflow is specifically required.
