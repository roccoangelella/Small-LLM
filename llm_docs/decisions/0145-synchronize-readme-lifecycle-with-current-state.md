---
status: accepted
date: 2026-09-04
supersedes: null
---

# 0145 — Synchronize README lifecycle with current project state

## Context and problem statement

The repository's active README layer drifted behind the project system of record. The root README still described the 20M/2B pretraining point as active even though 20M/2B, 100M/2B, and 100M/10B pretraining are complete. `beam/README.md` still presented the original 100M/10B Beam launch controls as current instructions, `kaggle/README.md` advertised a retired one-off 20M script, and `llm_docs/README.md` labeled completed 100M/10B runbooks as current operational entrypoints. `current/roadmap.md` had the same lifecycle drift and still described an in-progress 100M/10B continuation.

The project-memory precedence contract already makes `current/status.md` the authoritative present-state source. README documents need to remain useful without becoming independent, stale copies of that state.

## Considered options

- Leave README prose unchanged and require readers to notice that `current/status.md` overrides it.
- Delete completed-run commands and provider procedures from README files entirely.
- Duplicate detailed current status into every README and manually keep all copies synchronized.
- Keep concise current-state summaries in active README files, clearly label completed-run procedures as reproduction/history, and defer detailed live state to `current/status.md` and `current/roadmap.md`.

## Decision outcome

Chosen option: **keep active README files lifecycle-aware and subordinate to project memory**.

- The root README must describe the current project phase rather than an already-completed scaling point.
- Completed experiment/provider procedures may remain in README files when they are still useful for reproduction, recovery, or provider qualification, but they must be labeled as such and must not read as current launch authorization.
- Stable operator surfaces should be documented preferentially over retired or low-level one-off scripts.
- `llm_docs/README.md` must distinguish current operational entrypoints from completed-run reproduction procedures.
- `current/roadmap.md` is aggressively refreshed when an experiment changes lifecycle instead of retaining an obsolete active-run narrative.
- `llm_docs/decisions/README.md` should expose the newest accepted decisions so the navigation layer does not lag the decision files themselves.
- `current/status.md` remains the precedence source when any README and current-state claim disagree.

## Consequences

### Positive

- Repository onboarding reflects the actual completed 100M/10B pretraining state and current SFT/probe work.
- Historical execution knowledge remains discoverable without being confused with active authorization.
- Stable launcher interfaces receive more prominence than implementation files that may be retired or renamed.
- The project-memory navigation layer and high-freshness files agree on experiment lifecycle.

### Negative or limiting

- Lifecycle transitions now require a small documentation-gardening pass in addition to updating status/evidence.
- README files intentionally contain less detailed live-run state; readers needing exact current checkpoints or gates must follow the project-memory links.

## Validation

- All local Markdown links in the documentation indexes must resolve under `tests.test_project_memory`.
- README commands must point to files or installed entrypoints that exist in the current tree.
- `current/status.md`, `current/roadmap.md`, the root README, and provider/workspace README lifecycle wording must not describe the completed 100M/10B pretraining trajectory as active.
- The decision index must include ADRs 0140–0145 in the appropriate current navigation sections.

## Links

- [`../current/status.md`](../current/status.md)
- [`../current/roadmap.md`](../current/roadmap.md)
- [`../README.md`](../README.md)
- [`0144-consolidate-100m-10b-probes-and-test-low-lr-tail.md`](0144-consolidate-100m-10b-probes-and-test-low-lr-tail.md)
