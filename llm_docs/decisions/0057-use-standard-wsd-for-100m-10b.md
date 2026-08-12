---
status: accepted
date: 2026-08-12
supersedes: null
---

# 0057 — Use the standard WSD schedule for the fresh 100M / 10B run

## Context

ADR 0050 authorizes a fresh approximately-100M / 10B-token trajectory after the 100M / 2B behavioral gate, and ADR 0053 defines its rolling one-GiB Hugging Face dataset transport. ADR 0053 intentionally left the exact 10B WSD horizon policy unresolved.

The existing finite-dataset qualification system uses the project's standard one-pass WSD fractions: 5% warmup, 75% stable, and 20% decay, with a 0.1 minimum LR ratio. The user has now explicitly accepted using this same standard WSD policy for the 10B run rather than introducing a special horizon rule.

## Decision

Use the existing standard WSD contract for `modal-10b-b64`:

- warmup: 5% of the final planned optimizer-update/token horizon;
- stable: 75%;
- decay: 20%;
- minimum LR ratio: 0.1;
- derive exact update/token boundaries from the same finite-plan machinery used by the other scaling runs.

For a nominal approximately-10B target this is approximately 0.5B warmup tokens, 7.5B stable tokens, and 2.0B decay tokens; the exact boundaries remain those produced from the final deterministic training plan and block geometry.

The approximately-5B capability checkpoint from ADR 0050 therefore lies in the stable phase and remains an intermediate checkpoint rather than a terminal 5B WSD endpoint.

## Consequences

- No special 10B scheduler implementation or new scheduler qualification is required before launch.
- The 10B run remains directly consistent with the project's existing WSD convention.
- The scheduler choice is no longer a launch blocker.
- This decision does not change ADR 0050's behavioral gate or ADR 0053's dataset transport semantics.

## Links

- [`0050-scale-100m-to-fresh-10b-with-5b-capability-gate.md`](0050-scale-100m-to-fresh-10b-with-5b-capability-gate.md)
- [`0053-stream-10b-through-one-gib-hf-shards-and-cpu-stage-before-h100.md`](0053-stream-10b-through-one-gib-hf-shards-and-cpu-stage-before-h100.md)
- [`../current/status.md`](../current/status.md)
