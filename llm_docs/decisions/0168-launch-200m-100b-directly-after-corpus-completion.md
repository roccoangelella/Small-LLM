---
status: current
last_reviewed: 2026-09-08
---

# ADR 0168 — Launch 200M/100B directly after corpus completion

## Context

ADRs 0164 and the current roadmap required separate Modal H100 and Beam RTX4090 provider smoke qualification before the long 200M/100B dispatch. The production Modal path is already inherited from the previously exercised training stack, while the accepted 200M changes are width-only geometry changes and the 100B profile remains fail-closed on completed-dataset staging.

## Decision

For the Modal 200M/100B trajectory, remove the separate pre-launch H100 qualification run as a launch gate. Once `100b-b64-dataset-001` is fully published and its terminal READY/frontier state is verified, launch the real `200m-100b-data-001` production run directly on Modal H100.

The normal production runtime remains unchanged: its first real container still performs the built-in microbatch qualification and all existing dataset, finite-loss, memory, checkpoint, configuration-drift, and exact-resume safeguards remain active. This decision removes only the requirement for a distinct disposable H100 smoke/qualification run before production.

The completed 100B corpus remains a hard gate. Do not dispatch the long run while `producer_complete` is false, the final manifest hash is absent, or the READY frontier does not cover the frozen training horizon.

## Consequences

- No separate Modal H100 smoke job is required before `200m-100b-data-001` begins.
- The first production launch itself is the first live execution of the expanded 200M geometry on H100.
- Built-in automatic microbatch probing remains part of that production launch and is not bypassed.
- Dataset completion/terminal-frontier verification remains mandatory before launch.
- Beam qualification is not required to start the Modal trajectory; Beam remains an authorized recovery/alternate provider under the existing provider boundary.
