---
status: accepted
date: 2026-08-12
---

# 0048 — Verify the Modal dataset in the active workspace before skipping upload

## Context and problem statement

The VPS-only dataset preparation flow established by ADR 0043 used a machine-local file, `~/small-llm-data/.modal-2b-b64-upload.json`, to suppress duplicate uploads after the block-64 derivative had once been sent to `small-llm-data:/datasets/modal-2b-b64-dataset-001`. The marker recorded the Volume name, destination, and local manifest SHA-256, but it did not identify the Modal workspace or environment that actually owned the remote Volume.

That assumption is invalid when Modal accounts/workspaces are rotated for credits or recovery. A marker written for workspace A can survive on the VPS while the active Modal credentials now point at workspace B, where the Volume or dataset does not exist. In that state the old helper could incorrectly report the dataset ready without contacting the active Modal workspace. A forced `modal volume put --force` also treated the CLI success line as sufficient evidence, which made an unexpectedly fast transfer difficult to distinguish from a real complete upload.

## Considered options

- Add only the active profile name to the existing local marker and continue using the marker as the skip authority.
- Always upload the approximately 4 GB derivative on every invocation.
- Make the active Modal workspace/environment's remote Volume state authoritative, while retaining a local marker only as a diagnostic cache record.

## Decision outcome

Chosen option: **remote state in the actually authenticated Modal workspace/environment is authoritative.**

The operational contract is:

- `python modal/prepare_dataset.py` resolves the Modal workspace from the credentials actually in use and the environment from the active Modal context.
- The canonical `small-llm-data` Volume is opened with create-if-missing semantics, so switching to a fresh Modal workspace does not require a separate Volume-creation command.
- Before deciding whether to upload, the helper inventories `/datasets/modal-2b-b64-dataset-001` in that active context.
- The expected remote inventory is derived from the locally verified block-64 manifest: every declared train/validation shard must exist remotely with the exact declared byte size, and the remote `manifest.json` must exist with the exact local SHA-256.
- A normal run skips upload only when that remote verification succeeds. The VPS marker never suppresses an active-workspace check.
- Switching Modal accounts/workspaces therefore causes the normal command to observe the new workspace's missing/incomplete destination and upload automatically.
- `--force-upload` retains its repair meaning: it re-uploads even if the current active workspace already verifies.
- After `modal volume put --force` returns success, the helper performs the same remote verification again. The command fails if the remote dataset still does not verify; the CLI success line alone is not accepted as readiness evidence.
- The local upload marker is retained at schema version 2 only for diagnostics. It records workspace, environment, destination, manifest SHA-256, and whether remote verification succeeded.
- This preparation check deliberately avoids downloading every shard back to the VPS. Exact remote path/size inventory plus the small manifest hash establishes that the expected artifact was materialized. The existing first-use Modal training verifier remains the stronger content-integrity boundary and checks the dataset against manifest shard checksums before production use.

## Consequences

### Positive

- Modal workspace rotation is now a normal rerun of `python modal/prepare_dataset.py`; no stale local marker can make a fresh workspace look ready.
- A missing `small-llm-data` Volume is provisioned automatically in the active environment.
- Suspiciously fast uploads are no longer judged by wall-clock time or the CLI message; remote state is checked immediately afterward.
- Reruns remain idempotent in an already-populated workspace because verified remote data skips the transfer by default.
- `--force-upload` remains available for explicit repair without weakening post-upload verification.

### Negative or limiting

- Each preparation run now makes Modal API calls and recursively inventories the canonical dataset destination, even when no upload is needed.
- The preparation check does not re-download and hash the approximately 4 GB of shard bytes. Full shard-content verification remains the training runtime's first-use responsibility.
- The local marker is no longer sufficient to establish readiness while offline; Modal availability/authentication is required for upload readiness.

## Validation

This decision is satisfied when all of the following hold:

1. A matching version-1 marker from an old workspace does not suppress upload when the active workspace lacks the dataset.
2. A fully matching dataset in the active workspace skips upload on the normal command.
3. `--force-upload` re-runs the upload even when the active workspace already verifies.
4. A successful `modal volume put` followed by a failing remote inventory/manifest check makes preparation fail and does not write a successful marker.
5. The helper reports the workspace and environment used for the verification in its operator output/JSON result.
6. Focused regression tests cover remote inventory matching, workspace-switch behavior, skip behavior, force behavior, and fail-closed post-upload verification.

## Links

- [`0043-prepare-modal-block64-corpus-on-vps.md`](0043-prepare-modal-block64-corpus-on-vps.md)
- [`../runbooks/vps_to_modal_2b_dataset.md`](../runbooks/vps_to_modal_2b_dataset.md)
- [`../current/status.md`](../current/status.md)