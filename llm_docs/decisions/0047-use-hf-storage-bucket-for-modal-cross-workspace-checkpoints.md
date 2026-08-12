---
status: accepted
date: 2026-08-12
supersedes: 0046
---

# 0047 — Use a Hugging Face Storage Bucket for Modal cross-workspace checkpoints

## Context and problem statement

ADR 0046 established the correct high-level requirement: Hugging Face must be an integrated exact-resume transport for Modal so a new Modal account/workspace can continue a run whose original `small-llm-runs` Volume is not visible. Its first implementation used a private Git-backed model repository with rolling deletion and branch-history super-squashing to avoid accumulating every checkpoint forever.

During implementation, the current 2026 Hugging Face platform was re-audited. Hugging Face Storage Buckets are now a first-class mutable, non-versioned storage primitive explicitly intended for rapidly changing ML artifacts such as training checkpoints and optimizer states. That is a better fit than repeatedly rewriting and squashing a Git-backed model repository.

The final trained model still needs a normal versioned Hugging Face model repository under ADR 0044. The question here is only the transport used for live resumable trainer state.

## Considered options

- Keep ADR 0046's Git-backed model repository and super-squash branch history after every remote checkpoint.
- Keep only the Modal Volume and manually migrate checkpoint directories between Modal workspaces.
- Use a private Hugging Face Storage Bucket for mutable rolling trainer checkpoints, while reserving the normal model repository for the final published model artifact.

## Decision outcome

Chosen option: **use a private Hugging Face Storage Bucket as the integrated cross-workspace checkpoint transport for Modal.**

The operational contract is:

- Local verified joint checkpoints remain on `small-llm-runs` every 250 successful optimizer updates and at the final trainer boundary.
- A verified Hugging Face Storage Bucket checkpoint is published every 500 successful optimizer updates and at the final trainer boundary.
- The default bucket ID is derived from `SMALL_LLM_HF_REPO_ID` by appending `-checkpoints`. For example, `owner/small-llm-100m-qualification` maps to `owner/small-llm-100m-qualification-checkpoints`.
- `SMALL_LLM_HF_CHECKPOINT_BUCKET_ID` may override the derived bucket ID, but is not required for the standard workflow.
- The Modal runtime creates the private bucket if it does not yet exist.
- The existing two-phase checkpoint protocol remains authoritative: upload the complete checkpoint tree, read uploaded objects back and verify SHA-256 identities, publish the checkpoint manifest, then move `run/<run-id>/latest.json` only after the snapshot is valid.
- Bucket batch operations are treated as non-transactional. A partial upload is harmless because `latest.json` is not moved until verification succeeds. Restore follows only a verified latest pointer.
- After a new latest checkpoint is durable, mutable objects belonging to superseded checkpoint IDs are deleted. The current checkpoint and `latest.json` remain. No Git history or history-squash operation is involved in the production Modal path.
- Rolling transport does not retain a remote best-checkpoint history. It exists for exact continuation; validation/best-model analysis remains separate.
- On startup, if the current Modal workspace has no verified local run checkpoint, the runtime first tries the Storage Bucket `run/<run-id>/latest.json` checkpoint and restores it through the existing manifest-verifying restore path.
- For migration of the already-running 100M / 2B trajectory, two earlier Hugging Face model-repository layouts remain accepted as bootstrap sources when the bucket is empty: the short-lived ADR-0046 `run/<run-id>/latest.json` layout and the original `models/<run-id>/artifact.json` layout written by `modal/publish_hf.py`.
- A legacy bootstrap reuses the checkpoint's frozen microbatch rather than reprobeing and records the source-commit transition as infrastructure-only. Once a bucket checkpoint has been established, subsequent bucket restores again require the checkpoint source commit to match the launcher checkout.
- Dataset bytes are not duplicated into the checkpoint bucket. The checkpoint durability manifest records the run identity, dataset profile/run ID, dataset-manifest SHA-256, frozen microbatch, source commit, and an empty shard list because the immutable dataset remains independently reproducible from its frozen Kaggle/Modal dataset path.
- The Modal training image requires `huggingface-hub>=1.5,<2` for Storage Bucket APIs.

ADR 0044 remains in force for the final human-facing Hugging Face model artifact. After training completes, `modal/publish_hf.py --require-complete` may still publish the exact final verified checkpoint under the normal `models/...` model-repository namespace.

## Consequences

### Positive

- A new Modal account/workspace can restore exact trainer state from Hugging Face without access to the old Modal Volume.
- The remote storage abstraction now matches its workload: mutable checkpoints are stored as mutable objects rather than Git commits.
- The quota failure caused by accumulating many historical checkpoint commits is avoided; only the latest resumable bucket checkpoint is retained for the run.
- There is no destructive model-repository branch squash in the production Modal path.
- The existing exact-resume state remains intact: model, optimizer, WSD scheduler, FP16 scaler, RNG state, counters, and data cursor are all inside the verified joint checkpoint.
- The same-workspace 250-step Modal cadence and cross-workspace 500-step HF cadence provide distinct recovery tiers.
- The final model repository remains cleanly separated from transient trainer-state storage.

### Negative or limiting

- Remote checkpoint publication remains synchronous at 500-step boundaries and adds periodic wall-clock overhead.
- Cross-workspace recovery can recompute work after the newest 500-step remote boundary, while same-workspace recovery can use the newer 250-step Modal boundary.
- Storage Buckets require the current Hugging Face Hub client and consume the account's Hugging Face storage allowance.
- If both the old Hugging Face checkpoint copy and the old Modal workspace checkpoint are gone before the first bucket checkpoint is established, no software change can reconstruct the missing trainer state.

## Validation

This decision is satisfied when all of the following hold:

1. A Modal run creates or opens its private checkpoint bucket and publishes `run/<run-id>/latest.json` every 500 successful optimizer updates and at final completion.
2. Each pointer move occurs only after checkpoint object read-back hashes and checkpoint manifests verify.
3. After publishing a newer checkpoint, superseded checkpoint objects are deleted while the current checkpoint and latest pointer remain readable.
4. Starting the same run with an empty `small-llm-runs` Volume restores the bucket checkpoint, reuses its frozen microbatch, resumes the same W&B identity, and continues from the saved block cursor.
5. The pre-bucket `run/<run-id>/latest.json` repository layout and the older `models/<run-id>/artifact.json` layout can bootstrap the already-running 100M / 2B trajectory once when a surviving legacy copy exists.
6. The final completed model can still be published independently to the normal Hugging Face model repository required by ADR 0044.

## Links

- [`0044-publish-100m-2b-final-model-to-hugging-face.md`](0044-publish-100m-2b-final-model-to-hugging-face.md)
- [`0046-use-rolling-hf-as-modal-cross-workspace-checkpoint-transport.md`](0046-use-rolling-hf-as-modal-cross-workspace-checkpoint-transport.md)
- [`../runbooks/modal_training_launcher.md`](../runbooks/modal_training_launcher.md)
- [`../current/status.md`](../current/status.md)
