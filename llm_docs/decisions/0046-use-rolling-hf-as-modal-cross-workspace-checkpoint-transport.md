---
status: accepted
date: 2026-08-12
supersedes: 0045
---

# 0046 — Use rolling Hugging Face as the Modal cross-workspace checkpoint transport

## Context and problem statement

The approximately-100M / 2B trajectory originally treated the persistent `small-llm-runs` Modal Volume as the exact-resume transport and Hugging Face as a separate periodic/final backup destination. That works only while the same Modal workspace remains available. Moving the run to a different Modal account/workspace exposed two failures in the operational model:

1. Modal Volumes are workspace-scoped, so a new account cannot automatically see the prior `small-llm-runs` checkpoint chain.
2. The external ten-minute `modal/publish_hf.py` loop accumulated many distinct checkpoint paths in the private Hub repository and exhausted private storage.

The user wants Modal to use Hugging Face for checkpoint continuity in the same role it already plays for segmented Kaggle training, so a fresh Modal workspace can restore the latest verified remote state without manually copying the old Volume.

## Considered options

- Keep Modal Volume as the only exact-resume transport and manually copy run state between workspaces.
- Keep the external periodic `publish_hf.py` loop and manually delete old Hub checkpoints.
- Enable the trainer's verified two-phase Hugging Face checkpoint protocol inside Modal, use a model/run-specific identity, automatically restore it when the local Modal run Volume is empty, and bound remote history to the latest resumable checkpoint.

## Decision outcome

Chosen option: **Modal training uses Hugging Face as an integrated cross-workspace checkpoint transport while retaining the Modal Volume as the faster same-workspace durability layer.**

The operational contract is:

- Local verified joint checkpoints remain every 250 successful optimizer updates on `small-llm-runs`.
- A verified Hugging Face checkpoint is published every 500 successful optimizer updates and at the final trainer boundary.
- The Hugging Face checkpoint namespace uses the model/run identity (for example `run/100m-2b-data-001/...`), not the dataset-only identity, so reusing the same finite corpus for another model cannot collide.
- Publication remains two-phase: checkpoint bytes and their manifest are verified before `latest.json` moves.
- Modal uses **rolling latest-only retention** for this transport. After the new latest pointer is durable, older checkpoint folders for that run are removed from the branch head and the Hub branch history is super-squashed. The current latest pointer is read back after the squash before training continues.
- Rolling mode does not retain a remote best-checkpoint history. This transport exists for exact continuation; validation/best-model analysis remains separate.
- On startup, if the local Modal checkpoint directory has no verified checkpoint, the runtime first tries the rolling `run/<run-id>/latest.json` transport and restores it with full local-manifest/published-manifest verification.
- For migration of the already-running 100M / 2B trajectory only, the older `models/<run-id>/artifact.json` + `models/<run-id>/<step>/` layout written by `modal/publish_hf.py` is accepted as a legacy bootstrap source. Its frozen microbatch is reused instead of reprobed. A source-commit difference at this one legacy boundary is recorded as an infrastructure-only migration; once the rolling transport is established, later rolling restores again require the checkpoint source commit to match the launcher checkout.
- The remote checkpoint contains transport metadata including dataset identity, dataset-manifest hash, frozen microbatch, and source commit. Dataset bytes remain independently reproducible from the frozen Kaggle source and are not duplicated inside the checkpoint repository.
- `SMALL_LLM_HF_REPO_ID` must name a private repository intended for this run/checkpoint workflow because rolling cleanup super-squashes that repository branch history. Current file contents outside the old checkpoint folders are retained, but prior Git history is intentionally not preserved.

ADR 0044 remains in force for the final Hugging Face model artifact. The integrated rolling checkpoint transport satisfies resumability; `modal/publish_hf.py --require-complete` may still be used after completion to materialize the human-facing final `models/...` artifact namespace.

## Consequences

### Positive

- A new Modal account/workspace can resume directly from Hugging Face without access to the old `small-llm-runs` Volume.
- The exact model/optimizer/scheduler/scaler/RNG/data-cursor checkpoint remains verified before restore.
- Model/run-specific names eliminate the legacy cross-model dataset-key collision.
- Rolling retention prevents the current Hub branch from containing dozens of full 100M optimizer checkpoints simultaneously.
- History squashing removes superseded checkpoint commits from the active branch history rather than reproducing the unbounded ten-minute backup pattern.
- The local 250-step Modal Volume cadence remains available for cheap same-workspace retries, while the 500-step Hub cadence bounds cross-workspace recomputation.

### Negative or limiting

- Hugging Face publication is synchronous at the 500-step boundaries and therefore adds periodic wall-clock overhead.
- A cross-workspace failure can require recomputing up to the work since the latest 500-step remote boundary; a same-workspace Modal retry can still use the newer 250-step local checkpoint.
- `super_squash_history` is intentionally destructive to Git history. The configured repository should therefore be treated as an artifact/checkpoint store rather than a repository whose historical commit graph must be preserved.
- If the Hugging Face repository has already been deleted and the old Modal workspace is no longer accessible, no software change can reconstruct a checkpoint that exists nowhere. One surviving copy is required for the initial migration.

## Validation

The decision is satisfied when all of the following hold:

1. A Modal trainer run publishes `run/<run-id>/latest.json` every 500 successful updates and at final completion.
2. The latest pointer resolves to a checkpoint whose upload hashes and manifests verify.
3. After a second remote publication, the prior checkpoint folder is absent from the branch head, the branch is squashed, and the current latest pointer still reads back to the new checkpoint.
4. Starting the same run with an empty `small-llm-runs` Volume restores the rolling HF checkpoint, reuses its frozen microbatch, resumes the same W&B identity, and continues from the checkpoint block cursor.
5. The pre-ADR-0046 `models/<run-id>/artifact.json` layout can bootstrap the already-running 100M / 2B run once when the new Modal workspace is empty.

## Links

- [`0044-publish-100m-2b-final-model-to-hugging-face.md`](0044-publish-100m-2b-final-model-to-hugging-face.md)
- [`0045-run-periodic-hf-backups-only-while-modal-training-is-live.md`](0045-run-periodic-hf-backups-only-while-modal-training-is-live.md)
- [`../runbooks/modal_training_launcher.md`](../runbooks/modal_training_launcher.md)
- [`../current/status.md`](../current/status.md)
