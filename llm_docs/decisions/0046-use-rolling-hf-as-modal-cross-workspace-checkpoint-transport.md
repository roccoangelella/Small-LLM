---
status: superseded
date: 2026-08-12
supersedes: 0045
superseded_by: 0047
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

Chosen option at this stage: **integrate Hugging Face into Modal's exact-resume path and retain only the latest remote checkpoint.** The first implementation used a private Git-backed model repository, deleting superseded checkpoint folders and super-squashing branch history after a verified latest-pointer move.

The high-level cross-workspace-resume decision remains valid, but the Git-backed storage mechanism was superseded before production use by ADR 0047 after the current Hugging Face platform was re-audited. Storage Buckets provide mutable, non-versioned object storage intended for training checkpoints and avoid the need to manipulate model-repository Git history.

## Consequences

### Positive

- Established that a new Modal workspace should be able to resume from Hugging Face rather than depending exclusively on a workspace-local Volume.
- Reused the existing two-phase checkpoint publication and manifest-verified restore protocol rather than inventing a second checkpoint format.
- Established model/run-specific checkpoint identity and a bounded remote-retention requirement.

### Negative or limiting

- The initial implementation forced mutable checkpoint traffic through a Git-backed model repository.
- Keeping storage bounded required destructive branch-history squashing.
- The normal Hugging Face model repository is a better fit for the final model artifact than for rapidly changing optimizer checkpoints.

## Supersession

ADR 0047 keeps the 250-step local Modal checkpoint layer, the 500-step cross-workspace HF layer, the two-phase verified pointer protocol, and the legacy-bootstrap behavior, but moves the production remote checkpoint store to a private Hugging Face Storage Bucket. The final model remains in the normal Hugging Face model repository under ADR 0044.

## Links

- [`0044-publish-100m-2b-final-model-to-hugging-face.md`](0044-publish-100m-2b-final-model-to-hugging-face.md)
- [`0045-run-periodic-hf-backups-only-while-modal-training-is-live.md`](0045-run-periodic-hf-backups-only-while-modal-training-is-live.md)
- [`0047-use-hf-storage-bucket-for-modal-cross-workspace-checkpoints.md`](0047-use-hf-storage-bucket-for-modal-cross-workspace-checkpoints.md)
- [`../runbooks/modal_training_launcher.md`](../runbooks/modal_training_launcher.md)
- [`../current/status.md`](../current/status.md)
