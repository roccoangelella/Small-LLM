---
status: accepted
date: 2026-08-12
supersedes:
  - 0047
  - 0052
---

# 0054 — Unify Modal checkpoints on the Hugging Face model repository

## Context and problem statement

The 20M Kaggle trajectories already use the verified two-phase Hugging Face model-repository checkpoint protocol under `run/<run_id>/...`, while Modal was migrated to a separate Hugging Face Storage Bucket for rolling cross-workspace checkpoints. The separate bucket avoided Git history accumulation, but it created a second checkpoint transport and a second evaluation entrypoint. The completed 100M/2B checkpoint was subsequently moved into the human-facing model-repository namespace `models/100m-2b-data-001/step-00015267`.

The project now needs one predictable model-repository structure for 20M and 100M checkpoints without changing model geometry, optimizer math, schedule semantics, dataset ordering, or checkpoint bytes. Dataset object storage is a separate concern: the 10B rolling dataset still benefits from an HF Storage Bucket and is not changed by this decision.

## Considered options

- Keep the HF checkpoint Storage Bucket and maintain separate bucket-aware evaluation tooling.
- Use only stable `models/<run_id>/<step>` artifacts and give up periodic cross-workspace exact resume.
- Use one HF model repository for both live two-phase checkpoint durability and stable final model artifacts.

## Decision outcome

Chosen option: **use one Hugging Face model repository for checkpoints and final model artifacts**.

The canonical structure is:

```text
run/<run_id>/latest.json
run/<run_id>/checkpoints/<checkpoint_id>/last/...
models/<run_id>/artifact.json
models/<run_id>/<checkpoint_id>/...
```

`run/...` is the rolling exact-resume namespace. Modal continues to publish every 500 successful updates plus final, uses rolling latest-only cleanup, and the Git-backed store prunes superseded checkpoint paths and squashes history after the verified latest pointer advances. `models/...` is the stable human-facing artifact namespace. A completed Modal run automatically publishes its final verified checkpoint there.

The previous checkpoint bucket becomes a **legacy restore source only**. New Modal checkpoint writes must target `SMALL_LLM_HF_REPO_ID`. `SMALL_LLM_HF_CHECKPOINT_BUCKET_ID` is no longer part of the active checkpoint contract.

The HF dataset bucket remains active for the rolling 10B dataset transport because large mutable shard/object access is the workload Storage Buckets are designed to serve.

## Consequences

### Positive

- 20M and 100M checkpoints share one model-repository protocol and one repository identity.
- Standard Hub inspection no longer requires understanding a separate checkpoint bucket.
- Cross-workspace exact resume is retained through the existing two-phase `run/...` protocol.
- Completed models have a stable `models/...` location suitable for evaluation and later post-training.
- The manually moved 100M/2B checkpoint can be evaluated directly from `models/100m-2b-data-001/step-00015267`, even if `artifact.json` has not yet been written.

### Negative or limiting

- Frequent model-repository writes use Git/LFS semantics rather than mutable object-store semantics.
- Rolling cleanup must continue to prune old checkpoint paths and squash repository history so periodic checkpoint publication does not accumulate unbounded Git history.
- A stable final model may coexist with the live `run/...` checkpoint snapshot, so the repository intentionally contains both operational and human-facing namespaces.

## Validation

- Modal dry-run/import preflight loads the model-repository checkpoint adapter.
- A generated online trainer command contains `--remote-checkpoint-repo`, `--remote-create-repo`, and `--remote-rolling-latest-only`, and does not contain `--remote-checkpoint-bucket`.
- A new Modal checkpoint round-trips through `run/<run_id>/latest.json` and restores exactly in an empty Modal workspace.
- Source-commit mismatch remains fail-closed for a live model-repository resume.
- A completed Modal run publishes `models/<run_id>/<checkpoint_id>` and updates `models/<run_id>/artifact.json`.
- The stable-model prompt-suite entrypoint downloads and verifies the manually moved 100M/2B checkpoint and runs the frozen six-case short diagnostic unchanged.

## Links

- [`0047-use-hf-storage-bucket-for-modal-cross-workspace-checkpoints.md`](0047-use-hf-storage-bucket-for-modal-cross-workspace-checkpoints.md)
- [`0052-evaluate-modal-rolling-checkpoints-directly-from-hf-bucket.md`](0052-evaluate-modal-rolling-checkpoints-directly-from-hf-bucket.md)
- [`../runbooks/modal_training_launcher.md`](../runbooks/modal_training_launcher.md)
- [`../../modal/model_repo_checkpoint.py`](../../modal/model_repo_checkpoint.py)
- [`../../trainer/model_artifact.py`](../../trainer/model_artifact.py)
