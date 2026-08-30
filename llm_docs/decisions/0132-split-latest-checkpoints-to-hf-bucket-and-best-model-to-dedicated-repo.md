---
status: accepted
date: 2026-08-30
supersedes:
  - 0055
---

# 0132 — Split latest checkpoints to an HF Bucket and best checkpoints to dedicated model repositories

## Context and problem statement

ADR 0055 put live exact-resume checkpoints and stable model artifacts into the same Git-backed Hugging Face model repository. That simplified naming, but a long-running trajectory repeatedly commits roughly checkpoint-sized payloads. Even after pruning paths and super-squashing history, the active 100M/10B Modal run hit the private repository storage limit while publishing step 61,500. The training state itself was valid; the failure was checkpoint transport pressure from using a versioned model repository as mutable rolling storage.

There are also two different retention semantics hiding behind the word “checkpoint”:

- **latest** is operational recovery state. Only the newest verified exact-resume copy is useful for the rolling lane, so mutable object storage is the natural fit.
- **best** is a selected model snapshot. It should survive independently of later worse checkpoints and remain a normal Hugging Face model repository that humans and downstream tools can inspect.

Keeping both roles in one Git history makes the operational `latest` cadence pay model-repository history costs and makes “best” retention depend on rolling cleanup behavior.

## Considered options

- Keep both rolling `latest` and selected `best` in the shared Git-backed model repository and continue pruning/super-squashing history. This preserves one namespace but repeats the quota/history failure mode.
- Put both `latest` and `best` in a Storage Bucket. This avoids Git history, but removes the normal Hugging Face model-repository surface intended for selected model artifacts.
- Keep `latest` in mutable Bucket storage and publish `best` to a dedicated per-run model repository that is deleted/recreated on strict improvement. This matches each retention role to its storage semantics and bounds model-repository history.

## Decision outcome

Use separate Hugging Face transports by retention role:

```text
HF Storage Bucket
  run/<run_id>/latest.json
  run/<run_id>/checkpoints/<checkpoint_id>/last/...
  -> exact-resume latest only

Dedicated HF model repository per run
  best_model.json
  models/<run_id>/artifact.json
  models/<run_id>/<checkpoint_id>/...
  -> strict validation-loss best only
```

For Modal, the default checkpoint bucket is derived from `SMALL_LLM_HF_REPO_ID` as `<owner>/<name>-checkpoints`; `SMALL_LLM_HF_CHECKPOINT_BUCKET_ID` may explicitly override it. The rolling two-phase publisher still uploads and independently reads back the checkpoint before moving `latest.json`, then deletes superseded objects for that run.

The best model repository is dedicated to exactly one run. By default it is derived as `<owner>/<base>-best-<run_id>` after safe run-ID normalization. `SMALL_LLM_HF_BEST_MODEL_REPO_ID` may override the repository only when its name remains dedicated to the active run. Best selection uses held-out validation loss, with lower loss better. Publication occurs only on a strict improvement.

**Every update of an existing best-model repository deletes and recreates that marker-verified dedicated repository before publishing the new best checkpoint.** This deliberately prevents Git/LFS commit history from stacking across best updates. Before deletion, `best_model.json` must prove that the repository is the Small-LLM dedicated best repository for the same run. An absent, malformed, differently owned, or shared repository fails closed and is never deleted by this path.

Legacy model-repository `run/...` checkpoints remain restore/migration sources while old trajectories are moved. A legacy latest checkpoint must be byte/manifest verified and durably published to the Storage Bucket before its old checkpoint namespace can be considered for removal. Shared model repositories that contain stable `models/...` artifacts, other run namespaces, SFT/R-SFT state, or other provider state must not be deleted wholesale.

Stable completed artifacts are unchanged by this decision. They remain model artifacts; only rolling exact-resume state moves to mutable Bucket storage.

## Modal implementation

The Modal adapter keeps the trainer's `--remote-checkpoint-bucket`, `--remote-create-bucket`, and `--remote-rolling-latest-only` flags for exact resume, and adds `--best-model-repo` plus mandatory `--best-model-recreate` for model selection. The trainer seeds the replacement threshold from the exact-resume state's persisted `best_validation_loss` and any existing dedicated-model marker, then publishes only after validation strictly beats that historical threshold. If the dedicated repo is missing and the resumed checkpoint's own verified validation loss exactly equals the persisted historical best, startup may repair the missing best repo from that checkpoint. A merely available checkpoint that is worse than the persisted best is never relabeled as run-wide best.

The deep-decay migration gate compares local Volume, new Bucket latest, and legacy model-repository latest. It restores the newest verified state, republishes it to the Bucket when needed, and verifies the resulting Bucket pointer before H100 allocation.

Beam and Kaggle must adopt the same role split in their provider-specific launchers/adapters. Their implementation is a separate rollout step; they must reuse the same trainer best-model contract rather than invent a provider-specific model publication protocol.

## Consequences

### Positive

- Frequent rolling checkpoints no longer create Git/LFS history.
- Cross-workspace exact resume still uses the verified two-phase latest-pointer protocol.
- Best-model retention is independent of the most recent checkpoint.
- A best repository has bounded history: one recreated repository and one publication commit for the current best.
- Repository deletion is guarded by an explicit ownership marker and run identity.
- Existing stable model artifacts are not coupled to mutable checkpoint cleanup.

### Negative or limiting

- A run now has two HF identities to understand: one checkpoint Bucket and one best-model repository.
- Replacing a best model intentionally deletes the previous best repository before recreating it; a failed upload therefore leaves no stale repository masquerading as current best, but the previous remote best is not preserved there.
- Legacy mixed model repositories cannot be deleted wholesale until every valuable namespace is separately classified or migrated.
- Existing best state can only be initialized from a checkpoint whose validation metric and checkpoint bytes are both available and verified; W&B metrics alone are not enough to reconstruct a deleted checkpoint.

## Validation

- Modal online commands contain `--remote-checkpoint-bucket`, `--remote-create-bucket`, and `--remote-rolling-latest-only`, and do not contain `--remote-checkpoint-repo` for latest state.
- Modal online commands contain `--best-model-repo` and `--best-model-recreate`.
- Best publication requires held-out validation and a stable run ID.
- A strict validation-loss improvement publishes the verified local checkpoint; equal or worse loss does not replace the model.
- An existing best repository is marker-verified, deleted, recreated, and published in one new commit.
- An unmarked or differently owned model repository is never deleted.
- Bucket latest is independently read back after migration before the legacy source is eligible for cleanup.

## Links

- [`0055-unify-modal-checkpoints-on-hf-model-repository.md`](0055-unify-modal-checkpoints-on-hf-model-repository.md)
- [`../runbooks/100m_10b_deep_decay_modal.md`](../runbooks/100m_10b_deep_decay_modal.md)
- [`../../modal/model_repo_checkpoint.py`](../../modal/model_repo_checkpoint.py)
- [`../../trainer/best_model.py`](../../trainer/best_model.py)
