---
status: accepted
date: 2026-09-01
supersedes: null
---

# 0135 — Wire disposable Kaggle Probe A LR-reset branches

## Context and problem statement

The 100M/10B deep-decay continuation from step 15,500 shows no strong validation overfit signal, while late optimizer updates may be becoming small enough that the run could be LR-limited rather than data/model-limited. We need an empirical probe that compares LR-reset branches against the already-running control without spending effort on durable model publication.

The control branch remains `100m-10b-deep-decay-from-step15500`. Probe A originally forked the newest verified control checkpoint, then was pinned to `step-00068250` when that was believed to be the strict-best checkpoint. On Kaggle, `best_model.json` later resolved successfully and reported `step-00071750`, proving the replace-only dedicated best-model repository had advanced and the old `step-00068250` source was no longer the current HF best artifact. Probe A is now pinned to the current available strict-best checkpoint `step-00071750`.

The current best-model snapshot downloaded from Hugging Face contains the loss-bearing checkpoint payload, but Kaggle observed it without `local_manifest.json` at the checkpoint root. Probe A therefore reconstructs a deterministic local manifest from the downloaded checkpoint tree before using the standard `verify_local_manifest()` gate.

## Considered options

- Run a full new high-LR 100M/10B training attempt from step 15,500.
- Add Probe A branches to the existing HF-published continuation path.
- Add a separate Kaggle-only disposable probe launcher with W&B logging and no HF publication.
- Start Probe A from rolling latest.
- Start Probe A from the fixed strict-best checkpoint `step-00068250`.
- Start Probe A from the current available strict-best checkpoint `step-00071750`.
- Require the best-model repository to preserve `local_manifest.json` exactly.
- Reconstruct `local_manifest.json` locally for marker-verified best-model snapshots that contain the required checkpoint files.

## Decision outcome

Chosen option: **separate Kaggle-only disposable probe launcher pinned to `step-00071750`**, because the goal is loss-curve evidence from a comparable checkpoint rather than model retention, and the HF dedicated best-model repo currently points to that checkpoint. The probe creates two W&B-visible branches from the fixed strict-best checkpoint:

- `reset-low`: constant LR `1e-4`.
- `reset-mid`: constant LR `3e-4`.

The fixed source checkpoint is restored from the dedicated best-model repository:

```text
roccoangelella/small-llm-100m-qualification-best-100m-10b-deep-decay-from-step15500
models/100m-10b-deep-decay-from-step15500/step-00071750
```

If that downloaded checkpoint root lacks `local_manifest.json`, the Probe A wrapper rebuilds it by hashing every downloaded checkpoint file except publication metadata (`local_manifest.json`, `drive_manifest.json`, and `checkpoint_manifest.json`). The rebuilt manifest must cover at least `trainer_state.pkl` and `checkpoint.json`, and the normal `verify_local_manifest()` check still runs before the checkpoint is copied into the local control-checkpoint namespace.

The public `kaggle/probe_a_lr_reset_10b.py` entrypoint must force the 100M Hugging Face namespace before restore or dataset staging. Kaggle notebooks can retain environment variables from older 20M runs; Probe A must therefore set `SMALL_LLM_HF_REPO_ID=roccoangelella/small-llm-100m-qualification`, `SMALL_LLM_HF_CHECKPOINT_BUCKET_ID=roccoangelella/small-llm-100m-qualification-checkpoints`, and `SMALL_LLM_HF_DATASET_BUCKET_ID=roccoangelella/small-llm-100m-qualification-datasets` unless dedicated `SMALL_LLM_PROBE_A_*` overrides are provided.

Each branch uses the Kaggle dual-T4 exact 64-sequence optimizer block, logs training and validation to its own W&B run, and sets `--remote-publish-every-steps 0`. The launcher forbids remote checkpoint and best-model publication flags so probe models do not end up in Hugging Face.

The launcher must also isolate W&B identity per branch. Sequential Kaggle subprocesses can inherit notebook/global W&B environment such as `WANDB_RUN_ID`, `WANDB_ID`, `WANDB_NAME`, or `WANDB_RESUME`; Probe A therefore wraps each trainer subprocess with a branch-specific W&B environment and still passes the same branch-specific identity through CLI args. Since the fixed source creates fresh `from-step71750` W&B IDs, the resume policy is `allow`, not `must`.

Probe A must not call the deep-decay entrypoint's HF-runtime reexec helper directly. That helper restarts into `kaggle/deep_decay_10b_from_15500.py`, which would bypass Probe A and enter the normal HF-published deep-decay trainer. The public `kaggle/probe_a_lr_reset_10b.py` entrypoint must first restart into itself with private `huggingface_hub==1.5.0`, then delegate to `kaggle/probe_a_lr_reset_10b_impl.py` with the imported deep-decay restart shim disabled.

## Consequences

### Positive

- The control, reset-low, and reset-mid loss curves can be compared directly in W&B from a fixed source step.
- The probe tests LR-limited training without committing to a risky `1e-3` long phase.
- HF remains a read source for checkpoint/dataset hydration only; model publication is disabled.
- The probe is disposable and can be rerun from the same strict-best checkpoint while that checkpoint remains the dedicated best artifact.
- Reset-low and reset-mid are protected from W&B identity leakage and should appear as separate W&B runs.
- The public Probe A entrypoint survives Kaggle's old HF Hub client by re-executing back into itself rather than into the normal deep-decay trainer.
- Stale `SMALL_LLM_HF_REPO_ID` values from older 20M work cannot redirect Probe A to the wrong HF repo.
- A missing best-model `local_manifest.json` no longer blocks a marker-verified checkpoint restore when the required files are present and re-hashed locally.

### Negative or limiting

- The trainer still writes a final local checkpoint by its generic end-of-run behavior, but this checkpoint is local scratch only.
- The probe does not retain remote recoverability; interrupted Kaggle work may be lost.
- The branch source is no longer the newest verified rolling checkpoint; it is intentionally pinned to `step-00071750`.
- The abandoned `step-00068250` source is not recoverable from the current replace-only dedicated best-model marker; using it would require another preserved copy or a direct `SMALL_LLM_PROBE_A_SOURCE_REPO_ID` override.
- Reconstructing `local_manifest.json` validates the files as downloaded, but it cannot recover a historical manifest that was not present in the best-model snapshot.
- Rerunning the same branch from the same source step intentionally resumes or reuses that branch's own W&B run ID.

## Validation

Run on Kaggle:

```bash
python kaggle/probe_a_lr_reset_10b.py --dry-run
python kaggle/probe_a_lr_reset_10b.py
```

Expected fixed source fields:

```json
{
  "source": "fixed_best_model_checkpoint",
  "source_checkpoint_id": "step-00071750",
  "source_step": 71750,
  "base_hf_repo_id": "roccoangelella/small-llm-100m-qualification"
}
```

If the best-model checkpoint lacks `local_manifest.json`, the launcher should print `probe_a_rebuilt_local_manifest` and then continue through `verify_local_manifest()`.

Expected W&B run IDs:

- `100m-10b-probe-a-reset-low-from-step71750`
- `100m-10b-probe-a-reset-mid-from-step71750`

The trainer command must include `--remote-publish-every-steps 0` and must not include `--remote-drive-manifest`, `--remote-checkpoint-bucket`, `--remote-checkpoint-repo`, or `--best-model-repo`.

The trainer subprocess must clear inherited W&B identity environment and then set a branch-specific `WANDB_RUN_ID`, `WANDB_ID`, `WANDB_NAME`, `WANDB_RESUME=allow`, and `WANDB_RUN_GROUP=probe-a-lr-reset`.

The public entrypoint must re-exec `str(Path(__file__).resolve())` for the private HF Hub runtime, not `deep_decay_10b_from_15500.py`, and then import `probe_a_lr_reset_10b_impl`.

## Links

- `kaggle/probe_a_lr_reset_10b.py`
- `kaggle/probe_a_lr_reset_10b_impl.py`
- `tests/test_kaggle_probe_a_lr_reset.py`
