---
status: accepted
date: 2026-09-01
supersedes: null
---

# 0135 — Wire disposable Kaggle Probe A LR-reset branches

## Context and problem statement

The 100M/10B deep-decay continuation from step 15,500 shows no strong validation overfit signal, while late optimizer updates may be becoming small enough that the run could be LR-limited rather than data/model-limited. We need an empirical probe that compares LR-reset branches against the already-running control without spending effort on durable model publication.

The control branch remains `100m-10b-deep-decay-from-step15500`. Probe A should fork the newest verified control checkpoint, preserve model weights, optimizer moments, scaler state, RNG state, and data cursor, then test two constant-LR resets in separate W&B runs.

## Considered options

- Run a full new high-LR 100M/10B training attempt from step 15,500.
- Add Probe A branches to the existing HF-published continuation path.
- Add a separate Kaggle-only disposable probe launcher with W&B logging and no HF publication.

## Decision outcome

Chosen option: **separate Kaggle-only disposable probe launcher**, because the goal is loss-curve evidence rather than model retention. The probe creates two W&B-visible branches from the newest verified control checkpoint:

- `reset-low`: constant LR `1e-4`.
- `reset-mid`: constant LR `3e-4`.

Each branch uses the Kaggle dual-T4 exact 64-sequence optimizer block, logs training and validation to its own W&B run, and sets `--remote-publish-every-steps 0`. The launcher forbids remote checkpoint and best-model publication flags so probe models do not end up in Hugging Face.

The launcher must also isolate W&B identity per branch. Sequential Kaggle subprocesses can inherit notebook/global W&B environment such as `WANDB_RUN_ID`, `WANDB_ID`, `WANDB_NAME`, or `WANDB_RESUME`; Probe A therefore wraps each trainer subprocess with a branch-specific W&B environment and still passes the same branch-specific identity through CLI args.

Probe A must not call the deep-decay entrypoint's HF-runtime reexec helper directly. That helper restarts into `kaggle/deep_decay_10b_from_15500.py`, which would bypass Probe A and enter the normal HF-published deep-decay trainer. The public `kaggle/probe_a_lr_reset_10b.py` entrypoint must first restart into itself with private `huggingface_hub==1.5.0`, then delegate to `kaggle/probe_a_lr_reset_10b_impl.py` with the imported deep-decay restart shim disabled.

## Consequences

### Positive

- The control, reset-low, and reset-mid loss curves can be compared directly in W&B.
- The probe tests LR-limited training without committing to a risky `1e-3` long phase.
- HF remains a read source for checkpoint/dataset hydration only; model publication is disabled.
- The probe is disposable and can be rerun from whatever verified control checkpoint is current at launch time.
- Reset-low and reset-mid are protected from W&B identity leakage and should appear as separate W&B runs.
- The public Probe A entrypoint survives Kaggle's old HF Hub client by re-executing back into itself rather than into the normal deep-decay trainer.

### Negative or limiting

- The trainer still writes a final local checkpoint by its generic end-of-run behavior, but this checkpoint is local scratch only.
- The probe does not retain remote recoverability; interrupted Kaggle work may be lost.
- The branch source is the newest verified control checkpoint available at launch time, not necessarily a historical best checkpoint if that checkpoint has already been pruned.
- Rerunning the same branch from the same source step intentionally resumes that branch's own W&B run ID.

## Validation

Run on Kaggle:

```bash
python kaggle/probe_a_lr_reset_10b.py --dry-run
python kaggle/probe_a_lr_reset_10b.py
```

Expected W&B run IDs follow:

- `100m-10b-probe-a-reset-low-from-step<SOURCE_STEP>`
- `100m-10b-probe-a-reset-mid-from-step<SOURCE_STEP>`

The trainer command must include `--remote-publish-every-steps 0` and must not include `--remote-drive-manifest`, `--remote-checkpoint-bucket`, `--remote-checkpoint-repo`, or `--best-model-repo`.

The trainer subprocess must clear inherited W&B identity environment and then set a branch-specific `WANDB_RUN_ID`, `WANDB_ID`, `WANDB_NAME`, `WANDB_RESUME=must`, and `WANDB_RUN_GROUP=probe-a-lr-reset`.

The public entrypoint must re-exec `str(Path(__file__).resolve())` for the private HF Hub runtime, not `deep_decay_10b_from_15500.py`, and then import `probe_a_lr_reset_10b_impl`.

## Links

- `kaggle/probe_a_lr_reset_10b.py`
- `kaggle/probe_a_lr_reset_10b_impl.py`
- `tests/test_kaggle_probe_a_lr_reset.py`
