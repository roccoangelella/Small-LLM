---
status: evidence
date: 2026-08-31
run_id: 100m-10b-deep-decay-from-step15500
checkpoint_id: step-00070750
---

# 100M / 10B Kaggle rank-1 dedicated-best publication abort

## Observation

The Kaggle dual-T4 continuation reached `step-00070750` and successfully published the rolling exact-resume checkpoint to the HF Storage Bucket backend before aborting.

Captured event:

```json
{
  "remote_publication": {
    "best_updated": false,
    "checkpoint_id": "step-00070750",
    "elapsed_seconds": 25.987994205000177,
    "final": false,
    "rolling_cleanup": {
      "bucket_id": "roccoangelella/small-llm-100m-qualification-checkpoints",
      "checkpoint_id": "step-00070750",
      "deleted_files": 5,
      "status": "pruned"
    },
    "validation_loss": 2.827240757760592
  }
}
```

W&B summary reported `train/consumed_tokens=9273344000`, `train/block_id=70749`, `checkpoint/remote_id=step-00070750`, and `checkpoint/remote_final=False`.

## Failure

After remote publication, rank 1 entered dedicated-best model publication and failed local manifest verification:

```text
RuntimeError: checkpoint root is not a directory: /kaggle/working/small-llm/runs/100m-10b-deep-decay-from-step15500/checkpoints/step-00070750
```

Rank 0 then timed out at the next distributed barrier:

```text
RuntimeError: Timed out waiting 3600000ms for recv operation to complete
```

## Root cause

The Kaggle DDP shim intentionally no-ops `TrainingSession.save_checkpoint` on non-primary ranks. The generic trainer loop handled `checkpoint is None` for local checkpoint logging, but still let every rank evaluate dedicated-best publication.

Rank 1 also uses dummy validation metrics (`loss=0.0`) because rank 0 owns validation. Since best-model selection uses higher-is-better `metric=-loss`, rank 1 could incorrectly see `-0.0` as an improvement over the real best metric around `-2.824985`, then attempt to publish a best model from a non-existent local checkpoint path.

## Fix

Patched `trainer/cli.py` on `main` so distributed non-primary ranks do not own external side effects:

- skip remote publication configuration;
- skip dedicated-best metric lookup, resume repair, and publication;
- skip W&B initialization;
- keep local checkpoint no-op handling intact so rank-local training control flow can advance.

Regression coverage was added in `tests/test_trainer_best_checkpoint.py` with a non-primary distributed rank whose checkpoint save returns `None`; the test asserts no remote configuration, best lookup, best publication, or W&B initialization is attempted.

Commits:

- `19ee65efdc150e7a21b4c7620494aea9421f786d` — guard trainer remote side effects to primary rank.
- `885dd1b4ec2c4c36bed84bc70757473aeda9b6eb` — add regression coverage.

## Recovery implication

Because rolling remote publication completed and cleanup pruned successfully, exact-resume should target the HF Storage Bucket latest checkpoint `step-00070750`. The dedicated best-model repository should remain at the previous strict validation-loss best because `step-00070750` has validation loss `2.827240757760592`, worse than the recorded best `2.824985` at `step-00068250`.
