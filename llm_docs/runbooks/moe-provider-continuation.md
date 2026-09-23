# Continue one MoE run on Beam or Modal

Use a clean checkout of the same commit on every provider. The production run ID,
HF checkpoint bucket, source dataset and W&B entity/project must stay the same.
Only one provider writes this run at a time. Commands are procedures, not launch authorization.

## Configuration

Provider secrets already used by the launchers: `HF_TOKEN`, `WANDB_API_KEY`, and
`SMALL_LLM_HF_REPO_ID`. `--checkpoint-bucket auto` follows the existing dense
convention: `SMALL_LLM_HF_CHECKPOINT_BUCKET_ID` if present, otherwise
`<SMALL_LLM_HF_REPO_ID>-checkpoints`. Pass an explicit bucket to remove ambiguity
between accounts. `--wandb-entity` (or provider `WANDB_ENTITY`) is required so
changing the GPU account cannot change the telemetry account.

First invocation only: `--resume new`. Every continuation: `--resume latest`
(the default). Missing progress is an error for continuation, even on an empty volume.
Do not delete a surviving old-account checkpoint until its remote upload is verified.

Beam:

```bash
python beam/moe_production_launch.py \
  --run-id RUN_ID --dataset-dir /data/moe-100b-superbpe-prod \
  --steps 762940 --source-commit COMMIT --resume latest \
  --checkpoint-bucket OWNER/CHECKPOINT_BUCKET --wandb-entity WANDB_ENTITY \
  --checkpoint-every-steps 2500 --microbatch-size 16 \
  --dataset-shard-bucket roccoangelella/small-llm-moe-100b-superbpe-dataset \
  --dataset-shard-run-id moe-100b-superbpe-b64-dataset-001
```

Modal (use the `modal` console script from the repository's torch-enabled venv):

```bash
modal run modal/moe_production_launch.py \
  --run-id RUN_ID --dataset-dir /data/moe-100b-superbpe-prod \
  --steps 762940 --source-commit COMMIT --resume latest \
  --checkpoint-bucket OWNER/CHECKPOINT_BUCKET --wandb-entity WANDB_ENTITY \
  --checkpoint-every-steps 2500 --microbatch-size 64 \
  --dataset-shard-bucket roccoangelella/small-llm-moe-100b-superbpe-dataset \
  --dataset-shard-run-id moe-100b-superbpe-b64-dataset-001
```

Both defaults use BF16 and compiled blocks. `--steps` is the absolute run target,
not the number of extra updates. Changing providers does not restart warmup.
Modal retains its existing wall-time drain for the platform timeout; no new
credit monitoring is introduced. Vast has no accepted-MoE production adapter here;
its historical benchmark is not a qualification of this complete migration path.

## What survives the account

- HF `run/RUN_ID/latest.json`: most recent verified full trainer checkpoint.
- HF `run/RUN_ID/best.json`: best validation-loss snapshot; it may be older.
- Checkpoint data: model/router, optimizer, scheduler, scaler, RNG, step/token counters,
  data cursor, validation metrics, and a receipt binding dataset/code/W&B identities.
- W&B: same entity/project/run ID, plots against `trainer/global_step`, checkpoint
  publication events and a resume marker. Replayed updates can have repeated x values.

No dataset shards are copied into the checkpoint bucket. The CPU gate restages
the required window from the original immutable dataset bucket and verifies the
receipt before GPU dispatch. A local checkpoint ahead of HF (interrupted upload)
is published before more training. A completed run repairs a final upload or best-pointer
update on CPU. Missing local publication sidecars are recovered from the verified
remote snapshot. Corrupt/mismatched state fails closed.

A hard kill between checkpoints replays the updates since the last completed HF
upload. At the existing 2,500-update cadence, the qualified warm throughput implies
about 28 minutes on 4090 or 14 on H100 between saves, excluding validation/upload.
W&B history already sent survives, but an unflushed tail is not guaranteed.

## Verification limits

Offline tests exercise the actual CLI and checkpoint pipeline with a real tiny CPU
MoE and fake remote SDKs. They do not establish HF upload latency, provider capacity,
or bitwise cross-GPU training equivalence. A live GPU/account handoff remains the
operational check for the first authorized segment; it is not an architecture sweep.
