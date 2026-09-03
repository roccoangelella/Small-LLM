# 100M / 10B Beam runbook

_Last reviewed: 2026-08-30_

This is the active operator procedure for the ADR-0071 full fresh trajectory.
HF remains the authoritative dataset and checkpoint backend; the Beam cache
Volume supplies preseeded dataset bytes to avoid paid GPU downloads.

## Frozen identity

```text
training run ID: 100m-10b-data-001
dataset run ID: modal-10b-b64-dataset-001
GPU: one Beam serverless RTX5090
updates: 76,294
target tokens: 10,000,007,168
WSD: 3,815 warmup / 57,220 stable / 15,259 decay
microbatch: 4 (live-qualified; 8 exceeded RTX5090 VRAM)
precision: fp16 autocast with FP32 master parameters
```

The complete corpus is already verified in HF and mirrored to
`beam://small-llm-cache/datasets/modal-10b-b64-dataset-001`. Do not restart
`beam/vps_dataset_producer.py` and do not use the paid producer path in
`beam/launch.py`.

## Preflight

The checkout must be clean and the locked environment installed:

```bash
uv sync --locked
uv run python beam/vps_train.py \
  --model 100M \
  --tokens 10B \
  --gpu RTX5090 \
  --dry-run
```

The payload must name `100m-10b-data-001`, `hf_rolling_shards`, RTX5090,
auto-microbatch candidates `8,12,16`, and `remaining plan`. Beam secrets
`WANDB_API_KEY`, `HF_TOKEN`, and `SMALL_LLM_HF_REPO_ID` must already exist.

## Full launch

Run from the repository root in a persistent tmux session:

```bash
uv run python beam/vps_train.py \
  --model 100M \
  --tokens 10B \
  --gpu RTX5090 \
  --microbatch-size 4
```

Do not pass `--max-steps-this-session`. The wrapper uses only the external VPS
dataset feed, then performs CPU import, checkpoint-aligned staging, and
fresh-container visibility gates before allocating the GPU. The requested
microbatch 4 still runs the four-update finite-loss, gradient, memory, and
throughput qualification before continuing directly into production. Do not
return to auto 8/12/16 on this device: live microbatch 8 exceeded VRAM.

## Concurrent approximately-5B Kaggle evaluation

Training does not stop for this evaluation. Remote live checkpoints publish
every 500 successful updates with rolling latest-only retention. Capture:

```text
checkpoint: step-00038000
consumed targets: 4,980,736,000
```

Start the Kaggle copy/evaluation as soon as the HF latest pointer names that
checkpoint, before `step-00038500` replaces it. Record the exact checkpoint,
source commit, dataset identity, and decoding/eval manifests in the resulting
evidence. This is an intermediate stable-phase checkpoint, not a terminal 5B
WSD model, and its result does not pause or terminate the Beam trajectory.

## Resume

Rerun the same uncapped microbatch-4 command from active source commit
`1f9dff920ecc45ce2fdb43fd875514a18391273d` or the exact source commit recorded by a later checkpoint. Commit
`42b0376` is accepted only as the one-time infrastructure-migration parent for
the verified step-250 checkpoint. CPU staging realigns to the next unconsumed
block before a new GPU allocation. Never change the microbatch, precision,
dataset identity, or run ID on resume.

New Beam segments keep rolling `latest` in the private HF checkpoint Bucket,
derived as `<SMALL_LLM_HF_REPO_ID>-checkpoints` unless
`SMALL_LLM_HF_CHECKPOINT_BUCKET_ID` overrides it. The provider adapter adds the
dedicated recreate-on-improvement best-model repository automatically. It reads
Bucket latest first; a newer legacy model-repository pointer is CPU-verified,
published, and read back in the Bucket before any GPU allocation.

Beam sets `SMALL_LLM_CHECKPOINT_FSYNC=0`. Checkpoint files and manifests still
use staging plus atomic rename and are hash-verified on restore, but the runtime
does not issue local-disk power-loss barriers against Beam's distributed
Volume. Do not remove this provider-specific setting: the first step-250 save
completed its final rename and then blocked indefinitely on the parent
directory `fsync`.

## Aggressive WSqD continuation from step 15,500

The separate continuation uses:

```bash
python beam/aggressive_wsqd_10b_from_15500.py --gpu RTX4090
```

It starts only from the exact uncooled parent `step-00015500`, but every new
GPU-worker start resolves the highest locally manifest-verified checkpoint in
`100m-10b-aggressive-wsqd-from-step15500/checkpoints` before invoking the
trainer. This is required because Beam may retry a lost worker with the
original function arguments. Confirm the CPU preflight's
`resume_checkpoint_id` and the GPU `aggressive_wsqd_10b_gpu_resume_resolved`
event name the same newest valid checkpoint; never accept a retry that silently
returns to step 15,500 after later continuation checkpoints exist.

The crash/billing supervisor uses the same RTX4090 command. It records a GPU
rate handoff before any lane change so the `$30` cap includes time already spent
on a prior allocation; do not reset the billing baseline just to change GPU.

The automated guard is configured in the user's UTC crontab every five minutes:

```cron
CRON_TZ=UTC
*/5 * * * * cd "$HOME/Projects/Small-LLM" && set -a && . ./.env && set +a && SMALL_LLM_BEAM_BILLING_MODE=account_zero SMALL_LLM_BEAM_CAP_BASIS=notional "$HOME/Projects/Small-LLM/.venv/bin/python" ops/monitor_aggressive_wsqd_10b_beam.py >> /tmp/small-llm-aggressive-monitor/hourly.log 2>&1
```

The monitor applies a 10% safety factor to the notional estimate and stops
matching active Beam tasks when that estimate reaches the `$30` budget. A
control-plane error blocks relaunch and is written to the same log.

This host does not permit a cron daemon to create its PID file, so the active
enforcement loop runs in the persistent tmux session
`small-llm-billing-guard` with the same five-minute cadence. The crontab entry
remains useful if the process is moved to a normal host with cron enabled.
