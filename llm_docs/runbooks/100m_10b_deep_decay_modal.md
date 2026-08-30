# 100M / 10B deep-decay on one Modal H100

ADR 0114 moves the frozen ADR-0095 continuation to one exact Modal H100. The
run keeps its existing W&B/Hugging Face ID and rolling 10B data stream. Modal
changes only execution slicing for an existing continuation checkpoint:

```text
global optimizer block:  64 sequences
Kaggle checkpoint slice: microbatch 2
Modal H100 slice:        microbatch 16
Modal accumulation:      4 ordered slices/update
```

## Prerequisites

Use a clean committed checkout with the authenticated Modal profile. The
`small-llm-training` secret must contain `HF_TOKEN`, `WANDB_API_KEY`, and
`SMALL_LLM_HF_REPO_ID`. The base repository identity should resolve to
`roccoangelella/small-llm-100m-qualification` for the current production run.
Unless explicitly overridden, rolling `latest` checkpoints use the private Bucket
`roccoangelella/small-llm-100m-qualification-checkpoints`, while strict validation-loss
`best` uses the dedicated model repository
`roccoangelella/small-llm-100m-qualification-best-100m-10b-deep-decay-from-step15500`.
The best repository is replaced only after its ownership marker is verified, and each
strict improvement deletes/recreates it before publishing the new best checkpoint.

## Inspect without allocating an H100

```bash
modal run modal/launch.py \
  --action deep-decay \
  --model 100M \
  --tokens 10B \
  --dry-run
```

The payload must show `modal_single_h100_block64`, `H100!`, global block 64,
microbatch 16, the unchanged run ID, final step 76,294, and resume policy
`newest_verified_continuation_hf_then_exact_step_00015500_only`.

## Launch or resume

```bash
modal run --detach modal/launch.py \
  --action deep-decay \
  --model 100M \
  --tokens 10B
```

For a bounded infrastructure gate or deliberate session slice:

```bash
modal run modal/launch.py \
  --action deep-decay \
  --model 100M \
  --tokens 10B \
  --max-steps-this-session 4
```

Do not pass `--gpu`, `--microbatch-size`, or `--dataset-dir`. The action pins
the exact H100 request, microbatch 16, and CPU-managed rolling cache path.

## Restore order and fail-closed gate

The CPU function compares the local Modal run Volume, the checkpoint Storage Bucket
`run/100m-10b-deep-decay-from-step15500/latest.json`, and the legacy shared model-repository
pointer while migration is still needed. The newest verified continuation wins. If the
verified local/legacy checkpoint is newer than the Bucket, CPU prepare republishes it to
the Bucket, prunes superseded Bucket objects for this run, and independently reads back
the resulting `latest.json` before any H100 allocation. Only when no continuation exists
may the adapter restore the exact original two-phase step-15,500 source.

Before H100 allocation it verifies the checkpoint manifest, global step,
consumed targets, last-consumed block, full 100M GDN-2 model config, hybrid
optimizer config/state, FP16 scaler, Python/Torch/CUDA RNG state, frozen WSqD
fields, scheduler committed targets, and expected LR. It then stages and
SHA-verifies the dataset window beginning at that exact checkpoint step.

An existing microbatch-2 or historical microbatch-4 continuation is copied to
an atomic staging directory, rewritten to microbatch 16, assigned the matching
configuration hash, strictly reverified, and installed. The original downloaded
tree remains as a hidden same-Volume provider-migration backup. No model,
optimizer, scaler, Python/Torch RNG, counter, or cursor object is changed. A
two-T4 checkpoint has two CUDA generator states; the one-H100 migration keeps
rank zero's CUDA RNG bytes unchanged as the single live device state and keeps
the original two-state tree in the hidden backup. Any other CUDA RNG topology
fails closed.

## Live confirmation

Treat the lane as live only after logs show:

- restoration of the expected Hugging Face checkpoint ID;
- the expected WSqD LR for its committed-target count;
- NVIDIA H100 hardware, microbatch 16, and block 64;
- finite loss, gradient norm, and throughput on successful updates;
- a segment-final or cadence `latest` checkpoint published to the checkpoint Bucket under the unchanged run ID;
- when validation loss strictly improves, the dedicated best-model repository is recreated and points at that verified checkpoint.

Modal functions have a 24-hour timeout. If a full remaining segment cannot
finish inside that window, rerun the same command. The CPU gate realigns the
rolling window before every new H100 allocation; there is no GPU retry shortcut.
