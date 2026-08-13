# 100M / 10B incremental Modal runbook

This runbook is the operational procedure for the ADR-0058 dataset pipeline and the ADR-0050 fresh 100M / 10B trajectory. It describes a technically launchable pipeline; ADR 0050 still requires the completed 100M / 2B behavioral qualification to authorize the actual fresh 10B H100 run.

## Frozen identities

```text
training run ID: 100m-10b-data-001
dataset profile: modal-10b-b64
dataset run ID: modal-10b-b64-dataset-001
source: nvidia/Nemotron-ClimbMix
source revision: 5eaa64b9c0c85b7f56af01d7dffdb0795816b12b
context length: 2,048
optimizer block: 64 sequences
shard target: 1 GiB
nominal budget: 10,000,000,000 target tokens
whole-block horizon: 76,294 updates
exact target prefix: 10,000,007,168 target tokens
training-validation prefix: 16 deterministic blocks
GPU topology: one Modal H100
```

Standard WSD is frozen before update 1:

```text
warmup:  3,815 updates /   500,039,680 target tokens
stable: 57,220 updates / 7,499,939,840 target tokens
decay:  15,259 updates / 2,000,027,648 target tokens
minimum LR ratio: 0.1
```

## Required secrets

The Modal secret `small-llm-training` must provide the same credentials used by the normal Modal training path, including:

```text
HF_TOKEN
WANDB_API_KEY
```

Configure either `SMALL_LLM_HF_DATASET_BUCKET_ID` explicitly or the normal Hugging Face repository identity from which the launcher derives the dataset bucket. The dataset bucket is durable object storage; it is not mounted as the trainer filesystem.

## Dry run

From a clean checkout on the exact source commit to be launched:

```bash
modal run modal/launch.py --model 100M --tokens 10B --dry-run
```

The printed contract must identify:

```text
dataset_profile = modal-10b-b64
dataset_transport = hf_rolling_shards
incremental_dataset_producer = true
run_id = 100m-10b-data-001
```

A dirty controlling checkout is rejected.

## Launch

After ADR 0050's capability gate authorizes the 10B experiment:

```bash
modal run --detach modal/launch.py --model 100M --tokens 10B
```

Do not prebuild or upload the complete 10B derived corpus. Do not provide `--dataset-dir` for this profile.

The launcher performs this allocation order:

```text
CPU import/runtime preflight
        ↓
spawn cheap CPU dataset producer
        ↓
spawn independent CPU staging gate
        ↓
producer range-reads pinned ClimbMix and publishes verified READY shards
        ↓
stager waits for checkpoint-aligned current shard + successor + frozen validation
        ↓
stager downloads, SHA-256 verifies, and commits the local cache Volume
        ↓
launcher rechecks producer/stager health
        ↓
only now spawn H100 training
```

A producer failure before staging completes cancels the staging call and propagates before H100 allocation. A staging failure cancels the producer call. The rolling H100 function has no automatic Modal retry that could bypass CPU restaging.

## Producer durability invariant

Each producer checkpoint must execute in this order:

```text
finalize immutable shard bytes
→ upload to HF dataset bucket
→ complete remote read-back verification
→ atomically write producer progress
→ commit the Modal producer Volume
→ publish the immutable READY frontier
→ optionally evict producer-local shard bytes
```

READY never names an unverified shard. Mutable `.tmp`/`.part` objects are never trainer-visible.

The producer continues until both the source-token target and the frozen 76,294-block training horizon are satisfied. A small packing tail beyond the training horizon may exist in the final corpus manifest, but the trainer consumes only the frozen prefix.

## H100 rolling cache

The trainer-facing bootstrap manifest is immutable. Mutable progress lives in `shard_frontier.json`.

At steady state the local train cache is bounded around current + next shard:

```text
train shard N:       being consumed
train shard N+1:     asynchronously prefetched and verified
later train shards:  HF only until needed
validation prefix:   retained locally for repeated eval
```

When the last block of shard N is acknowledged, N+1 must already have completed verification before N is evicted. The exact N+1 prefetch future is promoted to current; the boundary does not redownload or rehash the same remote object unnecessarily.

If the producer frontier falls behind the trainer, the trainer waits. It must never skip a block, reorder shards, or substitute a later shard.

## Resume

The CPU staging gate derives the next required block from the newest durable training checkpoint before H100 allocation. It stages the shard containing that block plus a verified successor when one is expected.

The H100 function rechecks that the checkpoint cursor has not advanced since CPU staging. If it changed, the H100 call fails rather than consuming a stale staged window; rerun the launcher so CPU staging realigns first.

Old consumed train shards do not need to remain local. A rollback can redownload the required immutable shard from the HF dataset bucket.

Producer-local state is committed on every READY durability boundary. A same-workspace/container retry resumes the exact source cursor. The immutable HF READY shards remain durable independently of the producer container.

## Session length

The rolling H100 Modal function has a 24-hour execution timeout. If measured 100M/10B throughput cannot finish the remaining plan inside that boundary, run explicit training segments:

```bash
modal run --detach modal/launch.py \
  --model 100M \
  --tokens 10B \
  --max-steps-this-session <N>
```

Choose `N` from measured H100 throughput with comfortable headroom. The next invocation restages from the newest durable checkpoint. Do not rely on an automatic H100 retry.

For a partial training segment, the CPU producer is allowed to keep running as its own spawned Modal call so it can extend the READY frontier while the next H100 session is not allocated.

## Expected terminal artifacts

At dataset completion, HF dataset storage must contain the immutable shard inventory plus:

```text
run_contract.json
shard_frontier.json
manifest.json
ready.json
```

The completed `shard_frontier.json` binds the final manifest SHA-256. The final schema-v2 `manifest.json` is the canonical completed-corpus record for later reuse.

The training trajectory retains the normal Modal/Hugging Face checkpoint and W&B durability contracts. Completion of dataset production does not itself authorize or imply successful model training.

## Failure interpretation

Fail closed on any of the following:

- run-contract hash or run-ID mismatch;
- READY frontier regression or mutation;
- shard byte-size or SHA-256 mismatch;
- missing checkpoint-aligned current shard;
- missing successor before H100 dispatch when a successor is expected;
- unfrozen validation prefix;
- checkpoint cursor advancing after CPU staging;
- producer/stager exception before H100 dispatch;
- trainer attempting to read beyond the frozen 76,294-block horizon.

Do not repair these by skipping data, changing order, altering the manifest identity, or weakening verification. Diagnose the producer/frontier/checkpoint state and relaunch through the CPU gate.
