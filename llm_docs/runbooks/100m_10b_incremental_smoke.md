---
status: current
last_reviewed: 2026-08-13
---

# 100M / 10B incremental Modal/HF live smoke

Run this once before the production 100M / 10B H100 trajectory. It is an infrastructure qualification for ADR 0060; it does not replace the ADR-0050 behavioral/capability gate.

## Prerequisites

Use the same Modal account/workspace and `small-llm-training` secret expected by `modal/launch.py`. The secret must provide `HF_TOKEN` and `SMALL_LLM_HF_REPO_ID`. `SMALL_LLM_HF_DATASET_BUCKET_ID` is optional; when absent the normal `<SMALL_LLM_HF_REPO_ID>-datasets` dataset bucket convention is used.

Run from a clean checkout of the commit you intend to qualify. The smoke records that source commit in its checkpoint transport.

## Dry run

```bash
modal run modal/incremental_smoke.py --dry-run
```

This prints the bounded smoke geometry and unique run identities without creating remote functions or allocating an H100.

## Live smoke

```bash
modal run modal/incremental_smoke.py
```

Do not add `--detach` for the qualification run. The local entrypoint supervises the producer and CPU staging calls and should remain attached so a failure is visible immediately.

## What the live smoke does

1. Creates unique `smoke-incremental-*` run IDs and a dedicated private smoke checkpoint model repository.
2. Starts a 4-CPU incremental ClimbMix producer using the approved mixture, context 2,048, block 64, and approximately-4-MiB train shards.
3. Uses a single-use producer-local 10% validation split only so one real validation block is available quickly. Production remains at the frozen 0.1% split.
4. Starts the CPU staging gate concurrently and refuses H100 dispatch until current + successor train shards and frozen validation bytes are remotely durable and locally SHA-verified.
5. Requires the producer still to be active when the gate opens, proving the trainer is consuming a live READY prefix rather than a completed corpus.
6. Runs a 20M model on one H100 for 16 successful optimizer updates. This crosses the first 16-block smoke shard boundary and verifies that the consumed shard is evicted only after its successor is local.
7. Publishes the segment-final checkpoint through the production HF model-repository two-phase transport with W&B disabled.
8. Moves the canonical local run directory to a unique backup path, commits the Modal Volume, and leaves the HF checkpoint pointer untouched.
9. Runs the CPU stage again and requires next block 16 to be derived from the HF pointer.
10. Runs a second H100 segment for four updates and requires an actual `hf_model_repo` restore before reaching step 20.
11. Cancels the still-running CPU producer when the smoke exits.

The intentionally large producer source target exists only to keep the producer active while both H100 segments run. The smoke cancels that producer after success or failure; it does not try to build the declared source target.

## PASS criteria

Treat the smoke as passed only when the final JSON contains `"status": "passed"` and the evidence also shows:

- first segment checkpoint at step 16;
- `consumed_shard_evicted: true` and `successor_present: true`;
- the local checkpoint directory was moved aside while the remote pointer remained valid;
- second CPU stage starts at block 16;
- second segment reports remote restore source `hf_model_repo`;
- final checkpoint is step 20.

The H100 smoke function has no automatic Modal retry. Any H100 failure is a failed qualification and should be investigated from the preserved smoke evidence rather than silently retried.

## Artifact policy

The smoke is deliberately non-destructive. Dataset objects remain under the unique `run/smoke-incremental-dataset-<nonce>/...` prefix, the temporary checkpoint model repository remains under its unique smoke name, and the moved local checkpoint remains on the Modal run Volume. This keeps the exact integration evidence inspectable after either PASS or failure and ensures cleanup cannot affect production identities.

After a passing smoke has been reviewed, these isolated smoke artifacts may be removed separately. Never reuse a smoke dataset, checkpoint repository, or local backup as production training state.

## Production launch boundary

A passing live smoke closes only the ADR-0060 infrastructure gate. The full command remains:

```bash
modal run --detach modal/launch.py --model 100M --tokens 10B
```

Do not execute that production command until the separate ADR-0050 behavioral/capability gate has also been explicitly accepted.
