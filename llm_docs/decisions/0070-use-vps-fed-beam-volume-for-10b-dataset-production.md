---
status: accepted
date: 2026-08-13
---

# ADR 0070: Produce 10B data on the VPS and mirror READY shards into Beam

## Context

The first live Beam 100M/10B attempt showed that running the incremental ClimbMix producer as a paid Beam CPU function is uneconomic. The producer had generated only 210 of the planned 76,294 training blocks when approximately $0.06 had already been spent, while the GPU had not started because the two-shard lead buffer was not yet ready.

Beam supports copying local files directly into a distributed Volume with `beam cp`, and the training adapter already mounts `small-llm-cache` at `/cache`. The VPS can therefore perform the source HTTP reads and deterministic dataset construction continuously in a persistent `tmux` session without consuming Beam CPU runtime.

## Decision

For the Beam 100M/10B lane, run incremental dataset production on the VPS with `beam/vps_dataset_producer.py` instead of allocating the Beam CPU producer.

Keep the Hugging Face Storage Bucket as the authoritative durable dataset backend and monotonic READY frontier. At each existing durable producer boundary the VPS first uploads/verifies finalized shards in HF, commits producer progress, then the durability hook copies the same verified local shard to `beam://small-llm-cache/datasets/<run_id>/<logical_name>`. Only after the hook returns may the ordinary incremental builder publish the updated HF READY frontier and evict the local producer copy.

Launch training through `beam/vps_train.py`. That wrapper reuses the canonical Beam import, CPU staging, visibility, checkpoint, microbatch, and GPU-dispatch gates but replaces `_stage_with_incremental_producer` with stage-only behavior, so no paid Beam dataset-builder function is started. It also uses VPS-specific GPU handlers that activate a trainer-process preseed guard. The guard treats the HF frontier as metadata only: READY shard bytes must appear in the mounted Beam Volume, are SHA-verified before use, and are allowed 120 seconds to become visible. If the VPS feed falls behind, training fails closed and can resume later rather than silently downloading the shard from HF while a GPU is allocated.

The existing two-training-shard startup lead, frozen validation set, 1-GiB target shard geometry, deterministic source ordering/stratification, 10B training horizon, and checkpoint identity are unchanged.

Because Beam documents that distributed-Volume writes can take up to roughly 60 seconds to become visible to other containers, operators should not launch the GPU immediately after the second `beam cp` completes. Leave at least one propagation window, or simply start `beam/vps_train.py` after the corresponding durable producer/READY log is visible. The GPU-side 120-second guard remains a final safety bound.

## Consequences

Dataset source reading and construction no longer incur Beam CPU charges. The VPS may keep producing and mirroring future shards while Beam trains, so production and training remain overlapped.

Each finalized shard is uploaded twice from the VPS: once to HF for authoritative durability and once to Beam for low-latency training consumption. This intentionally trades VPS/network bandwidth for avoiding paid Beam producer time and GPU-side dataset downloads.

The original `beam/launch.py` path remains available for prior behavior, but the cost-controlled incremental 10B Beam procedure is now the VPS-fed `beam/vps_dataset_producer.py` plus `beam/vps_train.py` pair.
