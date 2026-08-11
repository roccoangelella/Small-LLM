---
status: accepted
date: 2026-08-11
---

# 0035 — Accelerate and persist eval_core_v1

## Context

The first full 500M-parent post-SFT qualification attempted to self-provision `eval_core_v1` inside an ephemeral Kaggle GPU session. The legacy builder scanned the pinned Nemotron-ClimbMix tokenized JSONL source serially in 256 MiB regions using 8 MiB HTTP range reads, parsed every source JSON/token array, and only afterward discarded the approximately 99.9% of documents outside the frozen 0.1% validation partition. The full stratified suite also requires every retained cluster, including rare clusters, to reach its document and target-token floors. In practice this made construction of a roughly tens-of-megabytes permanent evaluation artifact take many hours while the GPU was idle.

The user decided that this path must be optimized rather than accepted as normal evaluation overhead.

The first accelerated attempt used 8 concurrent regions with 32 MiB range reads. Production Kaggle evidence showed repeated `IncompleteRead` failures on those large requests and long head-of-line stalls where `partial_output` remained at 0 MiB even while later workers were active. That configuration is therefore rejected as the default.

## Decision

Keep the existing `eval_core_v1` corpus semantics, split identity, cluster quotas, source revision, deterministic 256 MiB work-plan order, binary format, verifier, and manifest schema unchanged, but replace the production self-build path with an exactness-preserving streaming scanner.

The accelerated scanner:

1. Finds JSONL record boundaries directly in raw downloaded chunks and computes the frozen train/validation assignment from permanent source identity before materializing the line. For the approximately 99.9% of non-validation records, it therefore avoids both constructing a `ParsedRecord` containing the full line and JSON/token deserialization.
2. Materializes and structurally validates the complete JSON/token array only for records whose permanent source identity hashes into the frozen validation partition. Invalid validation candidates and excluded clusters are still rejected by the unchanged structural/cluster gates.
3. Keeps the immutable 256 MiB work-plan and consumes results strictly in its original deterministic order. Concurrency therefore cannot change selected documents, per-cluster quota filling, or output ordering.
4. Uses a conservative default of 4 concurrent region workers and 8 MiB HTTP range reads. `SMALL_LLM_EVAL_SCAN_WORKERS` may tune concurrency without changing corpus identity. The smaller request size is chosen from direct Kaggle evidence that 32 MiB requests repeatedly produced partial HTTP reads.
5. Prefetches several work-plan regions per worker so a slow early region does not leave otherwise available workers idle. Prefetched results remain committed only in frozen work-plan order.
6. Reconstructs the legacy `scanned_records` counter at the exact candidate boundary where the original builder would stop, so the generated manifest remains byte-for-byte compatible when the source stream is identical.
7. Reports actual source-scan telemetry (`downloaded_bytes`, records scanned, regions finished, regions committed) in addition to output-file size, because output bytes are not a meaningful progress proxy during sparse validation discovery.

Repository tests cover raw streaming candidate reconstruction across small chunk boundaries, ordered concurrent region consumption, and accelerated-versus-legacy manifest/binary equivalence on the same deterministic synthetic stream.

## Persistence policy

`eval_core_v1` is a permanent immutable evaluation corpus, not a per-run scratch artifact. Once a verified production corpus has been built, preserve it as an immutable attached dataset (normally a private Kaggle dataset for the current workflow) and reuse it for future evaluations.

The evaluator auto-discovers a unique attached `eval_core_v1` manifest under `/kaggle/input`, verifies all hashes, and uses it directly. If no attached corpus exists, the accelerated self-build remains a fallback into the writable local cache.

## Existing pretraining validation artifact

The already-built pretraining `validation.bin` cannot be used to recreate this exact stratified suite by itself. The pretraining writer stores the accepted token stream and aggregate per-cluster counters, but not record-level source identity plus `cluster_id` alongside each validation document. Those record-level labels are required to enforce the frozen equal minimum document/target floors per retained cluster. Changing the suite to a non-stratified sample of `validation.bin` would therefore be a scientific-policy change and is not part of this optimization.

## Consequences

- Future attached-corpus evaluations avoid the ClimbMix source scan entirely.
- A first build remains network-intensive because exact source record boundaries must still be discovered, but unnecessary record materialization/JSON parsing is removed from the 99.9% rejection path and network concurrency is kept below the empirically fragile 8-worker/32-MiB configuration.
- Worker count, fetch size, prefetch depth, and progress reporting are operational only; deterministic consumption order keeps scientific identity independent of those settings.
- The permanent solution remains to build/verify this artifact once and persist it, rather than repeatedly invoking any self-build path.

## Validation

Before treating the accelerated production path as fully qualified:

1. Run the repository eval-core streaming/equivalence/order tests.
2. Run one production Kaggle build and verify that `verify_eval_core` passes.
3. If an earlier legacy-built `eval_core_v1` completes, compare its manifest and file hashes against the accelerated build; they should match exactly.
4. Publish the verified corpus once and confirm a fresh Kaggle evaluation auto-discovers it under `/kaggle/input` and performs verification without rebuilding.

## Links

- `dataset/eval_core.py`
- `dataset/eval_core_accelerated.py`
- `dataset/src/build.py`
- `trainer/eval_entrypoint.py`
- `tests/test_eval_core.py`
- `tests/test_eval_core_accelerated.py`
- `../reference/post_training_sft.md`
- `../runbooks/sft_s0_runbook.md`
