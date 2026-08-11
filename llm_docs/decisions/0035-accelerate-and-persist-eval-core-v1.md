---
status: accepted
date: 2026-08-11
---

# 0035 — Accelerate and persist eval_core_v1

## Context

The first full 500M-parent post-SFT qualification attempted to self-provision `eval_core_v1` inside an ephemeral Kaggle GPU session. The legacy builder scanned the pinned Nemotron-ClimbMix tokenized JSONL source serially in 256 MiB regions using 8 MiB HTTP range reads, parsed every source JSON/token array, and only afterward discarded the approximately 99.9% of documents outside the frozen 0.1% validation partition. The full stratified suite also requires every retained cluster, including rare clusters, to reach its document and target-token floors. In practice this made construction of a roughly tens-of-megabytes permanent evaluation artifact take many hours while the GPU was idle.

The user decided that this path must be optimized rather than accepted as normal evaluation overhead.

## Decision

Keep the existing `eval_core_v1` corpus semantics, split identity, cluster quotas, source revision, deterministic work-plan order, binary format, verifier, and manifest schema unchanged, but replace the production self-build path with an exactness-preserving accelerated scanner.

The accelerated scanner:

1. Computes the frozen train/validation assignment from permanent source identity before JSON parsing. Only validation candidates are deserialized and token-validated.
2. Scans immutable 256 MiB source regions concurrently. The default is 8 workers and 32 MiB forward HTTP reads; `SMALL_LLM_EVAL_SCAN_WORKERS` may tune concurrency without changing corpus identity.
3. Buffers only validation candidates from each worker and consumes completed regions strictly in the original deterministic work-plan order. Concurrency therefore cannot change selected documents or ordering.
4. Reconstructs the legacy `scanned_records` counter at the exact candidate boundary where the original builder would stop, so the generated manifest remains byte-for-byte compatible when the source stream is identical.
5. Emits source-region progress in addition to the existing build heartbeat.

A repository equivalence test compares the accelerated and legacy builders on the same deterministic synthetic stream and requires identical manifests plus identical `fast.bin`, `full.bin`, and JSONL index bytes. A separate test requires concurrent region completion to be yielded in frozen work-plan order.

## Persistence policy

`eval_core_v1` is a permanent immutable evaluation corpus, not a per-run scratch artifact. Once a verified production corpus has been built, preserve it as an immutable attached dataset (normally a private Kaggle dataset for the current workflow) and reuse it for future evaluations.

The evaluator now auto-discovers a unique attached `eval_core_v1` manifest under `/kaggle/input`, verifies all hashes, and uses it directly. If no attached corpus exists, the accelerated self-build remains a fallback into the writable local cache.

## Consequences

- Future attached-corpus evaluations avoid the ClimbMix source scan entirely.
- A first build remains network-intensive because exact source record boundaries must still be discovered, but serial network I/O and unnecessary JSON/token parsing are removed from the critical path.
- Worker count is operational only; deterministic consumption order keeps scientific identity independent of concurrency.
- The current already-running legacy evaluation process is not modified in place; a new invocation must use a launcher pinned to the accelerated implementation commit.

## Validation

Before treating the accelerated production path as fully qualified:

1. Run the repository eval-core equivalence/order tests.
2. Run one bounded/production Kaggle build and verify that `verify_eval_core` passes.
3. If an earlier legacy-built `eval_core_v1` completes, compare its manifest and file hashes against the accelerated build; they should match exactly.
4. Publish the verified corpus once and confirm a fresh Kaggle evaluation auto-discovers it under `/kaggle/input` and performs verification without rebuilding.

## Links

- `dataset/eval_core.py`
- `dataset/eval_core_accelerated.py`
- `trainer/eval_entrypoint.py`
- `tests/test_eval_core.py`
- `tests/test_eval_core_accelerated.py`
- `../reference/post_training_sft.md`
- `../runbooks/sft_s0_runbook.md`
