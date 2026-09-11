---
status: accepted
date: 2026-09-08
supersedes: null
---

# 0169 — Reuse Triton/FLA cache seeds on the provider lanes

## Context

Rocco already implemented a pinned Kaggle seed in ADR 0102. Historical dense logs show long first updates on Modal and Beam too; these observations are not matched cache experiments and do not establish a MoE speedup. Edo authorized local review/integration of the separate `edo/triton-seed` work, with GPU verification deferred to the next useful authorized tests.

## Decision

Reuse the existing cache volume for archives; compilation uses `/tmp/small-llm-triton/<cache_id>` on ordinary local disk. Keep generated kernels out of Git and the cache out of scientific identity. Kaggle's existing implementation is unchanged.

- `trainer/triton_seed.py` builds a contract without importing torch: GPU name/capability, Python/host ABI, torch/CUDA-wheel/Triton/FLA versions, model family, precision, microbatch/context/chunk and kernel-facing source hashes (including stable dispatch and MoE). Dense and MoE are distinct; M0/M1 share compatible kernel geometry. Triton's own keys still govern individual kernel specializations.
- Restore checks contract, canonical path, archive and file hashes. Local reuse also verifies recorded files. A missing/rejected seed falls back to JIT; successful JIT can replace a rejected seed. Strict mode requires a device and valid seed and cannot be combined with disable.
- Publish an immutable `triton-cache-<sha256>.tar`, then atomically replace its manifest using a unique temporary file. Readers see a complete referenced generation; concurrent writers do not delete archives. Interrupted publication can leave an unreferenced archive, not a half-replaced seed. Local installation is serialized within a container; shared-volume flock is not assumed.
- The pilot records prepare and harvest/commit wall time and status per attempt. Cache restore/package/commit errors are best effort outside strict preparation and cannot turn completed training into failure. A cache error never hides a trainer/checkpoint failure.

## Consequences

The first useful run pays compilation plus harvest; subsequent compatible runs may save startup time. No minutes-to-seconds guarantee is established. `seeded` means verified bytes restored, not all kernels hit or numerical equivalence proven. Unseen shapes still JIT. GPU0 detection is scoped to the single-GPU provider lanes; reuse across other images/drivers remains unqualified even with matching recorded versions. Different kernels/configurations require their own verification.

Do not add dedicated seed-building runs. Harvest the actual upcoming workload, then check reuse in a later necessary fresh container; record net savings including archive I/O and provider commit. Persistent archives can outlive compute and interrupted publication can leave old generations; cleanup is a separate operation after runs stop.

## Validation

Local tests cover round trip, stale/corrupt seed fallback and repair, strict/disabled paths, lost local files, source/geometry separation, interrupted/concurrent publication and cache-commit failure after successful training. The existing pilot/CLI tests remain required. Real CUDA cache hits, cross-container provider durability and net startup savings are explicitly pending.

## Links

- [Kaggle precedent](0102-preseed-kaggle-t4-triton-cache-from-private-dataset.md)
- [Pilot contract](0168-moe-paired-pilot-controller-and-observation.md)
- [Runbook](../runbooks/moe-paired-pilot.md)
