---
status: accepted
date: 2026-08-13
supersedes: null
---

# 0060 — Require a live Modal/Hugging Face smoke before the 100M / 10B launch

## Context and problem statement

ADR 0058 changed the 100M / 10B dataset path from a completed-corpus handoff to a live producer/consumer pipeline. Network-free regressions cover the incremental run contract, READY frontier, CPU staging, rolling cache, durability ordering, and Modal CPU supervision, but they cannot prove the real service boundary: Modal Volume visibility, Hugging Face Storage Bucket read-back, CPU-to-H100 dispatch, model-repository checkpoint publication, or cross-container restore.

The production 10B launch is expensive enough that these integration boundaries should be exercised once with real services before the full H100 trajectory is authorized. The smoke must not reuse production run IDs or mutate production checkpoint history, and it must remain distinct from the ADR-0050 behavioral/capability gate.

## Considered options

- Trust the network-free regression suite and make the first 10B launch the integration test.
- Build the complete 10B corpus before testing the live path.
- Run a small opt-in live Modal/Hugging Face smoke that exercises the same incremental transport and checkpoint protocols under isolated identities.

## Decision outcome

Chosen option: **require one passing opt-in live Modal/Hugging Face incremental smoke before the production 100M / 10B H100 launch**, because it validates the external durability/orchestration boundaries without spending a material fraction of the production training budget.

The smoke contract is:

- use the real pinned ClimbMix range reader and approved production mixture weights;
- preserve production context length 2,048 and optimizer block geometry of 64 sequences;
- use approximately 4 MiB smoke train shards containing 16 full block-64 optimizer units, so a short H100 segment crosses a real shard boundary and can prove consumed-shard eviction plus successor availability;
- keep a frozen 64-block trainer horizon with the ordinary WSD implementation, while exercising only the first 20 updates;
- use a 20M model for the transport smoke because the completed 100M / 2B run already qualifies the production model/runtime on H100; this smoke is testing data/checkpoint orchestration, not 100M model capacity or throughput;
- keep W&B disabled for the smoke;
- run the CPU producer and CPU staging gate concurrently and refuse H100 dispatch if the producer has already completed, so the test observes a genuinely live incremental frontier;
- allocate the H100 only after current + successor train shards and a frozen validation shard are remotely durable, downloaded, and re-hashed on CPU;
- train the first H100 segment for 16 successful updates, crossing exactly the first smoke shard boundary;
- publish the final segment checkpoint through the same HF model-repository two-phase transport used by production;
- move the local smoke run directory to a unique backup path after segment one, preserving its bytes but removing it from the canonical resume location;
- re-run the CPU stage and require its next-block cursor to come from the HF model-repository pointer;
- run a second H100 segment for four updates and require an actual `hf_model_repo` checkpoint restore before training resumes;
- use a unique `smoke-incremental-dataset-<nonce>` dataset namespace and a dedicated private `<model-repo>-incremental-smoke-<nonce>` checkpoint repository so no production run pointer or history is touched.

The production validation split remains frozen at 0.1%. To make a real validation block available quickly, only the single-use smoke producer container raises its local split probability to 10% **before** importing the streaming producer. The smoke records this override as non-scientific metadata. It does not create a reusable scientific dataset and does not alter production configuration or the canonical launcher.

Live smoke artifacts are preserved under their isolated IDs after the run so a failed or passing result can be audited. The CPU producer is cancelled when the smoke exits so it cannot continue consuming source/network resources after the test.

## Consequences

### Positive

- A passing smoke proves real HF dataset upload/read-back, CPU lead staging, H100 rolling consumption, shard-boundary eviction, HF model-repository checkpoint publication, and remote-only resume before the expensive 10B trajectory.
- Production dataset and checkpoint identities remain untouched.
- The GPU work is bounded to two short H100 segments and 20 optimizer updates on the 20M model.
- A failed smoke leaves isolated evidence rather than corrupting or advancing a production run.

### Negative or limiting

- The smoke consumes a small amount of real Modal H100 time and Hugging Face storage.
- The 10% validation split is intentionally test-only, so the smoke does not qualify production dataset semantics or statistical mixture quality.
- Passing this smoke is an infrastructure gate only. It does not satisfy ADR 0050's requirement for explicit behavioral/capability evidence before launching 100M / 10B.

## Validation

The smoke passes only when all of the following are observed in one invocation:

1. the CPU producer is still active when the first CPU staging gate returns READY;
2. CPU staging verifies current + successor train shards and frozen validation bytes before H100 dispatch;
3. the first H100 segment reaches step 16 and the first train shard is absent locally while its successor is present;
4. the HF model-repository `latest` pointer names the step-16 checkpoint;
5. after the canonical local run directory is moved aside, the second CPU stage resolves next block 16 from the remote pointer;
6. the second H100 segment explicitly reports restore source `hf_model_repo` and reaches step 20;
7. no automatic Modal retry is configured on the H100 smoke function.

## Links

- [`../../modal/incremental_smoke.py`](../../modal/incremental_smoke.py)
- [`../../modal/incremental_smoke_support.py`](../../modal/incremental_smoke_support.py)
- [`../../tests/test_modal_incremental_smoke.py`](../../tests/test_modal_incremental_smoke.py)
- [`../runbooks/100m_10b_incremental_smoke.md`](../runbooks/100m_10b_incremental_smoke.md)
- [`0058-produce-10b-shards-concurrently-with-modal-training.md`](0058-produce-10b-shards-concurrently-with-modal-training.md)
- [`0050-scale-100m-to-fresh-10b-with-5b-capability-gate.md`](0050-scale-100m-to-fresh-10b-with-5b-capability-gate.md)
