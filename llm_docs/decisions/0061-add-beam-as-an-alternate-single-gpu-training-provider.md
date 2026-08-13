---
status: accepted
date: 2026-08-13
supersedes: null
---

# 0061 — Add Beam as an alternate single-GPU training provider

## Context and problem statement

The project now has monthly Beam GPU credit and should be able to run the same frozen Small-LLM pretraining contract outside Modal. Beam exposes one-off remote Python functions through the `beam-client` SDK, supports persistent volumes and injected secrets, and currently offers RTX 4090 (24 GiB), A10G (24 GiB), and H100 (80 GiB) GPUs. The existing training system already uses Hugging Face as the cross-workspace model/checkpoint durability boundary and, for the incremental 10B path, as the immutable dataset-shard durability boundary.

The provider migration must not silently change the model, optimizer, data order, schedule, checkpoint identity, or evaluation contract. It also must preserve the existing rule that dataset production/staging happens on cheap CPU before an expensive training GPU is allocated.

## Considered options

- Keep Modal as the only remote GPU provider.
- Add Beam as a separate provider-specific launcher while preserving the same scientific/runtime contract and shared Hugging Face durability.
- Move all cloud execution immediately to Beam and retire Modal.

## Decision outcome

Chosen option: **add Beam as an alternate provider-specific training lane, with a top-level `beam/` adapter parallel to `modal/`, while keeping Modal available and keeping scientific state provider-neutral**.

Beam GPU selection is an execution parameter, not a scientific change. RTX 4090 should be explicitly benchmarked/qualified because its 24 GiB memory may require a smaller execution microbatch than H100; the optimizer block remains the frozen 64 sequences and the existing real forward/backward microbatch qualification selects the fastest safe candidate. H100 remains available when memory or throughput makes it the better cost-per-run choice.

For the incremental 10B lane, Beam must preserve the producer/consumer ordering from ADR 0053/0058: CPU producer/stager establishes the verified current+successor shard lead before GPU dispatch. Beam volumes are cache/local-resume accelerators; Hugging Face remains the durable cross-provider checkpoint/model and dataset-shard boundary.

This decision adds infrastructure capability only. It does not close the ADR 0050/0060 scientific launch gates for the fresh 100M/10B run.

## Consequences

### Positive

- The same frozen training trajectory can consume Beam monthly credit without forking the scientific contract.
- RTX 4090 can be measured against H100 on actual Small-LLM kernels before committing substantial training spend.
- Exact resume remains portable between providers through the unified Hugging Face checkpoint/model layout.
- Dataset production can remain on CPU and continue concurrently with training instead of paying GPU rates for downloads.

### Negative or limiting

- Provider-specific launch/image/volume behavior needs its own tests and runbook.
- Beam distributed-volume writes can take time to become visible to other containers, so CPU producer/stager coordination must rely on the durable Hugging Face READY frontier and explicit local verification rather than assuming instant cross-container volume coherence.
- A 24 GiB RTX 4090 may pass only smaller microbatches and can have different Triton compile/cache behavior from Hopper GPUs.

## Validation

- Unit tests cover Beam profile/GPU resolution, packaging, CPU-before-GPU dispatch, and provider-neutral checkpoint restore semantics.
- A Beam dry run resolves the same model/token/dataset/run IDs as Modal.
- A short live Beam smoke/segmented training run restores/publishes a verified checkpoint through the shared Hugging Face model repository and logs W&B under the canonical run ID.
- RTX 4090 and H100 are compared using measured safe microbatch, tokens/second, peak reserved memory, and estimated dollars per billion target tokens before selecting a provider/GPU for a long run.

## Links

- [`0053-stream-10b-through-one-gib-hf-shards-and-cpu-stage-before-h100.md`](0053-stream-10b-through-one-gib-hf-shards-and-cpu-stage-before-h100.md)
- [`0055-unify-modal-checkpoints-on-hf-model-repository.md`](0055-unify-modal-checkpoints-on-hf-model-repository.md)
- [`0058-produce-10b-shards-concurrently-with-modal-training.md`](0058-produce-10b-shards-concurrently-with-modal-training.md)
- [`0060-require-live-modal-hf-smoke-before-100m-10b.md`](0060-require-live-modal-hf-smoke-before-100m-10b.md)
