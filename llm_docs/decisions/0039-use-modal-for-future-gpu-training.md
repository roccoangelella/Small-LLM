---
status: accepted
date: 2026-08-11
supersedes: null
---

# 0039 — Use Modal for future GPU training

## Context and problem statement

The active 20M / 2B Kaggle trajectory demonstrated that the qualified T4 path is too slow for the next scaling stage. The project needs a faster single-GPU execution platform without changing the finite-data, optimizer-update, checkpoint, WSD, or GDN-2 semantics that make existing scaling runs comparable.

The repository already has a frozen approximately-100M `substantive` geometry and a reusable schema-v2 2B corpus. The current GDN-2 production backend is `fla-core==0.5.2` with a Triton CUDA kernel, but accepted measured qualification is T4/SM75 only. A new GPU therefore requires a bounded hardware qualification rather than assuming that higher peak FLOPs imply correctness.

## Considered options

- Keep future training on Kaggle T4 and accept long wall-clock runs.
- Move new training to Modal but duplicate the trainer/scientific configuration in a second implementation.
- Move new training to Modal while reusing the existing trainer and finite-dataset contracts, with only the provider adapter being Modal-specific.
- Start immediately on Blackwell B200/B300 hardware despite the older pinned FLA/Triton stack.

## Decision outcome

Chosen option: **use Modal as the canonical GPU platform for new pretraining runs, with one profile-driven `modal/launch.py` adapter that reuses the existing trainer and finite-dataset contracts.** Kaggle remains the historical/reproduction lane for experiments already launched there; it is not deleted by this decision.

The initial Modal default is one H100 request. Modal may transparently upgrade that request to H200 at the H100 price. Hopper is preferred for the first migration because it is much faster than T4 while having a more mature kernel/software ecosystem than Blackwell for the project's pinned FLA/Triton stack.

The first Modal migration keeps FP16, saved `gdn_chunk_size=32`, FLA internal chunk 64, the 16-sequence prepared block, hybrid Muon+AdamW, manifest-derived WSD, seed 17, and 250-update durability/validation cadence. Execution microbatch is qualified on the rented GPU from candidates 4, 8, and 16 and then frozen into the run. Since the optimizer block contains 16 sequences, 16 is the maximum useful microbatch without changing optimizer geometry.

Modal Volumes become the durable checkpoint transport for Modal runs. This intentionally avoids reusing the legacy Hugging Face checkpoint namespace for a different model on the same dataset because that protocol currently keys checkpoint publication by dataset run ID and would collide across model sizes.

## Consequences

### Positive

- New runs can use substantially faster training GPUs without rewriting model/trainer logic.
- The already-built finite 2B corpus can be uploaded once to a read-only Modal Volume and reused across model sizes.
- Persistent run and Triton-cache Volumes make retries and kernel compilation platform-native.
- Microbatch selection uses measured throughput and memory on the actual GPU while preserving the exact optimizer batch.
- The 100M / 2B candidate is operationally supported by the existing `substantive` geometry without making this ADR itself an authorization to start that scientific run.

### Negative or limiting

- H100/H200 is a new hardware qualification surface; historical T4 evidence is not sufficient by itself.
- The first kernel invocation compiles for the new GPU architecture; a persistent Triton cache reduces but does not eliminate image/environment invalidation costs.
- Modal Functions are interruptible and have a 24-hour per-attempt limit, so exact checkpoint resume remains mandatory.
- BF16 and Blackwell are deliberately deferred until separate numerical/kernel qualification.
- Modal checkpoint durability is currently provider-native rather than mirrored through the legacy Hugging Face checkpoint protocol.

## Validation

Before accepting the first production Modal trajectory, require:

1. successful image build with the pinned PyTorch/FLA stack;
2. exact schema-v2 dataset verification and manifest-derived WSD plan;
3. finite GDN-2 forward/backward training probes on the rented GPU;
4. measured microbatch 4/8/16 throughput and VRAM evidence, with the selected candidate below 90% reserved memory;
5. a 250-update bounded segment with validation and durable checkpoint creation;
6. an intentional rerun proving automatic verified resume at the next immutable block;
7. continuous W&B step/token identity across the resume.

## Links

- [`../runbooks/modal_training_launcher.md`](../runbooks/modal_training_launcher.md)
- [`../reference/training_system.md`](../reference/training_system.md)
- [`../reference/gdn2_fla_backend.md`](../reference/gdn2_fla_backend.md)
