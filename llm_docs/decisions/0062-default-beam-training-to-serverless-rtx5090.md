---
status: accepted
date: 2026-08-13
supersedes: null
---

# 0062 — Default Beam training to serverless RTX 5090

## Context and problem statement

Beam exposes a wider on-demand catalog, but the project Beam credits are intended to be consumed through Beam's serverless GPU lane. The current account/machine listing shows serverless availability for A10G, RTX4090, and RTX5090. The same table separately lists $0.71/hour for RTX5090 and $0.66/hour for RTX4090 in its **on-demand** column; those numbers are not treated as serverless billing rates. The project should not accidentally dispatch an on-demand-only GPU such as H100 from the Beam training adapter.

The 100M Small-LLM training geometry is small enough that a consumer Blackwell GPU is a plausible execution target, but the exact safe microbatch and cost efficiency must be measured on the real GDN-2/FLA training kernel rather than assumed from nominal TFLOPS or VRAM.

## Considered options

- Default Beam training to serverless RTX4090.
- Default Beam training to serverless RTX5090.
- Keep H100 as the Beam default even though it is not in the serverless lane shown by the current account listing.
- Allow any Beam catalog GPU without distinguishing serverless from on-demand availability.

## Decision outcome

Chosen option: **Beam training uses serverless GPUs only, with RTX5090 as the default and RTX4090/A10G retained only as explicit serverless alternatives.**

The Beam adapter must fail closed on GPU names outside that serverless allow-list instead of silently falling back to an on-demand machine. RTX5090 is an execution default, not a scientific change: model geometry, optimizer block size, data order, precision contract, checkpoint identity, W&B run identity, and evaluation settings remain unchanged.

The first RTX5090 launch must still run the existing real forward/backward microbatch qualification. The optimizer block stays at 64 sequences; execution microbatch is selected from the safe candidates by measured throughput and the existing memory ceiling. No microbatch is frozen in advance merely because RTX5090 is expected to be faster than RTX4090.

## Consequences

### Positive

- Beam credits are spent on the intended serverless lane and scale to zero when the function is not running.
- RTX5090 becomes the primary serverless performance candidate; serverless credit burn and dollars per target token are measured from actual Beam runs rather than inferred from the table's on-demand price column.
- Accidental H100/on-demand dispatch from the Beam adapter is prevented.
- The same exact-resume Hugging Face checkpoint/model transport remains portable between Beam and Modal.

### Negative or limiting

- Beam's public documentation may lag the account's live machine catalog, so the exact `RTX5090` SDK alias must be verified by the live CLI/runtime before the first paid training segment.
- Blackwell-specific CUDA/Triton compatibility must be proven in a live smoke run; a successful import or dry run is not enough.
- If RTX5090 availability is temporarily exhausted, the operator must explicitly select another allowed serverless GPU rather than receiving an automatic on-demand fallback.

## Validation

- `beam/` profile and launcher tests assert `RTX5090` is the default and that only the serverless allow-list is accepted.
- Beam dry run reports `RTX5090` and never requests an on-demand GPU.
- A live RTX5090 smoke run verifies CUDA/FLA/Triton execution, safe microbatch qualification, checkpoint publication/restore, and W&B continuity.
- Compare measured tokens/second, peak reserved VRAM, and dollars per billion target tokens against RTX4090 before any long-run cost conclusion.

## Links

- [`0061-add-beam-as-an-alternate-single-gpu-training-provider.md`](0061-add-beam-as-an-alternate-single-gpu-training-provider.md)
- [`0055-unify-modal-checkpoints-on-hf-model-repository.md`](0055-unify-modal-checkpoints-on-hf-model-repository.md)
- [`0058-produce-10b-shards-concurrently-with-modal-training.md`](0058-produce-10b-shards-concurrently-with-modal-training.md)
