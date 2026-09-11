---
status: accepted
date: 2026-09-10
supersedes: null
---

# 0172 — Sorted single-sync MoE dispatch

## Context and problem statement

`DroplessTop1MoE.forward` scanned experts one by one with `torch.nonzero(expert_indices == e)` (a host–device synchronization each), gathered that expert's tokens, and merged outputs with an out-of-place `index_copy` that copies the whole [tokens, d_model] buffer every time. The 2026-09-08 RTX 4090 profile counted 5,120 `nonzero` calls per update (32 microbatches × 20 layers × 8 experts) and the reviewed ATen 2.10 semantics put the full-buffer copies at up to ~40 GiB per update. With 64 experts per layer the synchronizations grow eightfold. ADR 0168 (main, 2026-09-10) already requires the many-expert execution path to be benchmarked before any 100B run and notes that a Python loop issuing one small GEMM per expert is not sufficient evidence.

## Considered options

- Keep the per-expert scan.
- Sort tokens by expert once (`argsort(stable=True)`), read the E group boundaries with one host sync (`cumsum().tolist()`), run every expert on a contiguous slice of the sorted inputs, and write all outputs back once with an in-place `index_copy_` over the permutation.
- Padded per-expert batches with one batched GEMM for all experts (removes the expert loop; wastes FLOPs on padding; needs the max count on host). Deferred: it is the next step once the sorted path is profiled on the target GPU, and it composes with this decision.

## Decision outcome

Chosen option: sorted single-sync dispatch. Routing, router telemetry, `expert_counts`, the balancing controller and the dtype boundary of the merged output are unchanged; only the token bookkeeping changed. Because the stable sort keeps ascending token order inside each expert group, every expert receives exactly the same input batch as before, so outputs and gradients are bit-identical on CPU. Experts with no tokens are skipped and still receive no gradient.

## Consequences

- Per MoE layer and microbatch: one `argsort`, one gather, one boundary read, E expert executions, one `cat`, one in-place `index_copy_`; no `nonzero`, no full-buffer copies. Host synchronizations drop from E to 1 per layer.
- The expert loop still issues E small GEMMs; the batched-GEMM variant is the follow-up. GPU wall-clock and memory effects are not measured by this decision.
- Not changed: Top-1 semantics, dropless guarantee, z-loss, telemetry values, checkpoint identity.

## Validation

`tests/test_moe_sorted_dispatch.py`: bitwise equality of outputs, z-loss and all gradients against the previous dispatch (kept as the test oracle) on random routing and on routing collapse to one expert, and a profiler count asserting zero `nonzero`/out-of-place `index_copy`, one `index_copy_` and one `cumsum` per layer. Existing MoE model, controller, pilot protocol, evaluation and CLI observation suites pass (48 tests, 1 CUDA skip).

## Links

- [Batched Muon](0170-batch-muon-newton-schulz-by-matrix-shape.md)
- [Fused attention and chunked loss](0171-fused-attention-and-chunked-output-loss.md)
- [Pilot contract](0168-moe-paired-pilot-controller-and-observation.md)
