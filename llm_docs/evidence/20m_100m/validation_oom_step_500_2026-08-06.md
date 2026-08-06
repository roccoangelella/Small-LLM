---
status: observed
date: 2026-08-06
experiment: 20m_model_on_100m_tokens
---

# Step-500 validation OOM

## Observed run state

The first approximately-20M-parameter GDN-2 hybrid run on the fixed approximately-100M-token dataset completed optimizer update 500 and then failed during scheduled held-out validation on an NVIDIA T4.

Immediately preceding training telemetry was stable:

```text
steps shown: 495-500
training loss: 5.5370-5.7058
tokens per second: approximately 3.9k-4.0k
gradient norm: finite; steps 499-500 clipped at the configured 1.0 threshold
reported training VRAM: approximately 9.1 GiB
FP16 overflow evidence: none in the supplied failure excerpt
```

The failure began after update 500, which was the configured validation boundary.

## CUDA evidence

The allocator reported failed requests of:

```text
3,294,625,792 bytes
6,589,251,584 bytes
```

The terminal exception was:

```text
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 6.14 GiB.
GPU total capacity: 14.56 GiB
free at failure: 3.95 GiB
PyTorch allocated: 6.47 GiB
PyTorch reserved but unallocated: 3.99 GiB
```

The traceback ended in full-vocabulary `torch.nn.functional.cross_entropy` during validation.

## Root cause

Training split each 16-sequence optimizer block into microbatches of four. Validation did not: it forwarded all 16 sequences at context 2,048 and semantic vocabulary 50,257 in one call.

Approximate dominant allocations:

```text
FP16 logits:
16 * 2,048 * 50,257 * 2 bytes ~= 3.07 GiB

FP32-sized cross-entropy work:
16 * 2,048 * 50,257 * 4 bytes ~= 6.14 GiB
```

These estimates match the failed allocator requests. The incident is classified as a deterministic evaluation-memory bug, not model divergence or an optimizer overflow.

## Durability consequence

The trainer loop evaluated before writing the scheduled step-500 local checkpoint and before step-500 verified remote publication. The incident therefore also exposed an operational rollback window at an evaluation boundary.

The user explicitly chose not to make recovery of the local step-250 checkpoint a requirement for the next attempt.

## Implemented correction

Validation now defaults to one sequence per inference microbatch under `torch.inference_mode()`, releases optimizer gradients before evaluation, clears unused CUDA cache before and after evaluation, and deletes per-microbatch logits and loss immediately.

At validation microbatch one, the dominant cross-entropy allocation is approximately 0.38 GiB instead of approximately 6.14 GiB.

The Kaggle entry point also enables `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` as a secondary fragmentation safeguard. Microbatching is the primary correction.

## Code identities

```text
validation fix commit: 644c56bbbcb0de62271ad2f8ae5008371a984986
validation regression test commit: e202526a29240f9c4cb7c1bb959b3575d2a4da4b
first complete hotfix pin target: 667604133f742c8744b01320a665ae358e1e80de
```

## Validation performed

No GitHub Actions workflow was configured for the final commit. A targeted local reconstruction check compiled the corrected evaluator and executed a two-sequence validation batch, confirming that the model was called twice with batch size one, metrics were produced, invalid microbatch zero was rejected, and the original model training mode was restored.
