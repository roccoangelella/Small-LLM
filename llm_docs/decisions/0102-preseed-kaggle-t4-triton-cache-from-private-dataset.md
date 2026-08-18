---
status: accepted
date: 2026-08-18
---

# 0102 — Preseed Kaggle T4 Triton cache from a private Dataset

## Context

Kaggle startup for the 100M/10B deep-decay lane is dominated by Triton/FLA kernel compilation and autotuning. That makes infrastructure failures expensive to discover because a notebook can spend many minutes compiling before the first real optimizer update.

The Kaggle deep-decay execution contract is already unusually stable: two Tesla T4 GPUs, Python 3.13, PyTorch 2.10.0 + CUDA 12.8, Triton 3.6.0, `fla-core==0.5.2`, FP16, the frozen approximately-100M GDN-2 hybrid, context 2,048, configured GDN chunk 32, and local microbatch two. This makes a reusable device/runtime-specific cache practical.

## Decision

Use a **portable, manifest-verified Triton cache preseed for the Kaggle 100M/10B dual-T4 lane**.

- Keep generated Triton binaries out of Git. Git stores only cache tooling, contracts, and tests.
- Build the canonical cache once on Kaggle with two visible Tesla T4s under the exact pinned production runtime.
- `kaggle/triton_cache.py` is the dedicated cache lifecycle tool.
- Cache construction instantiates the frozen 100M GDN-2 model with `gdn_chunk_size=32` and runs the exact FP16 `2 x 2048` forward/backward prewarm on one T4. It creates no optimizer, checkpoint, dataset cursor, W&B state, or training update.
- The builder performs a compile/full-autotune process followed by a second fresh pinned-runtime process that reuses the same canonical disk cache before packaging.
- Package the cache as a flat `small_llm_triton_cache_manifest.json` plus `triton-cache.tar`, suitable for a **private Kaggle Dataset**. Optional publication uses the current Kaggle CLI and remains private by default.
- Use one canonical writable cache path under `/kaggle/working/small-llm/runtime-cache/...` so Triton metadata containing absolute child paths is restored at the same location where it was built.
- The manifest pins T4/SM75, Python/Torch/CUDA/Triton/FLA versions, model/precision/shape/chunk geometry, a focused source/kernel contract SHA-256, archive SHA-256, per-file SHA-256 values, and a tree hash.
- `kaggle/dual_t4_train_block64.py` attempts cache installation before importing Torch. The two torchrun ranks serialize installation with a filesystem lock; one rank atomically installs the cache and the other observes the installed manifest.
- A missing, stale, corrupt, wrong-runtime, wrong-path, or wrong-source cache is **not a training correctness failure** by default. The launcher reports the rejection and falls back to ordinary Triton JIT/autotuning. An explicit strict mode exists for cache qualification.
- The existing bounded production autotune fallback remains unchanged. The standalone cache builder does not install that qualification/production cap, so the seed is generated from normal full Triton autotuning.

## Consequences

### Positive

- Subsequent Kaggle sessions can reach the meaningful DDP/prewarm/training gate without repeating the dominant Triton compilation cost for already-covered kernels.
- Cache creation is scientifically inert: it never advances the deep-decay run or mutates model/optimizer/checkpoint state.
- The cache remains replaceable and self-invalidating at the repository-contract level while Triton's own native cache keys continue to protect individual compiled artifacts.
- Kaggle Dataset storage keeps generated binary artifacts out of repository history.

### Limits

- The first cache build still pays the full compile/autotune cost once.
- Future source/runtime/geometry changes can intentionally invalidate the cache and require a rebuild.
- Kernels not exercised by the canonical prewarm can still JIT on first use; the cache is an acceleration layer, not a promise that no compilation will ever occur.
- A private Dataset must be attached to a future Kaggle notebook (or its mount path supplied explicitly) for automatic preseed discovery.

## Operational commands

Build and package on a two-T4 Kaggle session:

```bash
python kaggle/triton_cache.py build
```

Build, package, and create/version a private Kaggle Dataset:

```bash
python kaggle/triton_cache.py build --publish OWNER/DATASET-SLUG
```

After attaching that Dataset, the normal training command is unchanged:

```bash
python kaggle/launch.py deep-decay --model 100M --tokens 10B
```

## Links

- [`0099-run-deep-decay-100m-10b-on-kaggle-dual-t4.md`](0099-run-deep-decay-100m-10b-on-kaggle-dual-t4.md)
- [`../runbooks/100m_10b_deep_decay_kaggle.md`](../runbooks/100m_10b_deep_decay_kaggle.md)
- [`../../kaggle/triton_cache.py`](../../kaggle/triton_cache.py)
- [`../../kaggle/dual_t4_train_block64.py`](../../kaggle/dual_t4_train_block64.py)
