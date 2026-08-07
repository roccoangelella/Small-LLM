---
status: accepted
date: 2026-08-07
supersedes: null
---

# 0011 — Publish a standalone Gated Delta Rule-2 package

## Context and problem statement

The Small-LLM project contains an independently authored PyTorch implementation of the Gated Delta Rule-2 recurrence, a differentiable WY-style chunkwise reference path, correctness/gradient parity tests, cache semantics, and real pretraining evidence. The implementation is useful beyond the Small-LLM model and may support later discussion with PyTorch maintainers.

The official NVIDIA GatedDeltaNet-2 repository is distributed under the NVIDIA Source Code License-NC. A public reusable package from this project must therefore preserve a clean permissive provenance and must not copy, translate, port, vendor, or redistribute NVIDIA source code.

PyTorch Foundation trademark guidance also advises against using the PyTorch name as part of an external product/project name.

## Decision outcome

Create and publish a standalone public project named **`gated-delta-rule`** under the MIT license.

The public package will:

- be derived from the MIT-licensed Small-LLM implementation and the published Gated DeltaNet-2 mathematical specification;
- include no source code from `NVlabs/GatedDeltaNet-2`;
- cite the original Gated DeltaNet-2 paper and authors for the algorithm;
- expose a reusable functional Gated Delta Rule-2 API and a standard `torch.nn.Module` wrapper;
- keep the high-level Gated DeltaNet-2 convenience block explicitly experimental rather than presenting it as an official NVIDIA reproduction;
- use standard PyTorch device/dtype/module conventions where practical;
- retain a tokenwise recurrent oracle and test the chunkwise implementation against it for values and gradients;
- preserve true FP64 reference behavior while using FP32 accumulation for FP16/BF16 inputs;
- include provenance, legal-hygiene, contribution, citation, and non-affiliation documentation;
- not use `PyTorch` as the repository/product name or use PyTorch/NVIDIA logos;
- treat any future fused Triton/CUDA implementation as a separate independently authored optimization validated against the reference path.

## Release gate

Before a public release is treated as authoritative, the standalone tree must pass its local test suite and install smoke test. The repository should record that legal/terms review is engineering due diligence rather than legal advice, and current third-party terms must be re-checked before relicensing, vendoring code, using trademarks/logos, or submitting upstream.

## Publication tooling note

The connected GitHub tool available in this project can write to existing repositories but does not currently expose repository creation or repository-visibility mutation. If that remains true at release time, prepare a complete Git repository/archive and publish it only through an account action that creates a new public repository; do not overwrite or repurpose an unrelated existing public repository.

## Links

- [`../research/pytorch_gdn2_upstream_review_2026-08-07.md`](../research/pytorch_gdn2_upstream_review_2026-08-07.md)
- [`../reference/gdn2_chunkwise_training.md`](../reference/gdn2_chunkwise_training.md)
