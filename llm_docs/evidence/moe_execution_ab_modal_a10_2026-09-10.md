# Paired A/B of ADRs 0170–0172 on one Modal A10 — 2026-09-10

Measured effect of batched Muon (ADR 0170), fused attention with a chunked output loss (ADR 0171)
and sorted MoE dispatch (ADR 0172) on training execution. Both arms ran **in the same container, on
the same GPU, on the same data, with the same seed**; the only difference is the code tree. This is
the causal control the 2026-09-08 profile could not provide.

## Setup

Modal A10, container `single_use`, pinned torch 2.10.0+cu128 / fla-core 0.5.2 / Python 3.13; M0 MoE
404,110,520 stored / 101,334,200 active, 8 experts Top-1, GDN-2 hybrid `gdn_chunk_size=32`, BF16,
64 × 2,048 per update (131,072 loss-bearing targets), `hybrid_muon_adamw`, normal initialisation,
seed 17, CPU threads 2, first seven blocks of the verified dataset volume
(`sha256 70a68397452588952b66ed6c7c8e0f5044659cbd7d25d7dd116878366ff06296`).

`new` is the tree at `eb264b6`. `old` is that tree with the eight changed files restored to
`d71c3fd` (`model/fused_loss.py` removed), materialised inside the container. Throughput is warm:
six updates after the first, synchronised wall time, cold update excluded.

Vast was attempted first and abandoned: ssh access is unobtainable from the team account
(`docs/investigations/2026-09-10-vast-4090-ab/REPORT.md` in the knowledge hub).

## Result

| arm | runs | warm targets/s | spread | peak allocated |
|---|---:|---:|---:|---:|
| old (`d71c3fd`) | 3 | **8,990.7** | 8,957.6 / 8,971.6 / 9,042.9 | 14.31 GiB |
| new (`eb264b6`) | 2 | **10,750.0** | 10,650.7 / 10,849.2 | 12.43 GiB |

**+19.6 % throughput and −1.89 GiB peak allocated.** Within-arm spread is below 1 %, so the
difference is an order of magnitude larger than the run-to-run noise.

**Independent validation of the measurement chain:** the old arm reproduces the 2026-09-08 Modal A10
figure of 9,036.344 targets/s to within 0.5 %, two days and two container generations apart.

## The memory is the second lever, and it is exclusive to the change

| configuration | warm targets/s | vs old | peak allocated |
|---|---:|---:|---:|
| old, microbatch 2 | 8,990.7 | — | 14.31 GiB |
| new, microbatch 2 | 10,750.0 | +19.6 % | 12.43 GiB |
| **new, microbatch 4** | **12,167.6** | **+35.3 %** | 17.28 GiB |
| old, microbatch 4 | **out of memory** | — | — |
| new, microbatch 8 | **out of memory** | — | — |

`old` at microbatch 4 dies allocating **1.54 GiB**, which is exactly the FP32 logits tensor for that
microbatch (4 × 2,048 × 50,304 × 4 bytes). The chunked loss removes that allocation, so microbatch 4
is available only to the changed code, and it contributes a further +13.2 %. Microbatch 8 exceeds
the A10's 22.06 GiB usable memory in both arms.

## Configuration levers, one variable each against `new` at microbatch 2

| lever | warm targets/s | vs new | verdict |
|---|---:|---:|---|
| TF32 for FP32 matmuls | 11,121.9 | +3.5 % | real but small; it changes Newton–Schulz numerics, so adopting it needs a learning comparison, not a throughput number |
| CPU threads 2 → 8 | 10,911.0 | +1.5 % | inside the between-run spread; no |

## Mechanism: counts per update, from the profiled update of each arm

| operation | old | new | change |
|---|---:|---:|---|
| `cudaLaunchKernel` | 532,955 | 418,796 | −21 % |
| `cudaStreamSynchronize` | 12,770 | 4,398 | −66 % |
| `aten::nonzero` (each a host sync) | 5,120 | **0** | eliminated |
| `aten::index_copy` (out-of-place, whole buffer) | 4,728 | **0** | eliminated |
| `aten::mm` | 80,448 | 60,893 | −24 % |
| `aten::bmm` | 960 | 150 | −84 % |
| `aten::copy_` | 128,656 | 129,984 | +1 % |
| `aten::empty_strided` | 124,182 | 120,793 | −3 % |

Operations that exist only in the new arm, and whose counts confirm each mechanism fires exactly
once where it should: `index_copy_` 640, `argsort` 640, `cumsum` 640 — one per MoE layer per
microbatch (20 × 32) for the sorted dispatch; `_scaled_dot_product_flash_attention` 160 — one per
full-attention layer per microbatch (5 × 32); `_foreach_mul_` 11 and `_foreach_add_` 10 — one pair
per Muon shape bucket, replacing roughly 19,600 per-matrix Newton–Schulz GEMMs.

Do not read speed from the profiled wall time (316.7 s old, 304.1 s new): the profiler dominates it.
Throughput evidence is the warm benchmark above.

## Correctness on GPU kernels

Losses from identical seed and data:

| step | old | new | new (repeat) |
|---:|---:|---:|---:|
| 1 | 10.983213 | 10.983180 | 10.983180 |
| 7 | 9.762262 | 9.762948 | 9.763017 |

The two arms agree to **3.3e-5** at step 1 (3e-6 relative), and the changed code is **bit-identical
across its two runs at step 1**. By step 7 the arms differ by 6.9e-4 while two runs of the same code
differ by 6.9e-5: divergence accumulates from non-deterministic MoE dispatch reductions in both arms,
and the arm-to-arm gap stays one order of magnitude above it. This is mixed-precision kernel
rounding, not a semantic change, and it confirms on GPU what the CPU tests proved exactly.

## Limits

One GPU model, one geometry (the 8-expert Top-1 M0, not the accepted 64-expert design), seven
updates per run, no quality or convergence claim, no 100B-readiness claim. Throughput is
`q_update`; the allocated, device and calendar clocks are untouched. The relative gain is specific
to this host's CPU/GPU balance: a slower host would exaggerate it, a faster GPU would too.
