# Accepted geometry on one Modal A10: batched expert GEMM against the per-expert loop — 2026-09-10

First execution measurement of the geometry fixed on 2026-09-09 (64 experts, Top-2, no shared
expert, 8 layers, width 256, SwiGLU experts of width 352, 8,000-entry tied vocabulary; ADR 0173),
and of what batching the expert GEMM buys at that expert count.

## Setup

Modal A10, single-use container, pinned torch 2.10.0+cu128 / fla-core 0.5.2 / Python 3.13. Model:
144,025,496 stored / 9,938,840 active parameters, BF16, `hybrid_muon_adamw`, seed 17, normal
initialisation, 64 × 2,048 per update (131,072 loss-bearing targets), CPU threads 2. Both arms use
**the same stacked expert parameters and the same router**; only `dispatch` differs, so the
comparison isolates execution. Token ids are folded into the 8,000 vocabulary rather than
retokenised: this measures execution, not learning. Throughput is warm — updates after the first,
synchronised wall time.

## Result

| dispatch | microbatch | warm targets/s | peak allocated |
|---|---:|---:|---:|
| per-expert loop | 2 | 7,240 | 4.01 GiB |
| **batched GEMM** | 2 | **43,693** | 4.61 GiB |
| per-expert loop | 8 | 25,736 | 5.85 GiB |
| **batched GEMM** | 8 | **58,115** | 8.84 GiB |
| **batched GEMM** | 16 | **60,657** | 15.16 GiB |
| batched GEMM | 32 | out of memory | — |

**At matched microbatch the batched GEMM is +504 % at microbatch 2 and +126 % at microbatch 8.**
Best measured configuration overall: **60,657 targets/s** at microbatch 16, which is **8.4× the
per-expert loop at microbatch 2**. Microbatch 32 exceeds the A10's 22.06 GiB; microbatch 16 fits
inside a 24 GB card with 7 GiB to spare.

## Mechanism, from the profiled update at microbatch 8

| per update | loop | batched GEMM |
|---|---:|---:|
| `cudaLaunchKernel` | 175,522 | **31,411** |
| `aten::mm` | 38,816 | 1,952 |
| `aten::bmm` | 150 | 918 |
| host synchronisations | 609 | 609 |

Launches fall by **82 %**. The loop issues three matrix multiplications per expert per layer per
microbatch — 8 microbatches × 8 layers × 64 experts × 3 ≈ 12k forward, roughly tripled by the
backward pass, which is the 38,816 measured. The batched path replaces them with three `bmm` calls
per layer. Synchronisation count is identical, as designed: both read the expert boundaries once per
layer.

## Numerical agreement between the two dispatches

Step-1 loss: loop 9.019950, batched 9.014205 — a relative difference of 6.4e-4. Under BF16
activations this is below one BF16 unit in the last place at that magnitude (≈ 0.06), and it is the
expected consequence of a different accumulation grouping in the GEMM, not of different semantics:
in FP32 on CPU the two dispatches agree to 1e-5 with identical weights
(`tests/test_moe_accepted_geometry.py`). Each dispatch is self-consistent — the loop returns
9.019950 at both microbatch 2 and 8.

## What it means for the run economics

Holding the measured throughput and quoting each host's own rate:

| configuration | targets/s | USD per 10⁹ | days for 100B | USD for 100B |
|---|---:|---:|---:|---:|
| M0, 101M active, A10, post-ADR-0170-0172, microbatch 4 | 12,168 | 30.22 | 95.1 | 3,022 |
| accepted geometry, A10, per-expert loop, microbatch 2 | 7,240 | 50.79 | 159.9 | 5,079 |
| **accepted geometry, A10, batched GEMM, microbatch 16** | **60,657** | **6.06** | **19.1** | **606** |
| M0, 4090 Vast, pre-change (2026-09-08 reference) | 11,073 | 8.98 | 104.5 | 898 |
| accepted geometry, 4090, batched GEMM — *extrapolated* | 60,657 | 1.64 | 19.1 | 164 |

The last row transports A10 throughput onto the 4090's rate and is an expenditure comparison, not a
benchmark: the 4090 has never run this geometry. Everything above it is measured.

**Without the batched GEMM the accepted geometry would have been slower than the M0 pilot it
replaces** — 7,240 against 12,168 targets/s — because it multiplies the number of expert groups.
That is the result that mattered.

## Limits

One GPU model, one container, five to seven updates per configuration, no repeat runs, so no
confidence interval. No quality, convergence or balance claim: the router is untrained and the ids
are folded, so expert loads here say nothing about a real corpus. Padding waste grows with routing
imbalance and was not stressed beyond the deliberate-collapse CPU test. `q_update` only; the
allocated, device and calendar clocks are untouched.
