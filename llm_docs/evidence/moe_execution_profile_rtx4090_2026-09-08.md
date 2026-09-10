# MoE execution profile on one RTX 4090 — 2026-09-08

Measured observation of where a training update spends its time; the reference point for the
execution-efficiency work recorded in ADRs 0170–0172 and the priority order in
[`../reference/training_execution_efficiency.md`](../reference/training_execution_efficiency.md).
Provenance: Vast.ai single RTX 4090 host, code `b6b48f5` (core algorithms unchanged through
`d71c3fd`), M0 MoE geometry 404,110,520 stored / 101,334,200 active parameters, BF16 autocast,
microbatch 2 sequences × 2,048, 64 sequences per update (131,072 loss-bearing targets), diagnostics
on. Raw `torch.profiler` export and update logs are retained by Edo outside the repository
(`evidence/profile.json` of the probe); the numbers below are copied from that export.

## Throughput (warm, six updates)

| quantity | value |
|---|---:|
| warm throughput (sum of synchronized update seconds, updates 2–7) | 11,073 targets/s |
| wall per update, unprofiled | 11.84 s |
| peak allocated CUDA memory | 14.31 GiB |
| host rate incl. 40 GB disk | 0.35778 USD/h |
| derived: continuous-update cost per 10⁹ targets | 8.98 USD |

## Profiled update 9 (profiler on; wall 19.0 s — proportions are the evidence, not the seconds)

Kernel/API counts per update:

| call | count |
|---|---:|
| `cudaLaunchKernel` | 521,704 |
| `aten::copy_` | 124,752 |
| `aten::empty_strided` | 120,522 |
| `aten::_to_copy` | 103,177 |
| `aten::mul` | 86,159 |
| `aten::mm` | 78,252 |
| `aten::add` (elementwise kernel) | 62,064 |
| `aten::add_` | 51,042 |
| `cudaMemcpyAsync` | 25,063 |
| `cudaStreamSynchronize` | 12,770 |
| `aten::nonzero` (each a host–device sync) | 5,120 = 32 microbatches × 20 MoE layers × 8 experts |
| `ChunkGDN2Function` fwd / bwd | 480 / 480 |

Self device time (GPU busy) per op, seconds: `mm` 1.253, `copy_` 1.160, `mul` 0.524,
`ChunkGDN2FunctionBackward` 0.264, `add_` 0.250, `div` 0.160, `bmm` 0.154, `_softmax_backward_data`
0.140, `masked_fill_` 0.112, `_softmax` 0.097; optimizer step scope 2.003. Sum of ATen op self
device time ≈ 5.0 s of the 19.0 s scope, of which matmul (`mm` + `bmm`) ≈ 1.41 s.

Self CPU time, seconds: unattributed Python inside the update scope 11.17; `cudaLaunchKernel`
3.37 (≈ 6.5 µs per launch); `mm` 1.60; `mul` 0.79; `copy_` 0.67; `empty_strided` 0.61;
`cudaMemcpyAsync` 0.48; GDN2 backward 0.43; `_to_copy` 0.39; optimizer step 0.37; `nonzero` 0.36.

`ampere_sgemm_*` (FP32 GEMM) kernels appear alongside BF16 GEMMs: Newton–Schulz runs in FP32 by
contract, and the pre-ADR-0171 attention computed scores in FP32.

## Derived quantities

- **Model FLOP utilization ≈ 4.1 %**: useful work 6 × 101.3 M × 11,073 tok/s = 6.7 TFLOP/s against
  the 165.2 TFLOP/s dense BF16 tensor-core peak of the RTX 4090 (NVIDIA Ada whitepaper; 330 is the
  sparse figure and must not be used as denominator). Ideal matmul time per update 0.48 s versus
  11.84 s measured.
- **Bytes are not the copy cost**: the out-of-place `index_copy` merge could move up to ~40 GiB per
  update on this geometry (ATen 2.10 semantics, verified in the 2026-09-08 review), but at ~1 TB/s
  that is ~40 ms; the 124,752 launches are the cost.
- **Optimizer work scales with stored, not active, parameters**: with 131,072 targets per update
  every expert receives tokens (expected 131,072 × k / E per layer; 16,384 here, 4,096 for 64
  experts Top-2), so every expert matrix has a gradient every update. Newton–Schulz costs
  10 × (4m²n + 2m³) per matrix regardless of how many tokens the expert saw. For the accepted
  E64/Top-2/h352/d256/L8 geometry: 1,536 expert matrices, ≈ 1.93 TFLOP per update, versus a
  6·N_active·tokens proxy of 7.82 TFLOP — about 25 % of training FLOPs, and 1,600 Newton–Schulz
  problems per update if issued one matrix at a time (ADR 0170 batches them by shape).
- **Where precision can act**: FP8 doubles the peak of the matmul share only (≈ 1.4 s of 19.0 s
  here); launches, Python, copies and synchronizations are precision-independent. Quantized
  optimizer state saves memory (0.15–1 GiB on this geometry), not time.

## Observed routing during the probe

Update 7: no empty expert in the batch, maximum expert load 69.15 %. Update 9: one empty expert of
160 and maximum load 86.09 % in one layer. Few updates with replay; not evidence about long-run
balance.

## What this evidence does not establish

No causal attribution of seconds to any single mechanism (the trace shows counts and self times,
not the critical path); no measurement of the accepted 64-expert geometry; no GPU measurement of
ADRs 0170–0172, whose equivalence was proven on CPU only.
