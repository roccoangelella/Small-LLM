# MoE two-T4 bottleneck investigation and safe microbatch change — 2026-09-24

**Scope:** isolated Kaggle notebook tests from the *same* verified live
step-277,500 input snapshot, with 64 sequences / 131,072 targets per global
update. We never used run 003's W&B ID as an output and passed no Hugging Face
checkpoint-publication flags. The live writer was left alone.

## Real W&B comparison, same code family and data

| T4 run | BF16 microbatch / rank | W&B steps | warm median `train/tokens_per_second` (steps 3–16) | final validation loss |
| --- | ---: | ---: | ---: | ---: |
| [baseline](https://wandb.ai/rocchissimo936-none/Small-LLM/runs/moe-2t4-kaggle-perf-20260924-5323e2fc) | 1 | 16 | 15,275 | 2.624157 |
| [new public `moe-8e-top1` code](https://wandb.ai/rocchissimo936-none/Small-LLM/runs/moe-2t4-kaggle-mb4-20260924-81dc6944) | 4 | 16 | **18,166** | 2.624554 |

That is **+18.9%** in the actual W&B training metric, not an inference
from the cold first step. All 16 training rows arrived for both runs;
W&B finished normally. The new run was launched from a clean public checkout
at `dbbd1c4297428a6c9fd59ab98c9d1978ca343ee1`; it completed
training, one-block validation, and a local checkpoint at 277,516. A new
two-rank, W&B-disabled run then loaded **that** checkpoint and completed
step 277,517 with validation/save. A fresh process verified the new
Torch-ZIP checkpoint and resolved `latest` to 277,517; afterward we removed
only the temporary benchmark checkpoint copies, retaining W&B/log evidence.
The difference in final validation loss is ~0.000397; this checks
short-run sanity, *not* long-run numeric equivalence.

## Bottleneck, not idle GPU or missing second rank

- The same Kaggle stack reports Tesla T4 compute capability 7.5. Native BF16
  support is **false** (`torch.cuda.is_bf16_supported(including_emulation=False)`)
  even though the default emulation-inclusive query returns true. A 2048²
  GEMM microbenchmark on one idle T4 gave median **11.106 ms BF16**, **0.970
  ms FP16**, **5.390 ms FP32** over 25 CUDA-event repetitions. This small
  GEMM is not the model, but proves the accepted BF16 recipe cannot use the
  T4's native half-precision tensor-core path.
- A one-step `torch.profiler` sample of the original BF16 DDP topology on
  rank 0 had 3,862 `aten::bmm` and 7,936 `aten::mm` calls, consuming ~3.15 s
  and ~1.57 s in their respective self-CUDA categories; 30 NCCL reductions
  were measured, with ~1.53 s self CUDA in the all-reduce category. Profiler
  attribution is nested and its cold step is slower: these numbers **must not
  be added** or interpreted as a warm-step speed estimate. The observed
  MAGMA sgemmEx kernels take BF16 inputs into float math on this GPU.
- Both T4s reported ~99–100% GPU utilization throughout the warmed-up
  microbatch-4 W&B test; ~9.5 GiB was occupied per card in the sampling.
  The live H100 run at the same time reported ~346k targets/s but uses a
  **native-BF16 H100, compiled blocks, microbatch 32**, so parameter count
  alone is not a fair cross-hardware throughput prediction.

## Measured tuning and limits

An isolated four-step sweep of identical starting checkpoint/blocks, with
W&B and remote checkpoint writes disabled, yielded warm (steps 2–4) medians:
MB1 (16-step W&B reference) 15.3k; MB2 15.9k; MB4 17.5k; MB8 17.9k targets/s.
MB8 reached **14,952,693,760 bytes PyTorch-reserved on a T4** on one block:
there is too little headroom to make it the live default. MB4 peaked at
~9.72 GB reserved on the same blocks. Enlarging DDP's gradient bucket cap
from the default to 200 MB at MB4 gave **17.47k**, not an improvement over
17.54k: do not add that tweak.

Unqualified out-of-tree FP16 and no-autocast FP32 variants did **not** finish
a first global step within 210 s and 180 s respectively; neither produced
a usable throughput number. Their cause needs separate investigation, and
neither was committed or used with the live run. Replacing accepted BF16 with
FP16 also changes training numerics and potentially gradient overflow
handling; no such precision switch is authorized by this throughput test.

**Conclusion:** the safe execution-only optimization is microbatch 4 (8
accumulation passes per rank instead of 32), yielding a measured ~19% gain.
This **does not** make T4 match H100: the hardware lacks native BF16 and the
routing+GDN workload performs thousands of small GEMMs per optimizer step.
No promise of an order-of-magnitude speedup under the frozen BF16 recipe.
