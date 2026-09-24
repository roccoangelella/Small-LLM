# Kaggle two-T4 MoE W&B performance test — 2026-09-24

We ran a **separate** 16-step MoE test in the existing Kaggle notebook kernel,
from a copied and verified step-277,500 snapshot of run 003. The training
process used both Tesla T4s, the frozen BF16/64-sequence model and dataset,
and microbatch 1 per rank. Only the test run wrote to its own local checkpoint
folder. There were **no HF checkpoint-publication or best-model flags**;
the read-only HF dataset shard source remained enabled. The live run 003,
its checkpoint files, and its W&B ID were not used as outputs.

- W&B: [temporary two-T4 test](https://wandb.ai/rocchissimo936-none/Small-LLM/runs/moe-2t4-kaggle-perf-20260924-5323e2fc),
  run ID `moe-2t4-kaggle-perf-20260924-5323e2fc`, **finished**.
- All 16 `train/tokens_per_second` rows reached W&B, steps 277,501–277,516.
  Each global step processed 131,072 targets. Initial step: **11,002 tok/s**;
  steps 3–16: **15,275 tok/s median** (12,812–15,974; mean 15,122),
  **8.58 seconds/step** median. The last W&B point was 15,388 tok/s.
- Both GPUs sampled at ~95–99% reported GPU utilization and ~6–7 GiB
  occupied memory during sustained training. This is not idle ranks or an
  initial compilation delay mistaken for steady-state throughput.
- Final one-block validation passed (loss 2.624157); a local checkpoint at
  277,516 was saved (~1.16 GB). W&B finished normally. No remote-publication
  event occurred. The isolated local checkpoint copies can be deleted after
  verifying the complete-checkpoint contract; the W&B run and local log remain.
- At the same time, live H100 run 003 W&B reported ~345,929 tok/s, configured
  for BF16, **microbatch 32, compiled blocks**, unlike this uncompiled T4
  microbatch-1 execution. That is ~23× faster than the warmed-up T4 median.
  The numbers are real but not a like-for-like kernel/batch comparison.

**Why the gap is credible:** the T4 is compute capability 7.5. Kaggle's
PyTorch 2.10 reports BF16 supported only when **emulation is allowed**;
`torch.cuda.is_bf16_supported(including_emulation=False)` is `False`.
For a simple 2048² GEMM on one idle T4, median event times over 25 runs
were **0.97 ms FP16, 11.11 ms BF16, 5.39 ms FP32**. That microbenchmark
is not a model-throughput estimate, but directly confirms BF16 is a bad
compute format on these cards. The accepted live checkpoint must not silently
switch to FP16: doing so changes training numerics and has not been qualified
as a continuation recipe.

**Integration caveat discovered during cleanup:** the generic checkpoint
scanner tried to `pickle.load` the Torch ZIP serializer's `trainer_state.pkl`.
This fails with `UnpicklingError` for the *valid* local MoE checkpoint, even
though the direct training resume loader understands the format. The scanner
now selects the existing streamed Torch loader for ZIP checkpoints and keeps
plain-pickle support; a focused regression test covers production `latest`
resolution. The first clean Kaggle checkout at `46fe269` subsequently
verified `complete_checkpoint` on the **actual** saved ZIP state (step 277,516,
36,374,577,152 consumed targets); a fresh Python process also resolved
`latest` to that step with exactly one remaining update. A long-lived notebook
kernel had cached the older scanner module and incorrectly quarantined a later
isolated test snapshot; that snapshot was restored **only after** checking
its local manifest, metadata, and Torch state in a fresh process. New
background notebook processes import the fixed module. This test did **not**
resume the live W&B run or publish to its HF checkpoint bucket.

W&B setup note: the first launch using `--wandb-resume never` failed to create
a new run (`no data but must resume` in W&B core), before any optimizer step.
It was stopped. Explicit `resume="allow"` successfully created the unique
empty test run; the bounded trainer then resumed **that separate ID** with
`--wandb-resume allow` and completed normally. This does not affect the
live run 003's ID or resume policy.
