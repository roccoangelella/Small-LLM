# 100M/2B SFT dual-T4 microbatch-4 startup OOM — 2026-08-13

The first live 100M/2B SFT attempt that passed dependency and parent-artifact
startup gates reached the two-Tesla-T4 DDP engine with the frozen FP16
PyTorch 2.10.0 / CUDA 12.8 / Triton 3.6.0 / `fla-core==0.5.2` stack.

Rank 0 started the deliberate no-step FLA/Triton backward prewarm at local
shape `4x2048`. During backward, PyTorch reported:

```text
CUDA out of memory. Tried to allocate 786.00 MiB.
GPU 0 total capacity: 14.56 GiB
free: 6.81 MiB
allocated by PyTorch: 13.79 GiB
reserved but unallocated: 91.66 MiB
```

The failure occurred inside `_prewarm_raw_model` before DDP wrapping completed.
The prewarm explicitly performs no optimizer step, and the process failed
before the training session could consume an SFT block or create a checkpoint.
Rank 1 was terminated by `torchrun` after rank 0 failed.

This rules out per-rank execution microbatch 4 for the 100M SFT model on a
14.56-GiB Tesla T4 under the qualified runtime. The next bounded hardware
start uses microbatch 2. That is an execution-slicing change only: the same
global SFT target count is normalized once, DDP average compensation is
unchanged, and each immutable SFT block still produces exactly one optimizer
update.
