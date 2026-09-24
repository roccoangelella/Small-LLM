# MoE run 003 on the Kaggle two-T4 notebook

This continuation uses **both T4s** through the accepted MoE-specific adapter
`kaggle/moe_train_dual_t4.py`. It splits each 64-sequence optimizer block 32/32,
reduces the global gradients, merges per-expert score quantiles across ranks,
and commits the same Quantile Balancing buffers on both replicas. Rank 0 alone
owns W&B, validation, checkpoints, HF uploads and shard eviction; rank 1 reads
the verified local rolling shard window. `kaggle/src/dual_t4_train.py` is a
**dense-model** DDP shim; do not use it to train MoE. Do not change the run's
BF16 precision. T4 lacks native BF16: the pinned PyTorch 2.10 stack
**emulates** BF16 here. Training works, but it is much slower than on H100;
do not extrapolate the first cold step or silently switch the live checkpoint
to FP16. The six-config Triton autotune cap is installed before FLA imports;
the cache is session-local.

This run is `moe-100b-superbpe-003`, not the older run 001 or the older
`roccoangelella/small-llm-moe-100b-superbpe-dataset` corpus. The verified
latest receipt pins origin `5f08941028cce913f336a8aa2f8ce5fb6231ca4c`,
W&B `rocchissimo936-none/Small-LLM/moe-100b-superbpe-003`, private checkpoint
bucket `roccoangelella/small-llm-100m-qualification-checkpoints`, and the
**v2** dataset bucket `abcastor/small-llm-corpus-100b-v2-dataset` / run
`moe-100b-superbpe-b64-dataset-003`. The corpus plan is 762,940 *absolute*
steps. The pinned run origin and the new clean `moe-8e-top1` executor commit
are separate identities; do not try to check out the origin SHA on this branch.
The code validates the receipt, stages the needed block from the remote
frontier, and verifies its frozen manifest and contract before GPU dispatch.

## Notebook cells

Use the **saved** notebook, not just an interactive kernel command: Kaggle
background execution replays saved cell source. The notebook provided to the
agent contained only a cell loading `GITHUB_TOKEN`, `WANDB_API_KEY`, `HF_TOKEN`,
and `SMALL_LLM_HF_REPO_ID` from `UserSecretsClient`. All four names were
resolvable in the interactive kernel; GitHub credentials are unnecessary for
this public clone. Do not paste secret *values* into cells or outputs.

Add and save a setup/preflight cell **after** the existing secret cell:

```python
import os, pathlib, subprocess, sys
repo = pathlib.Path('/kaggle/working/Small-LLM')
if not repo.exists():
    subprocess.run(['git', 'clone', '--depth', '1', '--branch', 'moe-8e-top1',
                    'https://github.com/roccoangelella/Small-LLM.git', str(repo)], check=True)
else:
    subprocess.run(['git', '-C', str(repo), 'pull', '--ff-only', 'origin', 'moe-8e-top1'], check=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
                'huggingface-hub>=1.5,<2', 'wandb==0.26.1', 'fla-core==0.5.2',
                'tokenizers==0.23.2', 'tiktoken==0.14.0'], check=True)
subprocess.run([sys.executable, str(repo / 'kaggle/moe_resume.py')], check=True)
```

The last line runs the **read-only-remote** preflight: it restores the newest
verified checkpoint into `/kaggle/working/moe-resume-003`, stages the matching
train/validation window, checks the receipt and W&B run, then exits without
training or publishing. Local disk downloads are expected. It fails if the
branch is dirty/stale or the remote pointer advances while staging; retry
against the new pointer rather than ignoring the failure. On a fresh Kaggle
background session the setup cell must execute before the start cell.

Only after the *other* provider is stopped, its last checkpoint's HF
`latest.json` has been verified, W&B no longer reports `running`, and the
preflight succeeds, save and execute a separate training cell:

```python
import pathlib, subprocess, sys
repo = pathlib.Path('/kaggle/working/Small-LLM')
subprocess.run([sys.executable, str(repo / 'kaggle/moe_resume.py'), '--start'], check=True)
```

The launcher re-checks the pointer and writer state before training and uses
`--wandb-resume must` on the same W&B ID. The nine-hour wall drain performs
validation and a *synchronous verified remote checkpoint* even between regular
1,000-step checkpoint boundaries. If the cell fails, inspect the traceback and
remote pointer; do not start a second writer. A subsequent Kaggle session
replays setup and uses `--resume latest`, never the older `best` pointer or a
local checkpoint behind remote latest.

## Qualification / remaining gate

On the supplied Kaggle session, the HF token read the private checkpoint and
v2 corpus; the CPU gate restored the optimizer/cursor and staged the matching
train/validation window. An earlier **single-T4** isolated probe at step
265,000 completed two updates and local checkpoint restores (~7,050 target
tokens/s); this is historical qualification, not the production topology.
The dual-T4 offline probe restored the real step-272,500 snapshot, trained
step 272,501, validated one block, saved a 1.16 GB exact-resume checkpoint,
and reloaded that new checkpoint across both ranks to complete step 272,502.
It used the real accepted model and frozen WSD schedule with W&B and remote
publication disabled. The first dual update reached ~10,703 target tokens/s
and 4.32 GB peak per-GPU allocation at microbatch 1; microbatch 2 was
slower (~9,241 targets/s), so the production default remains 1. Router bias
matched the serial update **exactly** and the largest model-weight difference
was 2.67e-5 (BF16/reduction-order drift). The clean public
`moe-8e-top1` checkout at `7ed4e29` separately passed preflight at HF
step 277,500, then completed an isolated two-rank GPU step and an exact
second local resume from its newly saved checkpoint. A separate 16-step
**online W&B** benchmark with a unique ID confirmed 15.3k warmed-up targets/s
on both T4s; the original H100 run showed ~346k at BF16 microbatch 32 with
compiled blocks. W&B logged all 16 steps, final validation and local save
passed, and HF checkpoint publication was disabled. Its Torch-ZIP checkpoint
also exposed a generic `latest` discovery bug fixed in
`dataset/src/checkpoint_sequence.py`; direct resume alone did not test this
path. [Offline measurements](../evidence/2026-09-24-moe-kaggle-dual-t4-probe.md) · [W&B benchmark](../evidence/2026-09-24-moe-kaggle-wandb-perf-test.md).

**This does not authorize starting now.** W&B still reported the original
writer `running` during qualification; no simultaneous second writer, live W&B
resume, or live HF publication was attempted. The Kaggle notebook's new cells
must be saved via its UI before background execution; a Jupyter proxy kernel
execution alone does not modify the saved notebook source. Verify latest/W&B
state immediately before requesting a background run. A slower second writer
can still overwrite a newer `latest.json`, so speed is not a concurrency guard.
