# 2026-09-24: live MoE run 003, isolated Kaggle T4 resume qualification

The user-supplied Kaggle proxy exposed a kernel with two Tesla T4s (15,360
MiB each), Python 3.12.12, and notebook source consisting of one seven-line
secret-loading cell. `GITHUB_TOKEN`, `WANDB_API_KEY`, `HF_TOKEN`, and
`SMALL_LLM_HF_REPO_ID` resolved in that kernel; no secret contents were printed.
The notebook source did **not** contain a clone, installer, or trainer cell.
The proxy's `/api/contents/.virtual_documents/__notebook_source__.ipynb`
was a plain source fragment, not a valid saved `.ipynb` for editing; executing
kernel commands does not persist a Kaggle background notebook version.

W&B API reported `rocchissimo936-none/Small-LLM/moe-100b-superbpe-003`
**running** (summary step moved from ~270,117 to ~275,609 during this work).
HF read of `roccoangelella/small-llm-100m-qualification-checkpoints` resolved
`latest.json` to `step-00265000` at the time of the probe. Its immutable
receipt identifies run origin `5f08941028cce913f336a8aa2f8ce5fb6231ca4c`,
W&B entity/project/run ID, and the checkpoint bucket. W&B launch config
identified the *v2* public dataset bucket
`abcastor/small-llm-corpus-100b-v2-dataset` and dataset run ID
`moe-100b-superbpe-b64-dataset-003`; the corpus contract plans 762,940
absolute updates. The HF token could read the private checkpoint and public
dataset. Runtime `huggingface-hub` needed upgrade from 1.4.1 to >=1.5 for
Storage Bucket support; `fla-core==0.5.2` was absent until installed.

Using the public `moe-8e-top1` checkout at `33faf6f7de5f21de404f584f0355e204ad6e0f8f`,
`moe_production.prepare_dataset` restored the 1,161,780,490-byte trainer
state and validated the source cursor/rolling dataset, with first resumed
block 265,000 (64 sequences). It passed model identity E64/Top-2, BF16,
8,000 semantic IDs and 8,192 padded rows. FLA/Triton compilation with
unbounded tuning stalled on the first backward microbatch for >12 minutes.
Installing the already-qualified six-candidate cap from the dense Kaggle
runtime finished cold tuning in ~5 minutes; later BF16 updates took ~20 s
on **one** T4. T4 native BF16 is unavailable, but the actual PyTorch BF16
training step worked. The second T4 was not used; no MoE DDP implementation
was claimed.

To avoid touching the live writer, a *local copy* of the verified checkpoint
was run with W&B and remote publication **disabled**, a one-update budget,
one held-out validation block, and the same accepted MoE, optimizer, data
cursor, and WSD schedule. The real CLI completed step 265,001 (loss 2.77538,
131,072 targets, peak GPU allocation 4,316,308,480 bytes), validation loss
2.61939, and a 1.16 GB local checkpoint. A second isolated CLI invocation
resumed that local checkpoint and completed step 265,002 (loss 2.77538,
validation loss 2.61991), writing another complete local checkpoint. This
second attempt found an observation-origin check that rejected a new provider's
own subsequent artifacts; `trainer/observation.py` was fixed and the resume
rerun succeeded. The isolated probe never advanced HF latest or wrote W&B.

**Not verified:** a saved/background Kaggle notebook, an authenticated W&B
`resume=must` from Kaggle, or publishing a Kaggle-created checkpoint to the
live HF latest pointer. The existing writer was still running, so testing
these on the same run would be a double-writer risk. This is not a green light
for background training. Procedure and safety gates:
[MoE Kaggle continuation](../runbooks/moe-kaggle-continuation.md).
