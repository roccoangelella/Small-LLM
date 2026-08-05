# 20M/100M W&B startup handling — 2026-08-05

Repeated Kaggle launches reached the real-training boundary but failed before the first optimizer update because `wandb.init()` exhausted its default 90-second initialization window.

Operational decision:

- keep W&B in online mode for the 20M-model/100M-token run;
- set `WANDB_INIT_TIMEOUT=300` in the one-click Kaggle entrypoint before importing the launcher;
- preserve the pinned scientific training commit, optimizer configuration, microbatch-4 selection, 16-sequence optimizer block, dataset order, checkpoint cadence, and W&B run identity;
- treat the PyTorch message about unavailable NumPy as a non-fatal warning separate from the W&B communication timeout;
- do not claim that a failed W&B initialization completed any optimizer update or produced a resumable checkpoint.

The change is operational only: it gives W&B up to five minutes to establish its online run on Kaggle without modifying training mathematics.
