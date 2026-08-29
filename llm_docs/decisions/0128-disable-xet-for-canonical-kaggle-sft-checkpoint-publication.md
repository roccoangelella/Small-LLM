---
status: accepted
date: 2026-08-28
---

# ADR 0128: disable Xet for canonical Kaggle SFT checkpoint publication

## Context

The 100M/2B 10% S0 run `100m-2b-sft-s0-10pct-001` was launched from Kaggle with the canonical command:

```bash
python kaggle/launch_sft.py train \
  --model 100M \
  --tokens 2B \
  --sft-fraction 10%
```

No `--max-steps-this-session` cap was supplied. The run reached optimizer step 3,750 and consumed 120,638,893 loss-bearing targets. Step 3,750 itself completed normally, and the local `step-00003750` checkpoint completed. The log then entered `publication:start` for that checkpoint and terminated before `publication:done`, after which W&B was finalized as failed.

The failure signature matches the already-observed Kaggle R-SFT failure mode in which a large Hugging Face Xet-backed trainer-state upload stalled and the training process was lost. R-SFT was already hardened by forcing `HF_HUB_DISABLE_XET=1` and `HF_HUB_DISABLE_PROGRESS_BARS=1`.

The frozen 10% train bundle contains 6,220 optimizer blocks / 200,099,738 realized train targets, so step 3,750 was not the planned end of the run.

## Decision

The canonical `kaggle/launch_sft.py` entry point must force:

```text
HF_HUB_DISABLE_XET=1
HF_HUB_DISABLE_PROGRESS_BARS=1
```

before any optional uv Python bootstrap and before the detached-worktree/DDP training subprocess is launched.

This hardening applies to the canonical SFT launcher, including the 100M/2B 10% run. It changes only the Hugging Face transfer mechanism and notebook progress rendering; it does not change the model, optimizer, scheduler, data stream, checkpoint identity, or scientific trajectory.

## Resume contract

The existing automatic verified-resume behavior remains unchanged. The SFT trainer compares valid local and verified remote checkpoints and selects the highest valid step. Therefore:

- if the original Kaggle filesystem still contains the valid local step-3,750 checkpoint, relaunch may resume from step 3,750;
- on a fresh Kaggle session, relaunch resumes from the newest successfully published and verified remote `latest` checkpoint, expected to precede the failed step-3,750 publication unless that publication actually completed its two-phase pointer update;
- the operator should use the same canonical training command and the same run ID derived from `--sft-fraction 10%`.

The two-phase Hugging Face publication protocol remains the durability gate; incomplete remote uploads must not advance `latest.json`.
