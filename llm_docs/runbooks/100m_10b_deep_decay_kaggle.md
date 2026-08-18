# 100M / 10B deep-decay on Kaggle 2xT4

ADR 0099 moves the ADR-0095 deep-decay continuation to Kaggle exact-batch two-T4 DDP while preserving the exact step-15,500 model/optimizer/scaler/RNG/data state, the 64-sequence global optimizer block, and the LR schedule. The source checkpoint used execution microbatch four; the Kaggle fork uses microbatch two because microbatch four OOMed on the 100M model/T4 path while microbatch two subsequently completed 250 real optimizer updates with material memory headroom.

## Kaggle prerequisites

Use a Kaggle notebook/session configured with **2x Tesla T4** and internet access. Clone or update the repository to `main`, then provide the same secrets used by production training:

- `HF_TOKEN`
- `SMALL_LLM_HF_REPO_ID`
- `WANDB_API_KEY`
- optional `WANDB_ENTITY`
- optional `SMALL_LLM_HF_DATASET_BUCKET_ID` when the dataset bucket does not use the default `<SMALL_LLM_HF_REPO_ID>-datasets` name

The launcher fails closed if both visible CUDA devices are not Tesla T4s.

## Inspect the frozen contract without training

```bash
python kaggle/launch.py deep-decay --model 100M --tokens 10B --dry-run
```

Expected geometry:

```text
source checkpoint:           step-00015500
source microbatch:           4
execution:                   kaggle_dual_t4_ddp_block64
world size:                  2
sequences per block:         64
sequences per rank:          32
Kaggle microbatch:           2
local microbatches/rank:     16
remote checkpoint cadence:   250 updates
final step:                  76294
```

Training math uses NCCL. Prewarm, cadence, and final rendezvous use a one-hour CPU/Gloo control group so rank-zero validation/checkpoint/publication work cannot reproduce the known ten-minute NCCL watchdog timeout from the 100M SFT lane.

## First live gate

Use one durability interval for the first block-64 live qualification:

```bash
python kaggle/launch.py deep-decay --model 100M --tokens 10B --max-steps-this-session 250
```

Before treating the Kaggle lane as qualified, verify the logs show:

- two Tesla T4 devices and `world_size=2`;
- startup banner `global block=64`, `32 sequences/rank`, `microbatch=2`, `control_barrier=gloo-1h`;
- exact source fork or verified Kaggle deep-decay restore;
- the expected LR for the resumed committed-target count;
- finite loss and gradient norms;
- W&B side effects only on rank zero;
- successful cadence rendezvous after validation;
- a manifest-valid local checkpoint at the expected final step;
- the same checkpoint published under `run/100m-10b-deep-decay-from-step15500/...` in the configured Hugging Face model repository.

For a fresh fork, the bounded segment must finish at `step-00015750`. If a newer valid Kaggle deep-decay checkpoint already exists, it must instead finish exactly 250 updates after that checkpoint.

## Continue the run

After the live gate, launch without a session cap:

```bash
python kaggle/launch.py deep-decay --model 100M --tokens 10B
```

Kaggle may stop a notebook before the full remaining horizon. Resume by rerunning the **same command** in a new two-T4 session. The launcher restores the newest manifest-verified Kaggle deep-decay Hugging Face checkpoint and stages the rolling dataset from that exact next block.

If desired, bound individual notebook segments explicitly:

```bash
python kaggle/launch.py deep-decay --model 100M --tokens 10B --max-steps-this-session 5000
```

Do not add `--resume`; resume is automatic and the canonical launcher rejects that flag.

## Fail-closed source rule

When no Kaggle microbatch-two deep-decay checkpoint exists yet, the launcher accepts only:

```text
run/100m-10b-data-001/latest.json
  checkpoint_id = step-00015500
```

It never substitutes a nearest checkpoint and never starts from an older cooled/diagnostic continuation. The fork keeps the exact source model/optimizer/scaler/RNG/data state, rewrites the execution microbatch field from four to two plus the already-authorized ADR-0095 scheduler fields, and recomputes the checkpoint configuration hash. Once the Kaggle deep-decay namespace has a valid microbatch-two checkpoint, that namespace becomes the resume authority.

## Preserved scientific state

The execution migration preserves:

- the 100M GDN-2 hybrid model;
- FP16 execution;
- hybrid Muon + AdamW optimizer and its restored state;
- scaler/RNG state;
- exact 10B corpus order and cursor;
- frozen 16-block validation prefix;
- 64-sequence global optimizer block;
- the ADR-0095 three-phase deep-decay schedule;
- final step 76,294 / 10,000,007,168 targets.

Microbatch is execution slicing rather than optimizer-batch size. The switch `4 -> 2` creates more accumulation slices inside the same 64-sequence update. DDP and finer accumulation can change floating-point reduction order, so the migration targets numerical equivalence rather than bitwise identity.

## T4 evidence

- Microbatch four OOM: [`../evidence/scaling/100m_2b_sft_t4_microbatch4_oom_2026-08-13.md`](../evidence/scaling/100m_2b_sft_t4_microbatch4_oom_2026-08-13.md)
- Microbatch two completed 250 real updates with peak allocated 8.35 GiB and peak reserved 11.70 GiB; the later failure was the cadence NCCL watchdog, not memory: [`../evidence/scaling/100m_2b_sft_step250_nccl_timeout_2026-08-13.md`](../evidence/scaling/100m_2b_sft_step250_nccl_timeout_2026-08-13.md)
