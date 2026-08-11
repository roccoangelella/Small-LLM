---
status: superseded
date: 2026-08-11
superseded_by: 0041
---

# Launch 100M / 2B pretraining on Modal

## Decision

Launch the approximately-100M-parameter model on the existing verified 2B-token finite dataset using the canonical `modal/launch.py` path.

The production run keeps the accepted pretraining contract unless a later ADR explicitly changes it:

- model preset: `100M` / trainer `substantive`;
- token budget: `2B`, reusing the model-agnostic `20m-2b` finite dataset contract;
- GPU request: `H100`, allowing Modal's compatible automatic H200 upgrade;
- precision: FP16 autocast with FP32 master parameters;
- microbatch: first-run qualification over 4, 8, and 16, then freeze the fastest safe result;
- optimizer batch: one complete 16-sequence prepared block per optimizer update;
- checkpoint cadence: every 250 successful optimizer updates plus the trainer's final checkpoint;
- checkpoint durability: `small-llm-runs` Modal Volume with verified automatic resume from the newest valid checkpoint;
- validation cadence: every 250 successful optimizer updates;
- W&B: online `Small-LLM` logging with stable run ID `100m-2b-data-001` and exact resume semantics;
- source/configuration drift: fail closed according to the Modal runtime contract.

The legacy Hugging Face dataset-keyed checkpoint publication remains disabled for this Modal run because the same 2B dataset is reused across model sizes and that namespace would collide. Modal Volume checkpoint durability is the canonical checkpoint transport for this trajectory.

## Rationale

The existing 20M / 2B Kaggle trajectory is too slow for the next model-scale experiment. ADR 0039 established Modal as the canonical platform for new GPU pretraining and preserved the scientific trainer contract while allowing faster Hopper hardware. This decision authorizes the first 100M / 2B production trajectory on that path.

## Launch command

```bash
modal run --detach modal/launch.py --model 100M --tokens 2B --gpu H100
```

## Supersession

ADR 0041 keeps the Modal/H100 launch authorization, checkpointing, W&B, precision, optimizer, and run identity, but replaces this ADR's 16-sequence dataset/optimizer block and 4/8/16 microbatch probe with a byte-preserving block-64 Modal corpus and 16/32/48/64 probe.
