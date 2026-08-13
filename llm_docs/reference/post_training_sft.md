# Supervised fine-tuning implementation reference

_Last reviewed: 2026-08-13 Europe/Rome_

## Current status

The SFT pipeline is implemented and the first frozen S0 qualification on the completed 20M/500M parent is finished. That run **failed behavioral qualification** despite strongly improving masked held-out SFT likelihood. It must not be described as a promoted instruction-tuning recipe.

Observed S0 result:

```text
parent: 20m-500m-dataset-001 / step-00015264
parent consumed targets: 500,156,416
SFT run: 20m-500m-sft-s0-001
SFT checkpoint: step-00000621
realized SFT train loss-bearing targets: 20,006,234
instruction behavior: 0/30 passed
EOS termination: 0%
runaway: 100%
base eval_core_v1: modest broad regression
```

Canonical evidence: [`../evidence/20m/20m_500m_sft_full_qualification_2026-08-11.md`](../evidence/20m/20m_500m_sft_full_qualification_2026-08-11.md).

ADR 0067 authorizes the next experiment on the completed 100M/2B parent. It keeps the S0 source stratification and 85/15 instruction/replay mixture unchanged, keeps the requested SFT horizon at 4% of the verified parent target count, and uses Kaggle two-T4 DDP. For the exact 2,001,000,448-target parent, the requested SFT train horizon is 80,040,017 loss-bearing targets. This is an experiment after the failed 20M/500M S0 qualification, not a promotion of the recipe in advance.

## Operator surface

The canonical Kaggle entrypoint is:

```text
kaggle/launch_sft.py
```

It provides profile-driven `prepare`, `publish`, `train`, and `eval` actions. The launcher verifies parent/bundle identities and exact resume state rather than relying on ad-hoc notebook commands.

The registered 100M/2B profile is `100m-2b-sft-s0-001`. Its Kaggle training path launches two NCCL processes across exactly two Tesla T4 GPUs. Variable-size SFT blocks are split between ranks without duplicating loss-bearing examples; an all-ignored graph-carrying row is used only when needed to equalize DDP synchronization points. W&B, evaluation, checkpoint saving, and remote publication remain rank-zero side effects. Resume selection is synchronized so both ranks load the exact checkpoint chosen by rank zero.

## S0 data contract

The implemented S0 source pins `HuggingFaceTB/smol-smoltalk` revision:

```text
f80219b491a28e79600fa320e075752f1ea0303e
```

The frozen instruction-source allocation within the instruction portion is:

```text
75.0% smol-magpie-ultra-short
10.0% smol-contraints
 7.5% smollm-rewrite-30k
 7.5% smol-summarize-20k
```

Overall S0 target mixture:

```text
85% filtered instruction targets
15% frozen original-distribution ClimbMix replay targets
```

Identity split:

```text
train 95.0%
validation 2.5%
test 2.5%
```

Prompt derivatives are grouped by a source-independent normalized non-assistant conversation identity before tokenization. Exact duplicates and in-repo behavior-suite contamination are removed. Prepared records retain source/provenance identity and the immutable bundle verifies all split manifests/shards/hashes.

## Budget rule

ADR 0032 historically defined requested SFT train loss-bearing targets as 4% of the verified completed parent pretraining target count. The completed 500M parent requested 20,006,256 and realized 20,006,234 complete-record targets.

ADR 0067 keeps the 100M/2B experiment on the same 4% budget rule. The completed 100M/2B parent has 2,001,000,448 targets, so the requested SFT horizon is exactly 80,040,017 targets by integer floor.

The SFT bundle builder remains finite and fail-closed: it verifies the requested target horizon and removes an incomplete output if the configured sources are exhausted before the permitted complete-record shortfall. No silent data repetition is introduced.

## Chat/loss contract

The GPT-2 vocabulary is unchanged. Chat text uses explicit `System:`, `User:`, and `Assistant:` prefixes and GPT-2 token 50256 as supervised assistant-turn EOS. Loss is active on assistant content/EOS and on ordinary replay next-token targets; system/user/role-prefix tokens are masked for instruction records.

One conversation occupies one model sequence. Cross-conversation packing is disabled. Overlength examples drop oldest complete dialogue pairs first; assistant targets are not silently truncated to force fit.

## Training contract

SFT starts from verified parent model weights but creates fresh optimizer/scheduler/scaler state; it does not inherit pretraining optimizer moments. Parent identity, tokenizer/model geometry, requested budget, prepared bundle identity, optimizer recipe, schedule, and RNG state are bound to the run/checkpoint and resume rejects drift.

For the 100M/2B profile, training resolves the completed stable parent model artifact rather than depending on a mutable rolling pretraining checkpoint. Kaggle DDP preserves one global optimizer update per immutable SFT block and scales each rank's local summed loss against the block's global loss-bearing target count, preserving the serial global-token objective.

## Qualification contract

Do not select SFT by held-out SFT loss alone. Qualification must inspect both:

- instruction/behavior acquisition (including EOS/runaway/repetition and task-category passes);
- base-capability retention on unchanged `eval_core_v1`.

The first S0 result demonstrates that masked SFT-distribution likelihood can improve strongly without producing usable deterministic instruction following. The 100M/2B 4% run must pass the same comprehensive parent-versus-SFT qualification before it is treated as a successful post-trained model.
