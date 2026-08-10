# Supervised fine-tuning implementation reference

_Last reviewed: 2026-08-10 Europe/Rome_

This document describes the current SFT implementation and operational contract. ADR 0032 authorizes qualifying SFT on the completed 20M/500M parent while the fresh 20M/2B pretraining run proceeds. ADR 0033 freezes the comprehensive post-SFT scorecard plus the pretraining-equivalent T4 microbatch/cadence defaults.

## Canonical user surface

The canonical Kaggle entry point is:

```text
kaggle/launch_sft.py
```

It is profile-driven in the same style as `kaggle/launch.py`. Registered parent profiles are currently:

```text
20M model / 500M pretraining parent
20M model / 2B pretraining parent
```

The launcher exposes three operational actions:

```text
prepare   build and verify the immutable SFT bundle
a train   (spelling: train) launch or exactly resume SFT
eval      run the comprehensive parent-versus-SFT qualification report
```

Use `python kaggle/launch_sft.py profiles` or `--dry-run` before a live job.

## Current data contract

The preparation path pins `HuggingFaceTB/smol-smoltalk` to an exact revision and retains only the configured small-model-oriented source subsets. The instruction-source allocation is:

```text
75.0% smol-magpie-ultra-short
10.0% smol-contraints
 7.5% smollm-rewrite-30k
 7.5% smol-summarize-20k
```

Those percentages are within the instruction portion and are measured by loss-bearing target tokens.

The identity split is frozen by ADR 0032:

```text
train:       95.0%
validation:   2.5%
test:         2.5%
```

Prompt derivatives are grouped before tokenization by normalized non-assistant conversation identity so alternate labels for the same prompt cannot cross splits. Prepared records carry source revision, Apache-2.0 license identity, upstream index, content hash, and split-group identity. Exact duplicate conversations and deterministic in-repo behavior-suite contamination are removed before immutable shard construction.

The current S0 implementation default remains 85% instruction / 15% original-distribution replay for the 500M engineering qualification. The historical 15% replay ratio is still an experimental baseline rather than a claim of global optimality; the 500M evidence is intended to inform whether it is retained unchanged for the 2B-parent run.

## Budget scaling

ADR 0032 replaces the historical fixed 4M SFT budget with 4% of the verified completed parent pretraining token count, measured in SFT loss-bearing target tokens.

For the completed 500M parent:

```text
verified parent consumed targets: 500,156,416
requested SFT target budget:       20,006,256
```

The immutable dataset never truncates a retained assistant target merely to hit the arithmetic ceiling. Therefore the manifest records the exact realized complete-record horizon and it may finish slightly below the requested ceiling by less than one maximum record span.

The 2B-parent budget is derived only after the final verified parent token counter exists; nominally it will be approximately 80M targets.

## Chat and loss contract

The existing GPT-2 vocabulary is unchanged. The byte-level chat template uses explicit `System:`, `User:`, and `Assistant:` text prefixes and GPT-2 token 50256 as the turn-termination token. The later accepted EOS decision is authoritative: each supervised assistant turn ends with an EOS target.

Training loss is:

```text
assistant content targets: active
assistant EOS targets:     active
system/user/role text:     masked
replay targets:            ordinary next-token targets
```

One conversation occupies one model sequence. Cross-conversation packing remains disabled. Overlength conversations drop oldest complete dialogue pairs first; an assistant response that still cannot fit without target truncation is rejected.

## Optimizer and execution contract

The current S0 trainer starts from verified native parent weights but creates fresh SFT optimizer/scheduler/scaler state. It does not resume pretraining optimizer moments.

Current operational defaults are:

```text
optimizer: hybrid Muon + AdamW
peak LR: 3e-5
weight decay: 0.0
schedule: one-pass WSD derived from immutable SFT block target counts
optimizer block target: approximately 32,768 loss-bearing targets
training microbatch: 4
precision/backend: CUDA FP16 autocast + qualified mixed FLA path
validation cadence: 250 optimizer updates
local checkpoint cadence: 250 optimizer updates
remote publication cadence: 250 optimizer updates
```

Variable-length SFT records are locally length-ordered/cropped at execution time so microbatch 4 does not transfer the full longest-in-block padding width to the GPU unnecessarily. The normalized loss remains the summed active-target cross-entropy divided by the optimizer block's exact active target count.

The WSD planner also supports tiny deterministic smoke datasets: it always reserves a non-empty final decay phase rather than overlapping the minimum warmup with decay.

## Checkpoint and resume identity

Every SFT checkpoint identity binds:

- the immutable parent checkpoint identity;
- the SFT bundle manifest identity;
- train split/shard identity and exact block cursor;
- model/trainer configuration;
- chat-template identity;
- masked-loss objective identity;
- optimizer/scheduler/scaler/RNG state.

Remote SFT publication uses a separate SFT run namespace. Automatic resume checks only that namespace and fails closed if parent/data/objective/configuration identities differ.

The canonical parent checkpoint namespaces are:

```text
500M parent: 20m-500m-dataset-001
2B parent:   20m-2b-dataset-001
```

Do not confuse those with the corresponding W&B run IDs (`20m-500m-data-001` and `20m-2b-data-001`).

## Comprehensive post-SFT qualification

ADR 0033 rejects both base-loss-only selection and SFT-loss-only selection. The post-SFT evaluator reports one scorecard with the immutable parent and SFT checkpoint side by side.

The current report includes:

- unchanged `eval_core_v1` intrinsic metrics: loss, perplexity, BPB, top-k accuracy, calibration, cluster/position slices, throughput and peak VRAM;
- the frozen base qualitative continuation/Q&A suite;
- held-out masked SFT validation loss/perplexity;
- deterministic instruction cases spanning direct QA, classification, transformations, extraction, exact formatting, explicit constraints, system-message adherence, multi-turn memory/correction, concise elementary reasoning, uncertainty and ordinary safe refusal;
- EOS termination, runaway rate, empty-response rate, role-label leakage, response-token length and trigram-repetition diagnostics;
- per-category and overall mechanically verifiable instruction pass rates;
- parent-versus-SFT metric deltas.

Exact-format cases are byte/line-structure sensitive after trimming outer whitespace; a required two-line answer does not pass as a one-line answer.

The report deliberately has no arbitrary single weighted master score. The 500M qualification trajectory is used to observe the real instruction-gain/base-retention tradeoff before choosing the 2B-parent checkpoint-selection policy.

## Current code surface

The reusable implementation lives under `post_training/sft/` and now includes:

- schema/template/filter/source adapters;
- deterministic data builder and immutable storage;
- pinned source preparation and identity-safe bundle creation;
- parent-checkpoint verification and SFT checkpoint identities;
- production SFT trainer and exact resume path;
- SFT checkpoint publication support;
- deterministic behavior evaluator;
- comprehensive parent-versus-SFT evaluator;
- optional base/SFT state-dict interpolation utility.

Installed CLI surfaces include:

```text
small-llm-sft-data
small-llm-sft-train
small-llm-sft-eval
```

The Kaggle launcher remains the preferred human-facing path.

## Remaining qualification gate

The implementation is operational in code but is not yet GPU-qualified evidence. Before the 500M SFT result is accepted, run:

1. the repository unit/integration suite;
2. deterministic bundle preparation and verification;
3. a bounded T4 FP16/mixed-FLA SFT smoke at microbatch 4;
4. an intentional interruption and exact resume proof;
5. 250-update validation/checkpoint/publication boundary checks;
6. the comprehensive parent-versus-SFT fast/full evaluator.

Only after those pass should the same lane be switched to the completed 2B parent.

## Related material

- SFT scaling decision: [`../decisions/0032-scale-sft-budget-with-pretraining-and-qualify-on-500m-first.md`](../decisions/0032-scale-sft-budget-with-pretraining-and-qualify-on-500m-first.md)
- SFT qualification/cadence decision: [`../decisions/0033-use-comprehensive-post-sft-qualification-and-pretraining-cadence.md`](../decisions/0033-use-comprehensive-post-sft-qualification-and-pretraining-cadence.md)
- SFT runbook: [`../runbooks/sft_s0_runbook.md`](../runbooks/sft_s0_runbook.md)
- Current roadmap: [`../current/roadmap.md`](../current/roadmap.md)
- General training contract: [`training_system.md`](training_system.md)
- Historical S0 design packet: [`../archive/post_training_s0_2026-08-06/README.md`](../archive/post_training_s0_2026-08-06/README.md)
