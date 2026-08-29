# 100M/2B 10% capacity-aware S0 dataset and training wiring — 2026-08-25

## Status

ADR 0122's 10% capacity-aware S0 dataset has now been built, verified, privately published to Kaggle, and accepted by ADR 0123 as the dataset for the next 100M/2B S0 training run.

The canonical experiment identity is:

```text
parent:          100m-2b-data-001
SFT/W&B run:     100m-2b-sft-s0-10pct-001
Kaggle slug:     small-llm-100m-2b-sft-s0-10pct-001
requested train: 200,100,044 targets (10% of 2,001,000,448)
realized train:  200,099,738 targets
```

The 306-target train shortfall is only complete-record packing and is accepted as the immutable training horizon. No padding or instruction repetition is introduced to close it.

## Capacity-aware build contract

The dedicated builder is:

```text
post_training/sft/s0_10pct_bundle.py
```

It enforces:

```text
parent targets:       2,001,000,448
10% train ceiling:      200,100,044
planned instruction:    170,085,037
planned replay:          30,015,007
```

Planned instruction-source targets:

```text
smol-magpie-ultra-short  160,707,411
smol-contraints            4,026,530
smollm-rewrite-30k         3,762,301
smol-summarize-20k         1,588,795
```

The train builder derives capacity-aware instruction weights from those counts, reads each instruction stream without replacement, keeps ClimbMix at 15%, and fails if realized per-source targets drift beyond packing tolerance. Build reports include explicit realized source target shares.

## Frozen held-outs

Validation and test are rebuilt deterministically at the historical 4% requested held-out horizon:

```text
2,106,316 requested targets per split
```

They retain the old `75/10/7.5/7.5` instruction-source weights and seed 17 and must match the completed 4% S0 identities:

```text
validation  26cb522729b4525498559d1ce131a181c30fd8fff573f3464e09030be803d09e
test        48e99ee51c201da398e227742ca7e023064a408c486cce16e20427d1ec7634d2
```

The completed publication reports `2,105,945` validation targets, 371 below its requested ceiling because of complete-record packing, with the exact frozen validation manifest above.

## Completed private Kaggle publication

The private publication/round-trip completed successfully with:

```text
file_count:   22
total_bytes:  773,987,135
tree_sha256:  c7550a377978231bfcc4d158ab11f8e2604e45921c5acb4e37e9557f12590b4d
status:       completed
```

Accepted split identities supplied by the completed publication:

```text
train:
  targets:             200,099,738
  manifest_sha256:     feefc3244bd8a2f369eec85e4a95410c2daf479016c04cf02c8042ca5a4010d3
  build_report_sha256: 8a131988c43349fb360f56dd41f7f552e9c1533c2550701db67b37ece6e820d7

validation:
  targets:             2,105,945
  manifest_sha256:     26cb522729b4525498559d1ce131a181c30fd8fff573f3464e09030be803d09e
  build_report_sha256: 37e3c4d98d1e7ed1ec077e0e92b7d79327c4ee2b473f0c4e86f0ec5e4d6c324d
```

The test manifest remains bound to the frozen identity above. The supplied publication excerpt did not include its build-report hash or realized token count, so this evidence does not invent those values.

## Training launcher wiring

`kaggle/sft_scaled_runtime.py` now binds the `100M/2B + --sft-fraction 10%` training path to the accepted published dataset identity before parent-model preflight, W&B initialization, or DDP launch.

For the 10% profile the launcher verifies:

- ADR-0122 recipe identity `s0-10pct-capacity-aware-v1`;
- requested train horizon `200,100,044`;
- exact accepted train target count, manifest hash, and build-report hash;
- exact accepted validation target count, manifest hash, and build-report hash;
- frozen test manifest hash.

A stale 10% rebuild, the completed 4% bundle, or any other SFT bundle therefore fails closed instead of silently starting the run.

The dataset resolver is profile-aware. On Kaggle, attaching the private `small-llm-100m-2b-sft-s0-10pct-001` dataset is sufficient when it is the unique SFT bundle input; an explicit `--dataset-dir` remains supported and receives the same identity checks.

Canonical training command:

```bash
python kaggle/launch_sft.py train \
  --model 100M \
  --tokens 2B \
  --sft-fraction 10%
```

The existing 100M/2B dual-T4 SFT wrapper already replaces the historical trainer's default 4% budget calculation with the requested profile fraction before invoking `train_cli`, so a `1/10` profile is accepted without changing the trainer objective. `train_cli` still verifies the complete bundle and constructs the WSD schedule from the realized immutable train block target counts.

Training mechanics remain intentionally unchanged from the qualified 100M/2B S0 path: two Tesla T4 GPUs, microbatch 2, LR `3e-5`, fresh SFT optimizer/scheduler/scaler state, exact global-token DDP objective, and 250-step durability/evaluation cadence.

## Verification state

Focused regression coverage now includes:

- `tests/test_sft_10pct_capacity_aware.py` for build/routing policy;
- `tests/test_sft_10pct_training_dataset.py` for the accepted publication identity, rejection of a wrong train manifest, and 10% train-command routing.

The project execution connector was unavailable during the training-wiring edit, so these tests could not be executed on the project VPS from this session. GitHub Actions continues to fail before exposing normal workflow steps/logs in the observed pushes, so no successful CI claim is made here.

## Relevant implementation commits

```text
f68157e9  Wire capacity-aware 10 percent S0 bundle
c27e39fd  Route 10 percent S0 through capacity-aware builder
fdfab079  Test capacity-aware 10 percent S0 routing
dedfc464  Pin 10 percent bundle build implementation
2162a46b  Bind 10 percent SFT training to verified dataset
dc253a07  Test verified 10 percent SFT training dataset binding
d1f0e735  Adopt verified 10 percent SFT bundle for training
```
