---
status: accepted
date: 2026-08-25
---

# ADR 0123: train the next 100M/2B S0 run on the verified 10% Kaggle bundle

## Decision

Use the completed, privately published ADR-0122 10% capacity-aware S0 bundle as the training dataset for the next 100M/2B supervised-fine-tuning trial.

The experiment identity remains:

```text
SFT run / W&B run: 100m-2b-sft-s0-10pct-001
Kaggle dataset slug: small-llm-100m-2b-sft-s0-10pct-001
parent: 100m-2b-data-001
parent consumed targets: 2,001,000,448
requested SFT horizon: 200,100,044 targets (10%)
```

The published bundle realized `200,099,738` train loss-bearing targets, only 306 targets below the requested ceiling because of complete-record packing. This is accepted as the immutable train horizon for the run; do not pad or repeat examples to close the 306-target remainder.

## Accepted private publication identity

The successful private Kaggle publication and round-trip reported:

```text
file_count:   22
total_bytes:  773,987,135
tree_sha256:  c7550a377978231bfcc4d158ab11f8e2604e45921c5acb4e37e9557f12590b4d
status:       completed
```

Bind the 10% training launcher to the following split identities:

```text
train:
  loss_bearing_target_tokens: 200,099,738
  manifest_sha256:            feefc3244bd8a2f369eec85e4a95410c2daf479016c04cf02c8042ca5a4010d3
  build_report_sha256:        8a131988c43349fb360f56dd41f7f552e9c1533c2550701db67b37ece6e820d7

validation:
  loss_bearing_target_tokens: 2,105,945
  manifest_sha256:            26cb522729b4525498559d1ce131a181c30fd8fff573f3464e09030be803d09e
  build_report_sha256:        37e3c4d98d1e7ed1ec077e0e92b7d79327c4ee2b473f0c4e86f0ec5e4d6c324d

test:
  manifest_sha256:            48e99ee51c201da398e227742ca7e023064a408c486cce16e20427d1ec7634d2
```

The test split manifest is the already-frozen completed-4% S0 identity required by ADR 0122. The supplied publication excerpt did not include the test build-report hash or realized target count, so those are not newly asserted here; the immutable test manifest identity remains the training/evaluation gate.

The validation split realized 371 targets below its `2,106,316` requested frozen horizon because of complete-record packing. That does not change its identity or invalidate longitudinal comparison.

## Training launcher contract

The canonical launcher remains:

```bash
python kaggle/launch_sft.py train \
  --model 100M \
  --tokens 2B \
  --sft-fraction 10%
```

On Kaggle, attach the private `small-llm-100m-2b-sft-s0-10pct-001` dataset as an input. The launcher may discover the attached bundle automatically, but before parent-model resolution, W&B initialization, or DDP launch it must verify that the bundle carries the ADR-0122 recipe and the exact accepted split identities above. An explicitly supplied `--dataset-dir` is subject to the same verification.

Do not silently fall back to the completed 4% S0 bundle or to another 10% rebuild with different split/build identities.

## Training mechanics

Keep the already-qualified 100M/2B SFT training mechanics unchanged so this remains primarily a data-scale experiment:

```text
2x Tesla T4 DDP on Kaggle
microbatch size: 2 per existing profile
learning rate: 3e-5
fresh SFT optimizer/scheduler/scaler state from the completed pretrained parent
one immutable optimizer update per SFT block
assistant-only instruction loss plus full-loss ClimbMix replay
existing WSD schedule derived from the realized immutable block stream
250-step validation/checkpoint/remote-publication cadence
```

The finite schedule must derive from the bundle's realized train block counts/target counts, not from an assumption that exactly 200,100,044 targets were packed.

## Qualification

This remains an experimental S0 scaling run. After completion, compare it directly with the pretrained parent and completed 4% S0 checkpoint on the frozen SFT held-outs, instruction behavior, EOS/runaway/repetition behavior, unchanged `eval_core_v1`, and subsequent reasoning/generalization probes. Do not promote the run based on SFT validation loss alone.
