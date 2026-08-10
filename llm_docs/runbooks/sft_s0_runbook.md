# SFT S0 qualification runbook

_Last updated: 2026-08-10 Europe/Rome_

This runbook is the canonical human procedure for the first operational supervised-fine-tuning lane. The completed 20M/500M checkpoint is the qualification parent. As soon as the fresh 20M/2B checkpoint is complete and post-pretraining-qualified, use the same lane with the 2B profile.

## 1. Canonical launcher

Use only:

```bash
python kaggle/launch_sft.py <action> --model 20M --tokens <500M|2B>
```

Do not create per-run SFT wrapper scripts.

Inspect the registered profiles first:

```bash
python kaggle/launch_sft.py profiles
python kaggle/launch_sft.py train --model 20M --tokens 500M --dry-run
```

The 500M parent checkpoint namespace is `20m-500m-dataset-001`. The 2B parent namespace is `20m-2b-dataset-001`. These are checkpoint/dataset run IDs, not W&B run IDs.

## 2. Required Kaggle environment

Use an NVIDIA T4 notebook with Internet enabled.

Required secrets/environment for live SFT training:

```text
GITHUB_TOKEN
WANDB_API_KEY
HF_TOKEN
SMALL_LLM_HF_REPO_ID
```

Recommended separate private repository for SFT checkpoints:

```text
SMALL_LLM_SFT_HF_REPO_ID
```

Optional:

```text
WANDB_ENTITY
```

Attach the exact verified pretraining dataset that will supply the 15% replay baseline. For the first qualification this is the accepted private 500M dataset. Do not attach multiple ambiguous replay datasets.

## 3. Prepare the 500M qualification bundle

The completed parent consumed exactly:

```text
500,156,416 pretraining target tokens
```

The 4% requested SFT ceiling is therefore:

```text
20,006,256 loss-bearing SFT targets
```

From the current repository clone:

```bash
cd /kaggle/working/Small-LLM
git switch main
git pull --ff-only

python kaggle/launch_sft.py prepare \
  --model 20M \
  --tokens 500M \
  --replay-root /kaggle/input/<exact-500m-pretraining-dataset>
```

Default outputs are:

```text
prepared pinned instruction source:
/kaggle/working/small-llm-sft-smoltalk-pinned

immutable 500M-parent SFT bundle:
/kaggle/working/small-llm-20m-500m-sft-bundle
```

The build pins the instruction source revision, applies deterministic identity splitting before tokenization, removes exact duplicates and direct deterministic-suite contamination, filters the S0 scope, mixes by active target tokens, writes immutable SFT shards, and verifies the completed bundle.

If the bundle is preserved as a private Kaggle dataset for a later session, attach exactly that verified version and pass its root with `--dataset-dir`.

## 4. First bounded T4 qualification

Before the full 500M-parent SFT trajectory, run a bounded session:

```bash
python kaggle/launch_sft.py train \
  --model 20M \
  --tokens 500M \
  --dataset-dir /kaggle/working/small-llm-20m-500m-sft-bundle \
  --max-steps-this-session 20
```

Acceptance checks:

```text
CUDA FP16/mixed-FLA path is active
training microbatch = 4
finite forward loss and gradients
no unresolved FP16 overflow failure
active-target loss normalization is finite
VRAM is stable
W&B telemetry appears under the SFT run identity
final bounded checkpoint is published and verified
```

This 20-update bounded session is a runtime smoke, not a scientific checkpoint.

## 5. Intentional resume proof

Rerun the identical command with a larger bounded endpoint, for example:

```bash
python kaggle/launch_sft.py train \
  --model 20M \
  --tokens 500M \
  --dataset-dir /kaggle/working/small-llm-20m-500m-sft-bundle \
  --max-steps-this-session 40
```

Automatic resume must restore the verified SFT `latest` checkpoint and continue from the exact next immutable SFT block. Parent checkpoint, bundle manifest, template, objective, optimizer configuration, scheduler state, scaler state, RNG state, and block cursor must agree or resume fails closed.

For the formal exact-resume gate, compare an uninterrupted deterministic fixture with an interrupted/restored fixture and require identical next-block identity plus numerically identical CPU/FP32 update state where the existing trainer tests support exact comparison.

## 6. Full 500M-parent qualification run

After the bounded T4 and resume gates pass:

```bash
python kaggle/launch_sft.py train \
  --model 20M \
  --tokens 500M \
  --dataset-dir /kaggle/working/small-llm-20m-500m-sft-bundle
```

Current operational defaults:

```text
optimizer: hybrid Muon + AdamW
peak LR: 3e-5
weight decay: 0.0
one-pass WSD schedule
optimizer block target: about 32,768 loss-bearing targets
microbatch: 4
validation: every 250 updates
local checkpoint: every 250 updates
verified remote publication: every 250 updates
```

The current qualification baseline uses 85% instruction / 15% original-distribution replay. Treat the replay ratio as an experimental baseline to evaluate on the 500M trajectory rather than a universal future constant.

## 7. Comprehensive post-SFT evaluation

Run the fast report first:

```bash
python kaggle/launch_sft.py eval \
  --model 20M \
  --tokens 500M \
  --dataset-dir /kaggle/working/small-llm-20m-500m-sft-bundle \
  --suite fast
```

Then the full report:

```bash
python kaggle/launch_sft.py eval \
  --model 20M \
  --tokens 500M \
  --dataset-dir /kaggle/working/small-llm-20m-500m-sft-bundle \
  --suite full
```

The report compares the immutable parent and SFT checkpoint on one scorecard rather than one weighted scalar. Inspect at least:

```text
base eval_core_v1 loss / perplexity / BPB
cluster and position deltas
top-k accuracy and calibration deltas
base qualitative continuations
held-out masked SFT validation loss
instruction pass rate and per-category rates
EOS termination and runaway rate
empty-response and role-label leakage rate
response length and repetition diagnostics
```

Do not select the SFT checkpoint from SFT validation loss alone and do not select it from parent-style pretraining loss alone.

## 8. Switch to the 2B parent

When the fresh 2B pretraining run is complete and its final/best checkpoint is accepted, obtain the verified exact parent consumed-target counter and build the new bundle:

```bash
python kaggle/launch_sft.py prepare \
  --model 20M \
  --tokens 2B \
  --parent-consumed-tokens <verified-final-parent-target-count> \
  --replay-root /kaggle/input/<exact-2b-pretraining-dataset>
```

Then use the same `train` and `eval` commands with `--tokens 2B`.

The nominal SFT horizon is approximately 80M loss-bearing targets, but the immutable manifest must use the exact verified completed parent counter rather than the nominal `2B` label.

## 9. Evidence required before calling the SFT lane qualified

- repository SFT tests pass;
- prepared-source and bundle manifests verify;
- T4 microbatch-4 smoke is finite;
- intentional exact resume passes;
- 250-update durability/validation/publication boundaries behave correctly;
- fast and full comprehensive reports are produced for parent and SFT checkpoints;
- no unresolved identity mismatch, non-finite event, runaway-generation regression, or severe base-capability collapse remains unexplained.

Related decisions: ADR 0032 and ADR 0033.
