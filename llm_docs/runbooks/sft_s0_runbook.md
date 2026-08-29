# SFT S0 qualification runbook

_Last updated: 2026-08-13 Europe/Rome_

This runbook is the canonical human procedure for the first operational supervised-fine-tuning lane. The completed 20M/500M checkpoint is the qualification parent. As soon as the fresh 20M/2B checkpoint is complete and post-pretraining-qualified, use the same lane with the 2B profile.

ADR 0034 makes the SFT data lane machine-agnostic: prefer the VPS for `prepare`/`publish` when the verified replay dataset already lives there, and reserve Kaggle accelerator sessions primarily for `train`/`eval`.

## 1. Canonical launcher

Use only:

```bash
python kaggle/launch_sft.py <action> --model 20M --tokens <500M|2B>
```

The supported actions are:

```text
prepare
publish
train
eval
profiles
```

Do not create per-run SFT wrapper scripts.

Inspect the registered profiles first:

```bash
python kaggle/launch_sft.py profiles
python kaggle/launch_sft.py train --model 20M --tokens 500M --dry-run
```

The 500M parent checkpoint namespace is `20m-500m-dataset-001`. The 2B parent namespace is `20m-2b-dataset-001`. These are checkpoint/dataset run IDs, not W&B run IDs.

## 2. Environment split and credentials

### VPS or ordinary Linux host — preferred for prepare/publish

`prepare` and `publish` do not require a GPU or a Kaggle filesystem. Run them from a normal repository clone when the verified pretraining replay dataset is already local.

The runtime chooses its work root in this order:

```text
SMALL_LLM_WORK_DIR if explicitly set
/kaggle/working if that directory actually exists
<repository-parent>/small-llm-work otherwise
```

For the usual VPS clone at `~/Projects/Small-LLM`, the no-override default is therefore:

```text
~/Projects/small-llm-work
```

If `/data` is the desired large persistent volume, set for example:

```bash
export SMALL_LLM_WORK_DIR=/data/small-llm/sft-work
```

Private Kaggle publication requires:

```text
KAGGLE_API_TOKEN
KAGGLE_USERNAME
```

Instead of `KAGGLE_USERNAME`, an explicit `--kaggle-dataset-handle owner/dataset` or `SMALL_LLM_SFT_KAGGLE_DATASET_HANDLE` may provide the destination identity.

Source preparation also needs ordinary Internet access for the pinned instruction dataset/package resolution.

### Kaggle T4 — preferred for train/eval

Use an NVIDIA T4 notebook with Internet enabled for the actual SFT training and GPU qualification path.

The launcher pins the already-qualified dual-T4 subprocess stack: PyTorch 2.10.0
from the CUDA 12.8 wheel index, Triton 3.6.0, and `fla-core==0.5.2`. The
runtime guard must fail closed if Kaggle or package resolution drifts from those
versions. If it fires, update the launcher and start a fresh session; do not
delete or loosen the guard to get a run moving.

Required secrets/environment for live SFT training:

```text
GITHUB_TOKEN
WANDB_API_KEY
HF_TOKEN
SMALL_LLM_HF_REPO_ID
```

`SMALL_LLM_HF_REPO_ID` must name the repository containing the selected
parent's stable `models/<parent-run-id>/...` artifact. It is not a generic
pointer to whichever Small-LLM repository was used most recently. The launcher
checks this on CPU before W&B setup or dual-GPU dispatch.

Recommended separate private repository for SFT checkpoints:

```text
SMALL_LLM_SFT_HF_REPO_ID
```

Optional:

```text
WANDB_ENTITY
```

When `train` or `eval` is given no `--dataset-dir`, the launcher keeps the Kaggle convenience of discovering exactly one SFT bundle under `/kaggle/input`. On another machine either pass `--dataset-dir` explicitly or set `SMALL_LLM_INPUT_DIR` to an alternative implicit input root.

For bundle construction, `--replay-root` always means the exact verified pretraining **dataset directory containing `manifest.json`**, never the `manifest.json` file itself.

## 3. Build and privately publish the 500M qualification bundle from the VPS

The completed parent consumed exactly:

```text
500,156,416 pretraining target tokens
```

The 4% requested SFT ceiling is therefore:

```text
20,006,256 loss-bearing SFT targets
```

On the current VPS layout, use:

```bash
cd ~/Projects/Small-LLM
git switch main
git pull --ff-only

python kaggle/launch_sft.py publish \
  --model 20M \
  --tokens 500M \
  --replay-root /data/small-llm/20m-500m-ops/kaggle-dataset
```

The replay path above is the directory that contains `manifest.json`. Do **not** append `/manifest.json`.

If desired, put SFT work products on the `/data` volume before launching:

```bash
export SMALL_LLM_WORK_DIR=/data/small-llm/sft-work
```

`publish` performs the full durable bundle path:

```text
pinned SmolTalk preparation
  -> global identity-safe 95/2.5/2.5 split
  -> immutable 4%-scaled SFT train/validation/test bundle
  -> full local bundle verification
  -> private Kaggle upload
  -> fresh Kaggle round-trip download
  -> complete tree SHA-256 comparison
  -> full bundle re-verification
  -> anonymous-access denial check
```

Default local outputs are rooted under the selected work root. For example, with `SMALL_LLM_WORK_DIR=/data/small-llm/sft-work`:

```text
prepared pinned instruction source:
/data/small-llm/sft-work/small-llm-sft-smoltalk-pinned

immutable 500M-parent SFT bundle:
/data/small-llm/sft-work/small-llm-20m-500m-sft-bundle

publication state/round-trip work:
/data/small-llm/sft-work/small-llm-20m-500m-sft/bundle-publication
```

On Kaggle the same launcher automatically retains `/kaggle/working` as its work root when that directory exists.

If you only need to construct/inspect the bundle without publishing it, use the same command with action `prepare`.

The build is idempotent for an already complete matching bundle: it verifies the existing bytes instead of replacing them. A bundle whose requested target horizon does not match the exact parent-derived 4% budget fails closed.

## 4. Start a fresh training session from the published bundle

After `publish` succeeds, start a new Kaggle session and attach exactly the verified private SFT dataset version. Normally no `--dataset-dir` is needed when exactly one bundle is attached under `/kaggle/input`:

```bash
python kaggle/launch_sft.py train \
  --model 20M \
  --tokens 500M \
  --max-steps-this-session 20
```

If the notebook contains other bundle-like inputs, identify the exact attached root explicitly:

```bash
python kaggle/launch_sft.py train \
  --model 20M \
  --tokens 500M \
  --dataset-dir /kaggle/input/private-sft-dataset/bundle-root \
  --max-steps-this-session 20
```

Acceptance checks for this bounded runtime smoke:

```text
correct parent checkpoint namespace resolves
attached bundle fully verifies
bundle requested budget equals exactly 4% of the verified parent counter
CUDA FP16/mixed-FLA path is active
training microbatch = 4
finite forward loss and gradients
no unresolved FP16 overflow failure
active-target loss normalization is finite
VRAM is stable
W&B telemetry appears under the SFT run identity
bounded final checkpoint is locally saved and remotely published
```

This 20-update bounded session is a runtime smoke, not a scientific checkpoint.

## 5. Intentional resume proof

`--max-steps-this-session` means **additional optimizer updates in that invocation**, not an absolute global endpoint.

Therefore rerun the exact same bounded command:

```bash
python kaggle/launch_sft.py train \
  --model 20M \
  --tokens 500M \
  --max-steps-this-session 20
```

The second invocation must restore the first invocation's verified checkpoint and execute the next 20 immutable optimizer blocks.

Automatic resume examines both:

```text
verified local step-* checkpoints
verified remote run/<sft-run-id>/latest.json
```

It validates the immutable parent, bundle, template/objective, trainer configuration, scheduler/scaler/RNG state, and exact block cursor, then chooses the newest valid boundary. This deliberately preserves a newer local save if remote publication was interrupted. In a fresh Kaggle session only the remote checkpoint remains, so it becomes the recovery source.

W&B must use strict resume after a training checkpoint has actually been restored.

For the formal deterministic exact-resume gate, compare an uninterrupted CPU/FP32 fixture with an interrupted/restored fixture and require identical next-block identity plus numerically identical update state under the existing exact trainer tests.

## 6. Full 500M-parent qualification run

After the bounded T4 and resume gates pass, rerun without a session limit:

```bash
python kaggle/launch_sft.py train \
  --model 20M \
  --tokens 500M
```

Current frozen operational defaults:

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
remote retention: latest verified checkpoint only; prune the same run and super-squash after publication
```

The controlled S0 mixture is:

```text
85% filtered instruction targets
15% original-distribution replay targets
```

Within instruction data:

```text
75.0% smol-magpie-ultra-short
10.0% smol-contraints
 7.5% smollm-rewrite-30k
 7.5% smol-summarize-20k
```

Do not change those mixture values between the 500M qualification and the first 2B-parent comparison without a new recorded decision.

## 7. Comprehensive post-SFT evaluation

Run the fast report first:

```bash
python kaggle/launch_sft.py eval \
  --model 20M \
  --tokens 500M \
  --suite fast
```

Then the full report:

```bash
python kaggle/launch_sft.py eval \
  --model 20M \
  --tokens 500M \
  --suite full
```

The full report compares the immutable parent and SFT checkpoint on one scorecard rather than one weighted scalar. Inspect at least:

```text
base eval_core_v1 loss / perplexity / BPB
top-1 / top-5 / top-10 accuracy deltas
calibration ECE delta
per-cluster loss/perplexity deltas
position-bucket loss deltas
base qualitative continuations
held-out masked SFT validation loss
held-out masked SFT test loss
instruction pass rate and per-category rates
EOS termination and runaway rate
empty-response and role-label leakage rate
response length and trigram-repetition diagnostics
```

Do not select the SFT checkpoint from SFT loss alone and do not select it from parent-style pretraining loss alone. The 500M trajectory is the evidence used to decide the later numerical selection/retention gates.

## 8. Switch to the 2B parent

When the fresh 2B pretraining run is complete and its final/best checkpoint is accepted, obtain the verified exact parent consumed-target counter and publish the new SFT bundle from whichever ordinary machine already holds the verified 2B replay dataset, normally the VPS:

```bash
python kaggle/launch_sft.py publish \
  --model 20M \
  --tokens 2B \
  --parent-consumed-tokens 2000000000 \
  --replay-root /path/to/verified-2b-pretraining-dataset
```

The `2000000000` value above is only an example of command shape. Replace it with the verified final parent consumed-target counter; do not use the nominal label if the completed counter differs.

Then attach the published SFT bundle on Kaggle and use the same `train` and `eval` commands with `--tokens 2B`.

For the current 100M/2B parent, the stable artifact is in
`roccoangelella/small-llm-100m-qualification`, not the historical 20M
qualification repository. Set the Kaggle secret accordingly, or make the
parent repository explicit:

```bash
python kaggle/launch_sft.py train \
  --model 100M \
  --tokens 2B \
  --parent-repo-id roccoangelella/small-llm-100m-qualification
```

The live 100M backward prewarm exceeded one T4 at per-rank microbatch 4 before
any optimizer step. The 100M/2B profile therefore uses per-rank execution
microbatch 2. This changes only the number of gradient-accumulation slices;
the global SFT block, full-block target normalization, DDP-average
compensation, clipping boundary, and single optimizer update are unchanged.

Microbatch 2 subsequently passed prewarm and completed 250 finite optimizer
updates. The first cadence boundary exposed a separate synchronization bug:
rank 1 entered the next NCCL barrier while rank 0 was still running the real
validation and behavior suites, then hit NCCL's 600-second watchdog. No
step-250 checkpoint was saved, so that attempt is not resumable. The pinned
runtime now waits for rank-zero cadence work through a one-hour CPU/Gloo
control group and returns both ranks to NCCL together. Restart the same command
from the parent; automatic resume must report no SFT checkpoint, and the next
acceptance boundary is successful step-250 evaluation, local save, and verified
remote publication.

The nominal SFT horizon is approximately 80M loss-bearing targets, but the immutable manifest and trainer gate use the exact verified completed parent counter rather than the nominal `2B` label.

## 9. Evidence required before calling the SFT lane qualified

- repository SFT tests pass;
- VPS/ordinary-host 500M `publish` path succeeds without any root-level `/kaggle` filesystem dependency;
- prepared-source and bundle identities verify;
- private Kaggle bundle publication and byte-identical round trip pass;
- anonymous bundle access is denied;
- selected per-profile T4 microbatch smoke is finite (20M: 4; 100M: 2);
- intentional exact local/remote resume passes;
- 250-update durability/validation/publication boundaries behave correctly;
- fast and full comprehensive reports are produced for parent and SFT checkpoints;
- no unresolved identity mismatch, non-finite event, runaway-generation regression, or severe base-capability collapse remains unexplained.

Related decisions: ADR 0032, ADR 0033, and ADR 0034.
