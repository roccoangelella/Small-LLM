# R-SFT R0 atomic production runbook

_Last updated: 2026-08-18 Europe/Rome_

Production R-SFT uses only the frozen special-token protocol:

```text
50257  <think>
50258  </think>
50259  <answer>
```

The historical textual delimiter arm is ablation-only.

## 1. Freeze the production reasoning corpus

The production corpus size and held-out-per-cell allocation are separate scientific decisions and must be frozen before bundle construction. Keep the already-accepted uniform 7 skills x 3 difficulty-cell record distribution and one-pass training policy.

## 2. Build the atomic production bundle

On a CPU/VPS host with the completed S0 dataset bundle available:

```bash
python post_training/R-SFT/build_atomic.py \
  --reasoning-jsonl /path/to/production-reasoning.jsonl \
  --s0-bundle /path/to/100m-2b-sft-s0-bundle \
  --output-dir /path/to/rsft-r0-production \
  --heldout-per-cell <FROZEN_COUNT>
```

The builder defaults to the production 32,768-target optimizer block, uses the exact canonical token spec in `post_training/R-SFT/reasoning-tokens.json`, computes the 10% retention target from atomic reasoning targets, preserves S0 instruction-source stratification, excludes ClimbMix replay, and verifies the resulting native bundle.

The bundle must declare:

```text
rsft.stage = r_sft_r0
rsft.delimiter_format = atomic
rsft.contract = atomic-production-v1
optimizer_target_tokens = 32768
```

## 3. Attach/publish the frozen bundle to Kaggle

Make the resulting bundle visible in the Kaggle notebook, for example under:

```text
/kaggle/input/rsft-r0-production
```

Do not train directly from raw `reasoning.jsonl`.

## 4. Dry-run the production launcher

```bash
python kaggle/launch_r_sft.py train \
  --model 100M \
  --tokens 2B \
  --dataset-dir /kaggle/input/rsft-r0-production \
  --dry-run
```

The dry-run must report:

```text
contract = atomic-production-v1
delimiter_format = atomic
run_id = 100m-2b-rsft-r0-001
topology = 2xTesla-T4-DDP
budget_mode = bundle-exact-one-pass
```

## 5. Launch on 2xT4

```bash
python kaggle/launch_r_sft.py train \
  --model 100M \
  --tokens 2B \
  --dataset-dir /kaggle/input/rsft-r0-production
```

Required live-training credentials:

```text
GITHUB_TOKEN
WANDB_API_KEY
HF_TOKEN
SMALL_LLM_SFT_HF_REPO_ID
```

Optional separate R-SFT checkpoint repository:

```text
SMALL_LLM_RSFT_HF_REPO_ID
```

The launcher uses the completed `100m-2b-sft-s0-001` parent, promotes only semantic rows 50257:50260, keeps the remaining padded rows zero, and reuses the qualified dual-T4 SFT engine for exact one-pass WSD training, W&B, checkpoints, resume, and remote publication.

## 6. Historical delimiter pilot

Only for reproducing the completed experiment:

```bash
python kaggle/launch_r_sft.py ablation --model 100M --tokens 2B --delimiter-format atomic
python kaggle/launch_r_sft.py ablation --model 100M --tokens 2B --delimiter-format textual
```

Do not use the textual pilot artifact as a production parent.
