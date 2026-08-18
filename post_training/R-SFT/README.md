# R-SFT R0 data lane

This folder owns the first reasoning-SFT dataset path: Gemini generation, strict JSON parsing, the R0 skill/difficulty matrix, matched delimiter serialization, S0 instruction retention, the extended tokenizer, and immutable bundles consumed by the Kaggle trainer.

## Current frozen pilot

The committed corpus is:

```text
artifacts/rsft-r0-pilot-630/generation/reasoning.jsonl
```

It contains 630 Gemini-generated examples: 7 skills x 3 difficulty bands x 30 examples. The generation manifest beside it records the exact source identity. The 30 examples in each cell are deterministically partitioned as 28 train, 1 validation, and 1 test, so both held-out splits cover all 21 cells.

The reasoning control-token spellings are now frozen in `reasoning-tokens.json`:

```text
50257  <think>
50258  </think>
50259  <answer>
```

The atomic arm emits those three IDs exactly once per reasoning example. The matched textual arm uses ordinary GPT-2-tokenized `Reasoning:` / `Answer:` boundaries and does not emit IDs 50257-50259, while retaining the same promoted 50,260-vocabulary model geometry.

## Kaggle: canonical pilot training

For the first 100M/2B pilot, do not manually tokenize `reasoning.jsonl` and do not attach a separately built R-SFT bundle. The Kaggle launcher builds the matched native bundles automatically from the committed corpus.

Dry-run:

```bash
python kaggle/launch_r_sft.py train \
  --model 100M \
  --tokens 2B \
  --delimiter-format atomic \
  --dry-run
```

Atomic arm:

```bash
python kaggle/launch_r_sft.py train \
  --model 100M \
  --tokens 2B \
  --delimiter-format atomic
```

Textual comparison arm:

```bash
python kaggle/launch_r_sft.py train \
  --model 100M \
  --tokens 2B \
  --delimiter-format textual
```

The launcher automatically:

1. reads the committed 630-example corpus and frozen token spec from the pinned git worktree;
2. resolves the completed 100M/2B S0 dataset bundle for the 10% retention lane, preferring an already attached Kaggle input and otherwise downloading private dataset `roccoangelella/small-llm-100m-2b-sft-s0-001`;
3. samples only S0 instruction records, preserving the S0 instruction-source stratification and excluding ClimbMix replay;
4. materializes and verifies both matched native bundles under `/kaggle/working`;
5. selects the requested arm and launches the qualified 2xTesla-T4 DDP trainer;
6. promotes the S0 model vocabulary from 50,257 to 50,260 semantic rows, initializing only the three new control rows;
7. trains exactly one pass over the frozen arm bundle with checkpointing, W&B and verified remote publication inherited from S0.

For this small delimiter ablation only, automatic Kaggle preparation uses 2,048 loss-bearing target tokens per optimizer block. The generic R-SFT bundle builder retains its 32,768-target default for larger runs. Both ablation arms use the same 2,048-target geometry.

Canonical arm run IDs are generated automatically:

```text
atomic   100m-2b-rsft-r0-atomic-pilot-001
textual  100m-2b-rsft-r0-textual-pilot-001
```

Manual `--dataset-dir`, `--s0-bundle`, `--token-spec`, and `--run-id` remain available only as explicit overrides.

### Kaggle credentials

Live training uses the same post-training credentials as S0:

```text
GITHUB_TOKEN
WANDB_API_KEY
HF_TOKEN
SMALL_LLM_SFT_HF_REPO_ID
```

`SMALL_LLM_SFT_HF_REPO_ID` must contain the completed `100m-2b-sft-s0-001` parent checkpoint. Set `SMALL_LLM_RSFT_HF_REPO_ID` if R-SFT checkpoints should go to a separate repository; otherwise the existing post-training repository fallback is used. `WANDB_ENTITY` is optional.

If the private S0 dataset is not attached to the notebook, automatic Kaggle download may require the notebook's Kaggle credentials. `SMALL_LLM_S0_KAGGLE_DATASET_HANDLE` can override the canonical private dataset handle.

## Generic data production

The lower-level builder remains available outside the Kaggle pilot launcher:

```bash
python post_training/R-SFT/produce.py build \
  --reasoning-jsonl artifacts/rsft-r0-pilot-630/generation/reasoning.jsonl \
  --s0-bundle /path/to/100m-2b-sft-s0-bundle \
  --token-spec post_training/R-SFT/reasoning-tokens.json \
  --output-dir artifacts/rsft-r0-pilot-630/bundles
```

It produces a shared pilot manifest plus `atomic/` and `textual/` native SFT bundles, each with `bundle-manifest.json`, `reasoning-tokens.json`, and train/validation/test shards. Verify them with:

```bash
python post_training/R-SFT/produce.py verify \
  --dataset-dir artifacts/rsft-r0-pilot-630/bundles
```

The retention sample is identical across the two arms. Since textual delimiters occupy a different number of target tokens from the three atomic delimiters, the builder uses one symmetric retention target derived from the mean reasoning-token totals and records each arm's realized retention share in `pilot-manifest.json`.
