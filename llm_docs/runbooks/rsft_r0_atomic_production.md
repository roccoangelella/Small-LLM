# R-SFT R0 atomic runbook

_Last updated: 2026-08-18 Europe/Rome_

The accepted 100M / 2B R0 R-SFT artifact is the already-completed atomic delimiter run:

```text
100m-2b-rsft-r0-atomic-pilot-001
```

It uses the frozen special-token protocol:

```text
50257  <think>
50258  </think>
50259  <answer>
```

The textual delimiter arm remains ablation-only. ADR 0100 promotes the completed atomic artifact itself; do **not** retrain the same 630-example corpus only to obtain a different run ID.

## 1. Current accepted R0 state

The frozen reasoning corpus already exists at:

```text
artifacts/rsft-r0-pilot-630/generation/reasoning.jsonl
```

The accepted atomic run already trained that corpus together with the frozen 10% S0 instruction-retention lane for one complete pass. The checkpoint remains identified by its original scientific provenance, including the historical `pilot` suffix.

## 2. Chat with the accepted artifact

Configure Hugging Face access. R-SFT first checks `SMALL_LLM_RSFT_HF_REPO_ID`, then falls back to the SFT/base checkpoint repository variables.

```bash
export HF_TOKEN=...
export SMALL_LLM_RSFT_HF_REPO_ID=owner/rsft-repository  # optional if shared
```

Run:

```bash
python chat.py --model_params 100M --num_tokens 2B --r-sft
```

`chat.py` resolves `100m-2b-rsft-r0-atomic-pilot-001` and fails closed unless the downloaded checkpoint is complete and carries:

```text
semantic_vocab_size = 50260
pipeline_state.rsft_format.version = 1
pipeline_state.rsft_format.stage = r_sft_r0
pipeline_state.rsft_format.delimiter_format = atomic
```

plus the exact `<think>`, `</think>`, `<answer>` token metadata at IDs 50257-50259.

## 3. Historical delimiter experiment

The original matched experiment remains reproducible for audit purposes:

```bash
python kaggle/launch_r_sft.py ablation --model 100M --tokens 2B --delimiter-format atomic
python kaggle/launch_r_sft.py ablation --model 100M --tokens 2B --delimiter-format textual
```

Do not use the textual checkpoint as an accepted R-SFT artifact or as the parent for later reasoning stages.

## 4. Future larger R-SFT corpus

`post_training/R-SFT/build_atomic.py` and the production Kaggle `train` path remain available for a **future explicitly approved larger/new reasoning corpus**. They are not instructions to duplicate the accepted 630-example R0 run.

If a future corpus is frozen, build an atomic-only native bundle on a CPU/VPS host with the completed S0 bundle available:

```bash
python post_training/R-SFT/build_atomic.py \
  --reasoning-jsonl /path/to/new-reasoning.jsonl \
  --s0-bundle /path/to/100m-2b-sft-s0-bundle \
  --output-dir /path/to/rsft-r0-expanded \
  --heldout-per-cell <FROZEN_COUNT>
```

The builder defaults to 32,768 loss-bearing target tokens per optimizer block, uses the canonical atomic token spec, computes the 10% retention target from atomic reasoning targets, preserves the S0 instruction-source stratification, excludes ClimbMix replay, and verifies the resulting bundle.

Only after a new corpus/run is separately approved should it be launched with the production `train` path. Until then, `100m-2b-rsft-r0-atomic-pilot-001` is the accepted R0 checkpoint.
