# R-SFT R0 data lane

This folder owns reasoning-SFT dataset production, the extended GPT-2 tokenizer,
S0 instruction retention, immutable bundles, and the historical delimiter
ablation.

## Production token contract

Production R-SFT is atomic-only:

```text
50257  <think>
50258  </think>
50259  <answer>
```

These are control symbols, not ordinary natural-language delimiters. The
historical textual `Reasoning:` / `Answer:` arm remains reproducible only as
ablation evidence and is not a production serialization option.

## Build a production atomic bundle

Once a production reasoning JSONL is frozen, build exactly one native atomic
bundle with:

```bash
python post_training/R-SFT/build_atomic.py \
  --reasoning-jsonl /path/to/reasoning.jsonl \
  --s0-bundle /path/to/100m-2b-sft-s0-bundle \
  --output-dir /path/to/rsft-r0-production \
  --heldout-per-cell <N>
```

The builder:

- infers and verifies a uniform record count across all 7 x 3 R0 cells;
- uses the frozen `<think>`, `</think>`, `<answer>` special-token mapping only;
- deterministically partitions every cell into train/validation/test;
- computes the 10% S0 retention target from the atomic reasoning-token count;
- samples exact tokenized S0 instruction records while preserving the S0
  instruction-source stratification and excluding ClimbMix replay;
- defaults to 32,768 loss-bearing target tokens per optimizer block;
- writes and verifies a native bundle marked `rsft.contract=atomic-production-v1`.

`--heldout-per-cell` is deliberately required because production held-out scale
has not yet been frozen. The builder refuses to infer that choice from the small
630-example pilot.

## Kaggle production training

Production training uses:

```bash
python kaggle/launch_r_sft.py train \
  --model 100M \
  --tokens 2B \
  --dataset-dir /kaggle/input/rsft-r0-production
```

The default production run ID is:

```text
100m-2b-rsft-r0-001
```

The launcher fails closed unless the bundle is `atomic-production-v1`, uses the
exact frozen token metadata, and uses the 32,768-target production optimizer
geometry. The run remains one exact pass over the frozen bundle and reuses the
qualified 2xTesla-T4 DDP SFT engine, S0 checkpoint loading, row promotion,
checkpointing, W&B, exact resume and remote publication.

Required live-training credentials are the same as S0:

```text
GITHUB_TOKEN
WANDB_API_KEY
HF_TOKEN
SMALL_LLM_SFT_HF_REPO_ID
```

Set `SMALL_LLM_RSFT_HF_REPO_ID` to publish R-SFT checkpoints to a separate
repository; otherwise the existing post-training repository fallback applies.

## Historical 630-example delimiter ablation

The frozen pilot corpus is:

```text
artifacts/rsft-r0-pilot-630/generation/reasoning.jsonl
```

It contains 630 Gemini examples (7 skills x 3 difficulty bands x 30 examples).
The pilot used 28 train + 1 validation + 1 test example per cell and a special
2,048-target optimizer geometry so that the tiny comparison had enough updates.

Reproduce either historical arm with:

```bash
python kaggle/launch_r_sft.py ablation \
  --model 100M --tokens 2B --delimiter-format atomic

python kaggle/launch_r_sft.py ablation \
  --model 100M --tokens 2B --delimiter-format textual
```

Canonical historical run IDs are:

```text
atomic   100m-2b-rsft-r0-atomic-pilot-001
textual  100m-2b-rsft-r0-textual-pilot-001
```

The two completed pilot summaries used the same S0 parent and shared source
manifest. Textual reported validation loss 2.0444399 on 1,779 targets; atomic
reported 2.4455797 on 1,653 targets. Production nevertheless selects atomic
special tokens by ADR 0099 because reasoning/answer boundaries are a semantic
machine protocol and must not be conflated with ordinary language tokens.
