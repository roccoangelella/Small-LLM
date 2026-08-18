# R-SFT R0 data lane

This folder owns reasoning-SFT dataset production, the extended GPT-2 tokenizer,
S0 instruction retention, immutable bundles, and the historical delimiter
ablation.

## Accepted R0 artifact

The accepted 100M / 2B R0 R-SFT checkpoint is the already-completed atomic arm:

```text
100m-2b-rsft-r0-atomic-pilot-001
```

ADR 0100 promotes this exact artifact rather than retraining the same data under
a different name. The historical `pilot` suffix remains part of its scientific
provenance and does not mean the checkpoint is unaccepted.

The frozen R0 reasoning corpus is:

```text
artifacts/rsft-r0-pilot-630/generation/reasoning.jsonl
```

It contains 630 Gemini examples (7 skills x 3 difficulty bands x 30 examples).
The atomic run trained that corpus plus the accepted 10% S0 instruction-retention
lane for one complete pass.

## Accepted token contract

R-SFT is atomic-only:

```text
50257  <think>
50258  </think>
50259  <answer>
```

These are control symbols, not ordinary natural-language delimiters. The
historical textual `Reasoning:` / `Answer:` arm remains reproducible only as
ablation evidence and is not an accepted serialization option.

## Chat with the accepted model

```bash
python chat.py --model_params 100M --num_tokens 2B --r-sft
```

R-SFT artifact lookup prefers `SMALL_LLM_RSFT_HF_REPO_ID`, then falls back to
`SMALL_LLM_SFT_HF_REPO_ID` and `SMALL_LLM_HF_REPO_ID`. The loader rejects
checkpoints that are incomplete, non-atomic, use a different marker spelling, or
do not use semantic vocabulary size 50,260.

## Historical 630-example delimiter ablation

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

The two completed summaries used the same S0 parent and shared source manifest.
Textual reported validation loss 2.0444399 on 1,779 targets; atomic reported
2.4455797 on 1,653 targets. Atomic is nevertheless the accepted artifact by ADR
0099/0100 because reasoning/answer boundaries are a semantic machine protocol
and must not be conflated with ordinary language tokens.

## Future larger reasoning corpus

`build_atomic.py` remains available for a future **new or larger** reasoning
corpus if that expansion is explicitly approved. It is not necessary for the
currently accepted R0 checkpoint.

For a future frozen corpus:

```bash
python post_training/R-SFT/build_atomic.py \
  --reasoning-jsonl /path/to/new-reasoning.jsonl \
  --s0-bundle /path/to/100m-2b-sft-s0-bundle \
  --output-dir /path/to/rsft-r0-expanded \
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

The production Kaggle `train` path remains tooling for such a future explicitly
approved expansion. Do not use it merely to duplicate the accepted 630-example
R0 run under `100m-2b-rsft-r0-001`.
