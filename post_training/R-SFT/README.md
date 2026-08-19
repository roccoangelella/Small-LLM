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

## Repeated-epoch corpus-size diagnostic

ADR 0101 adds an explicit experimental repeat lane. It starts again from the
completed S0 parent and replays the exact same immutable R-SFT train blocks in
the same order. This is intended to test whether the 29-update pilot simply had
too little exposure; it is not a production default.

For the requested ten-pass atomic probe:

```bash
python kaggle/launch_r_sft.py ablation \
  --model 100M \
  --tokens 2B \
  --delimiter-format atomic \
  --num-epochs 10
```

With the current 29-block atomic bundle this produces exactly 290 optimizer
steps. The automatic run ID is:

```text
100m-2b-rsft-r0-atomic-repeat-e10-001
```

The WSD schedule spans the full 10-pass stream rather than resetting each pass.
Checkpoint/resume uses logical block IDs across epochs, so a resume after (for
example) step 173 continues at the exact next repeated block. `--num-epochs 1`
preserves the historical one-pass behavior. The canonical production `train`
lane rejects `--num-epochs > 1`.

## First large production R-SFT

ADR 0104 approves the first large R-SFT corpus at:

```text
artifacts/rsft-superior-instruction-r0/reasoning.jsonl
```

It uses Superior Reasoning Stage-1 `instruction_following` only, removes
math/computation/code-primary tasks, requires every Superior record to fit the
real 2,048-token atomic R-SFT serialization without truncation, and merges the
frozen 630 Gemini logic examples. Exact realized counts and the JSONL SHA-256 are
recorded next to the corpus in `reasoning.jsonl.manifest.json`.

The production mixture keeps the previous R-SFT contract: 90% reasoning and 10%
completed S0 instruction retention by loss-bearing target tokens. The S0 lane
preserves its original instruction-source proportions and excludes ClimbMix
replay from retention.

Canonical Kaggle launch:

```bash
python kaggle/launch_r_sft.py train --model 100M --tokens 2B
```

No `--dataset-dir` is required. The launcher resolves the completed S0 bundle,
builds the native production bundle from the committed reasoning JSONL, verifies
it, and starts the qualified 2xT4 DDP trainer. `--dataset-dir` remains available
for an explicitly prebuilt verified production bundle, and `--s0-bundle` can
override S0 resolution.

The production builder can also be run directly:

```bash
python post_training/R-SFT/build_atomic.py \
  --reasoning-jsonl artifacts/rsft-superior-instruction-r0/reasoning.jsonl \
  --s0-bundle /path/to/100m-2b-sft-s0-bundle \
  --output-dir /path/to/rsft-r0-superior-instruction
```

The builder:

- accepts heterogeneous reasoning groups rather than requiring the historical
  uniform 7 x 3 Gemini matrix;
- deterministically holds out 1% validation and 1% test inside each
  `skill x difficulty` group, with at least one record per held-out split;
- uses only the frozen atomic `<think>`, `</think>`, `<answer>` token mapping;
- computes the 10% S0 retention target from the atomic reasoning-token count;
- samples exact tokenized S0 instruction records while preserving S0 source
  stratification and excluding ClimbMix replay;
- defaults to 32,768 loss-bearing target tokens per optimizer block;
- writes and verifies a native bundle marked `rsft.contract=atomic-production-v1`
  and `reasoning_corpus_contract=heterogeneous-groups-v1`.
