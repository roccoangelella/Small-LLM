# R-SFT R0 data lane

This folder owns reasoning-SFT dataset production, the extended GPT-2 tokenizer,
S0 instruction retention, immutable bundles, and the historical delimiter
ablation.

## Accepted R0 artifact

The current accepted 100M / 2B R0 R-SFT checkpoint is:

```text
100m-2b-rsft-r0-12306-001
```

It completed one exact production pass at `step-00000361`. Its historical training corpus contained 12,306 unique normalized prompts (7,683 unchanged context-fit Superior instruction rows, 3,993 unique accepted Variant-D rewrites, and 630 frozen Gemini logic anchors) at SHA-256 `e7d83f9809a65bcb50a6dea3087813d92fea1950a716b3c1eb13e87bfe263a5e`. The intermediate 12,306-row corpus file has been retired from the current tree now that the expanded corpus is complete; it remains recoverable from Git history at commit `2ae60bfa135017353f39da2ef34a6124cda465dc`. The earlier 630-example atomic/textual delimiter runs and the 10-epoch atomic repeat are historical experiments only.

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
2.4455797 on 1,653 targets. ADR 0099/0100 selected the atomic arm at that stage
because reasoning/answer boundaries are a semantic machine protocol rather than
ordinary language tokens. ADR 0105 later promoted the larger 12,306-row run as
the current accepted R0 artifact.

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
preserves the historical one-pass behavior. Production `train` now uses the same
exact-repeat mechanism when `--num-epochs > 1`, with an automatically distinct
epoch-specific run ID.

## Default production training corpus

The standard production R-SFT training input is now:

```text
artifacts/rsft-superior-instruction-r0-expanded/reasoning.jsonl
```

It contains 16,716 unique normalized prompts: 7,683 unchanged Superior instruction rows, 8,403 unique accepted Variant-D rewrites, and 630 Gemini logic anchors. All 8,473 curation-v2 keepers were processed; 70 accepted rewrites were excluded for normalized-prompt collisions. SHA-256 is `d13052b6fc33108ec65511b790a75f6473144855059b16b55167b046f787c405`, and every row fits the exact atomic 2,048-token serialization.

Canonical one-epoch Kaggle launch:

```bash
python kaggle/launch_r_sft.py train --model 100M --tokens 2B
```

A two-epoch production replay is now equally direct:

```bash
python kaggle/launch_r_sft.py train --model 100M --tokens 2B --num-epochs 2
```

For the 417-block expanded bundle this is 834 optimizer steps. The launcher automatically uses run ID `100m-2b-rsft-r0-16716-e2-001`; the one-epoch identity `100m-2b-rsft-r0-16716-001` cannot be reused for a multi-epoch run.

With no `--dataset-dir`, the launcher validates the committed 16,716-row corpus from pinned worktree commit `2ae60bfa135017353f39da2ef34a6124cda465dc`, resolves the completed 100M/2B S0 parent, builds/verifies the 90/10 `atomic-production-v1` bundle with 32,768 target tokens per optimizer block, and uses fresh default run ID `100m-2b-rsft-r0-16716-001`. It must not resume the historical 12,306-row run.

Direct bundle build:

```bash
python post_training/R-SFT/build_atomic.py \
  --reasoning-jsonl artifacts/rsft-superior-instruction-r0-expanded/reasoning.jsonl \
  --s0-bundle /path/to/100m-2b-sft-s0-bundle \
  --output-dir /path/to/rsft-r0-superior-instruction-expanded-16716
```

The verified reference build contains 417 train blocks and 13,420,823 train targets: 12,077,733 reasoning plus 1,343,090 S0 retention.

## Current accepted trained checkpoint

The `100m-2b-rsft-r0-12306-001` trajectory completed on 2026-08-19 at `step-00000361` and remains the accepted R-SFT model until a 16,716-row run is actually trained and qualified. Chat remains:

```bash
.venv/bin/python chat.py --model_params 100M --num_tokens 2B --r-sft
```

The explicit `--run-id 100m-2b-rsft-r0-12306-001` form is equivalent. `train` now defaults to `100m-2b-rsft-r0-16716-001`, while `eval` still resolves the accepted `100m-2b-rsft-r0-12306-001` model during this transition.
