# R-SFT R0 atomic runbook

_Last updated: 2026-08-21 Europe/Rome_

The current accepted trained 100M / 2B R-SFT model remains `100m-2b-rsft-r0-12306-001` at Hugging Face step `step-00000361`. The standard next training corpus is the completed 16,716-row expansion promoted by ADR 0108.

## 1. Frozen atomic contract

```text
50257  <think>
50258  </think>
50259  <answer>
context length: 2,048
optimizer target tokens/block: 32,768
reasoning/S0-retention target mix: approximately 90/10
```

Production R-SFT is atomic-only. The historical textual delimiter arm remains ablation evidence only.

## 2. Standard production corpus

```text
artifacts/rsft-superior-instruction-r0-expanded/reasoning.jsonl
rows:   16,716
sha256: d13052b6fc33108ec65511b790a75f6473144855059b16b55167b046f787c405
```

Composition is 7,683 unchanged Superior instruction rows, 8,403 unique simplified Superior rows, and 630 Gemini logic anchors. All 8,473 curation-v2 keepers were processed; 70 accepted rewrites were excluded for normalized-prompt collisions. Every row fits the exact atomic 2,048-token serialization.

The intermediate 12,306-row checkpoint corpus used to train the currently accepted model is no longer stored in the current tree. Its historical identity is SHA-256 `e7d83f9809a65bcb50a6dea3087813d92fea1950a716b3c1eb13e87bfe263a5e`; the deleted file remains recoverable from Git commit `2ae60bfa135017353f39da2ef34a6124cda465dc` for historical reproduction.

## 3. Canonical Kaggle training launch

```bash
python kaggle/launch_r_sft.py train --model 100M --tokens 2B
```

With no `--dataset-dir`, the launcher pins a detached worktree to `2ae60bfa135017353f39da2ef34a6124cda465dc`, SHA-validates the 16,716-row corpus, resolves completed S0 parent `100m-2b-sft-s0-001`, builds/verifies the native `atomic-production-v1` bundle, and launches 2xT4 DDP under fresh default run ID `100m-2b-rsft-r0-16716-001`.

Never point the expanded corpus at `100m-2b-rsft-r0-12306-001`; that ID belongs to the completed historical trajectory.

Useful dry run:

```bash
python kaggle/launch_r_sft.py train --model 100M --tokens 2B --dry-run
```

Direct deterministic bundle build:

```bash
python post_training/R-SFT/build_atomic.py \
  --reasoning-jsonl artifacts/rsft-superior-instruction-r0-expanded/reasoning.jsonl \
  --s0-bundle /path/to/100m-2b-sft-s0-bundle \
  --output-dir /path/to/rsft-r0-superior-instruction-expanded-16716
```

The verified reference build has 417 train blocks, 20,313 packed train records, 12,077,733 reasoning targets, 1,343,090 S0-retention targets, and 13,420,823 total train targets. Validation and test each contain four blocks.

## 4. Current accepted trained model

```bash
.venv/bin/python chat.py --model_params 100M --num_tokens 2B --r-sft
```

The accepted checkpoint repository remains `roccoangelella/small-llm-100m-qualification`, with `run/100m-2b-rsft-r0-12306-001/latest.json` pointing to step 361. The default `eval` command continues to resolve this accepted model until a new 16,716-row run is trained and promoted.

## 5. Historical expansion provenance

Expansion curation v2 remains at `artifacts/rsft-superior-instruction-r0-adaptation/manual-curation.expanded-v2.jsonl`: 8,473 keepers, 829 code exclusions, 212 math exclusions, and 110 safety exclusions. The committed final corpus and manifest are now the production identity. Completion evidence is recorded in [`../evidence/rsft_expanded_corpus_completion_2026-08-21.md`](../evidence/rsft_expanded_corpus_completion_2026-08-21.md).
