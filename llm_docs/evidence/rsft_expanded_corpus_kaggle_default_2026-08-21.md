# Expanded R-SFT corpus promoted to Kaggle default — 2026-08-21

The completed 16,716-row corpus at `artifacts/rsft-superior-instruction-r0-expanded/reasoning.jsonl` was already committed in `2ae60bfa135017353f39da2ef34a6124cda465dc`, SHA-256 `d13052b6fc33108ec65511b790a75f6473144855059b16b55167b046f787c405`.

The production Kaggle preparation contract now validates schema `small-llm-superior-reasoning-curated-complete-v1`, 16,716 rows, 8,403 unique adapted Superior rows, all 8,473 accepted keep rewrites, 70 duplicate-rewrite exclusions, exact serialized range 61–2,048, and the frozen corpus SHA. The default production bundle name is `rsft-r0-superior-instruction-expanded-16716`.

The standard training run identity is `100m-2b-rsft-r0-16716-001`, while the accepted trained checkpoint remains `100m-2b-rsft-r0-12306-001` for chat/evaluation until replacement training is completed and qualified. The detached Kaggle worktree is pinned to `2ae60bfa135017353f39da2ef34a6124cda465dc`, which already contains the final corpus and compatible atomic production code.

The tracked intermediate `artifacts/rsft-superior-instruction-r0-checkpoint-12306/` corpus was removed from the current tree. Its historical SHA-256 remains `e7d83f9809a65bcb50a6dea3087813d92fea1950a716b3c1eb13e87bfe263a5e` and the file is recoverable from commit `2ae60bfa135017353f39da2ef34a6124cda465dc`. The 8,313-row baseline corpus remains because it is construction provenance for the expanded corpus rather than a launcher default.
