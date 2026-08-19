# 100M / 2B R-SFT R0 12,306-row completion — 2026-08-19

## Observation

The ADR-0105 12,306-row R-SFT checkpoint was frozen, built into the native atomic production bundle, trained for one exact pass on the qualified Kaggle dual-T4 path, and published as a completed Hugging Face checkpoint.

## Frozen reasoning corpus

```text
artifacts/rsft-superior-instruction-r0-checkpoint-12306/reasoning.jsonl
```

Verified corpus facts:

- total reasoning rows: 12,306;
- unchanged Superior instruction rows: 7,683;
- unique accepted Variant-D Superior rewrites: 3,993;
- Gemini logic anchors: 630;
- normalized-prompt duplicates in emitted corpus: 0;
- atomic serialized-token range: 61–2,048;
- JSONL SHA-256: `e7d83f9809a65bcb50a6dea3087813d92fea1950a716b3c1eb13e87bfe263a5e`;
- accepted rewritten keepers omitted for prompt collision: 28;
- manually-kept over-context candidates still awaiting compression: 4,476.

The full manual curation covers all 9,624 frozen over-context candidates: 8,497 `keep`, 829 `exclude_code`, 212 `exclude_math`, and 86 `exclude_safety`.

## Native training bundle

The bundle was built against the completed 100M/2B S0 parent bundle with the frozen 90% reasoning / 10% S0 instruction-retention contract and 32,768 loss-bearing target tokens per optimizer block.

Verified train geometry:

- train optimizer blocks: 361;
- reasoning train targets: 10,448,098;
- S0-retention train targets: 1,161,354;
- total train targets: 11,609,452;
- validation reasoning records: 138, packed into 4 blocks;
- test reasoning records: 138, packed into 4 blocks.

## Completed R-SFT checkpoint

Run identity:

```text
100m-2b-rsft-r0-12306-001
```

Hugging Face repository and live namespace:

```text
roccoangelella/small-llm-100m-qualification
run/100m-2b-rsft-r0-12306-001/
```

The verified `latest.json` pointer names:

```text
step-00000361
```

This matches the 361-block one-pass training plan and is the completed final optimizer boundary.

## Hugging Face cleanup

After the completed 12,306-row checkpoint was verified, the three superseded R-SFT trial namespaces were deleted from the shared Hugging Face repository:

```text
100m-2b-rsft-r0-atomic-pilot-001
100m-2b-rsft-r0-atomic-repeat-e10-001
100m-2b-rsft-r0-textual-pilot-001
```

The current R-SFT namespace `100m-2b-rsft-r0-12306-001` and the completed S0 parent `100m-2b-sft-s0-001` were preserved. The deletion commits reported by Hugging Face were `72fe7de8159bffe5a88265aec87fe491e5d1390a`, `3539f65c9563620924e8720ef1a9246dbd12047b`, and `47296d7384db8a64761624b52a0f13065ede6e26` respectively.

## Remaining adaptation state

The future expansion work remains resumable locally but is not part of this completed run:

- GemRouter service: inactive;
- adaptation worker processes: none;
- frozen candidates: 9,624;
- final curation rows: 9,624;
- accepted Variant-D batch checkpoints: 1,122;
- remaining curated keepers needing compression: 4,476.

Generated attempts, logs, duplicate review extracts, cards/packs, and OpenCode scratch state were removed. The candidate cache, candidate manifest, final manual curation, and accepted batch checkpoints were retained locally. Provider fallback to NVIDIA is not part of the accepted adaptation path; any future resume uses the selected Gemini Variant-D compressor unless a later explicit decision changes that contract.

## Source commits

The frozen corpus and launcher wiring were committed as:

```text
b6bf8df5aa3c835278214a4ba4adb91225b6b672  Freeze 12,306-row R-SFT checkpoint
7c1b714efbdd320c1f00fb612cee4b87974a52f5  Pin 12,306-row R-SFT launch commit
```
