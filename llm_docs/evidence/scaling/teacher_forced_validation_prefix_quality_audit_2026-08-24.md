---
status: evidence
observed_at: 2026-08-24
revised_at: 2026-08-25
suite: teacher_forced_validation-default-4096
---

# Teacher-forced validation-prefix quality audit

## Scope

This audit covers **every source document that contributes a target to the default 4,096-token teacher-forced diagnostic**, not the entire frozen 16-block validation inventory.

The default diagnostic consumes active labels sequentially from validation block 0. With context 2,048, 4,096 scored targets are exactly the first two stored validation sequences. Because schema-v2 stores `context_plus_one` sequences with stride 2,048, sequence 0 contains validation-stream tokens `0..2048` and scores targets `1..2048`; sequence 1 contains stream tokens `2048..4096` and scores targets `2049..4096`. Thus `4,096` means **4,096 next-token predictions**, not a 4,096-token context window and not a separately curated 4,096-token dataset.

The corresponding validation stream was reconstructed from the pinned public `nvidia/Nemotron-ClimbMix` source using the repository's frozen source identity, work-plan hash ordering, deterministic document-level validation split, parallel document ordering, and EOD-separated sequence packing.

The reconstruction produced eight source documents through stream token 4,147. The evaluated target range is absolute validation-stream token indices `1..4096`, so document 7 is only partially represented by the metric even though its full source record was inspected.

Two independent positions supplied from the live teacher-forced report were used as an oracle:

```text
absolute stream target 1718 -> " coat"
context contains             -> "...the parg coat will crack"

absolute stream target 2025 -> " ball"
context contains             -> "...minerals and ballast substances..."
```

Both matched exactly. This is strong evidence that the reconstructed document order and token stream are the same prefix evaluated by the diagnostic.

## Important correction: `parg coat` is a source typo

The underlying Polli Construction page itself currently says `the parg coat will crack`, so Small-LLM did not create the string. However, the standard construction term is **`parge coat`**; dictionaries and construction references use `parge coat`, while the verb inflects as `parged` / `parging`. Bare `parg` is not the standard form in `parg coat`.

Therefore this position should be classified as a **source-authentic typo / lexical defect**. The model's 98.5% prediction of `"ing"` after `parg` forms the valid word `parging` and is linguistically better than reproducing the typo. Teacher-forced exact-token scoring nevertheless penalizes it because the frozen target is the source token `" coat"`.

This is not tokenizer or packing corruption, but it **is** noisy ground truth.

## Important correction: QA tails are often intentional Nemotron-CC synthetic data

Several records end with `Question: ... Answer: ...` pairs. These should not automatically be labeled scrape residue. ClimbMix is constructed from Nemotron-CC plus smollm-corpus, and Nemotron-CC intentionally generated **Diverse QA** variants by appending shuffled synthetic question-answer pairs to original document segments. NVIDIA's NeMo Curator implementation documents the same behavior.

Accordingly, QA tails in docs 3-6 are best treated as **intentional synthetic augmentation unless source-specific evidence shows otherwise**. They can still make a teacher-forced slice less representative of ordinary prose, but their presence is not itself evidence of scraping failure.

## Document-by-document audit

| doc | cluster | source tokens | teacher-forced targets | share of 4,096 | manual classification | findings |
|---:|---:|---:|---:|---:|---|---|
| 0 | 4 | 190 | 190 | 4.64% | clean; minor boilerplate | Coherent Learning Unboxed show notes about Ohio State's WOW program. `Share Episode` / `Shownotes` are harmless page boilerplate. |
| 1 | 12 | 638 | 639 | 15.60% | clean technical prose; minor editorial awkwardness | Coherent 2018 LED-driver/converter abstract. Some non-native phrasing and a keyword-list tail, but no structural contamination. |
| 2 | 12 | 392 | 393 | 9.59% | degraded source-quality / translated SEO prose | Frequent grammatical defects and technical imprecision. It begins with the incorrect `Lighting Emitting Diode`; standard usage is `light-emitting diode`. No packing/tokenizer corruption. |
| 3 | 12 | 702 | 703 | 17.16% | mostly clean article + source typo + web/synthetic residue | Polli Construction article is coherent, but contains the source typo `parg coat` where standard English is `parge coat`, private-use icon glyphs, a scraped comment, and an appended QA pair likely attributable to Nemotron-CC Diverse-QA augmentation. |
| 4 | 17 | 170 | 171 | 4.17% | severe composite/scrape contamination + synthetic QA | One ClimbMix record splices at least two identifiable web sources, then degenerates into broken search-result-like text (`AFeb... Thewhat... residenzasanmichele.eu`). The final QA pair is likely intentional augmentation. `ballast substances` itself is source-authentic; the surrounding record is badly contaminated. |
| 5 | 17 | 403 | 404 | 9.86% | coherent source with moderate extraction damage + synthetic QA | Readable coconut-flour recipe, but broken anchor concatenation (`True Food Detoxfor beginners`), missing URL, Laura/Lauren inconsistency, decorative separator, plus two likely synthetic QA pairs. |
| 6 | 12 | 756 | 757 | 18.48% | truncated/scrape-derived article + synthetic QA | Starts mid-explanation, contains awkward wording, ends on dangling heading `Material for drain tile`, then four likely synthetic QA pairs. Local prose remains understandable. |
| 7 | 6 | 889 | 839 | 20.48% | coherent human science blog with factual/technical caveats | Comparative-genomics/BLAST explainer with typos and several oversimplifications/inaccuracies. No evidence of source concatenation or Small-LLM pipeline corruption. |

## Manual quality buckets by evaluated target share

These buckets are deliberately coarse and apply only to this tiny fixed prefix:

```text
clean / minor editorial issues                829 / 4096 = 20.24%   (docs 0, 1)
usable with notable source/technical caveats 1542 / 4096 = 37.65%   (docs 3, 7)
degraded but locally coherent                1554 / 4096 = 37.94%   (docs 2, 5, 6)
severe composite/scrape contamination         171 / 4096 =  4.17%   (doc 4)
```

The strongest negative finding is not random binary corruption. It is that the prefix contains a substantial amount of mediocre web prose, source typos, truncation/anchor damage, deliberate synthetic QA augmentation, and one clearly composite scrape failure. Exact-token probabilities on those spans need source-quality context before being interpreted as model-language failures.

## No evidence of Small-LLM tokenizer or packing corruption

No inspected anomaly required a tokenizer, decoder, sequence-packer, or shard-corruption explanation. The source contract reuses existing GPT-2 IDs verbatim, document assignment is a deterministic hash of permanent source identity, and the validation packer inserts an EOD boundary before concatenating the next document. The reconstructed pinned-source prefix matched both live teacher-forced oracle positions exactly.

Consequently:

- `parg` + target `" coat"` is a **source-authentic typo**, not a Small-LLM transformation bug;
- `" ball"` beginning `ballast` is source-authentic;
- private-use glyphs, broken anchors, truncations, composite fragments, and synthetic QA suffixes are already present inside ClimbMix records;
- document-to-document packing preserves an EOD boundary and is not responsible for the internal composite text in document 4.

## Diagnostic-design finding: the default 4,096 targets are not representative

The 4,096 targets occupy only two sequences out of the 16-block frozen validation prefix:

```text
16 blocks * 64 sequences/block * 2048 targets/sequence = 2,097,152 frozen targets
4,096 / 2,097,152 = 0.1953125%
```

Only four accepted clusters occur in the default slice:

| cluster | intended mixture share | default 4,096-target share |
|---:|---:|---:|
| 4 | 3.64% | 4.64% |
| 6 | 20.17% | 20.48% |
| 12 | 22.26% | **60.84%** |
| 17 | 7.61% | 14.04% |

The other 15 accepted clusters receive zero targets even though together they represent about **46.32%** of the approved mixture. Cluster 12 is sampled at about 2.73x its intended mixture share.

Therefore `loss`, `perplexity`, and the confidence/rank summaries are valid deterministic statistics for this exact fixed prefix, but **not broad validation estimates**. Their best use is checkpoint-to-checkpoint comparison and raw outlier inspection.

## Exact meaning of the two-sequence prefix

The packer concatenates accepted validation documents with EOD boundaries into one stream and emits training-like 2,049-token records with a one-token overlap between adjacent records.

For the default test:

```text
sequence 0 stored stream tokens: 0 ... 2048
sequence 0 model inputs:         0 ... 2047
sequence 0 scored targets:       1 ... 2048

sequence 1 stored stream tokens: 2048 ... 4096
sequence 1 model inputs:         2048 ... 4095
sequence 1 scored targets:       2049 ... 4096
```

So the test scores **every next-token transition in the first two packed validation sequences**. It does not randomly choose 4,096 positions. It also means context resets at the beginning of sequence 1, matching the fixed-length training geometry: the first target of sequence 1 is predicted from only its first input token, not from all 2,048 preceding stream tokens.

The eight source-document target spans are:

```text
doc 0: targets    1..190    = 190
doc 1: targets  191..829    = 639
doc 2: targets  830..1222   = 393
doc 3: targets 1223..1925   = 703
doc 4: targets 1926..2096   = 171
doc 5: targets 2097..2500   = 404
doc 6: targets 2501..3257   = 757
doc 7: targets 3258..4096   = 839 (partial document)
```

## Recommended follow-up (not yet an adopted evaluation-design decision)

Keep the existing fixed-prefix diagnostic for historical comparability, but add a separate deterministic teacher-forced view sampled across the frozen validation inventory rather than only the first two sequences. A block-stratified fixed-position design can preserve determinism while improving cluster/document coverage. Representative rows should carry source-document identity, document-boundary information, and an explicit source-quality/synthetic-QA indicator when available.

Do not silently clean the current held-out prefix after observing model errors. If a curated-quality diagnostic is wanted, create it as a separate frozen suite.
