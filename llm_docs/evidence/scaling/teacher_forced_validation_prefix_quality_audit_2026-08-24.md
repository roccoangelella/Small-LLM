---
status: evidence
observed_at: 2026-08-24
suite: teacher_forced_validation-default-4096
---

# Teacher-forced validation-prefix quality audit

## Scope

This audit covers **every source document that contributes a target to the default 4,096-token teacher-forced diagnostic**, not the entire frozen 16-block validation inventory.

The default diagnostic consumes targets sequentially from validation block 0. With context 2,048, 4,096 targets are exactly the first two stored validation sequences. The corresponding validation stream was reconstructed from the pinned public `nvidia/Nemotron-ClimbMix` source using the repository's frozen source identity, work-plan hash ordering, deterministic document-level validation split, parallel document ordering, and EOD-separated sequence packing.

The reconstruction produced eight source documents through stream token 4,147. The evaluated target range is absolute validation-stream token indices 1..4,096, so document 7 is only partially represented by the teacher-forced metric even though the entire source document was inspected for this audit.

Two independent positions supplied from the live teacher-forced report were used as an oracle:

```text
absolute stream target 1718 -> " coat"
context contains             -> "...the parg coat will crack"

absolute stream target 2025 -> " ball"
context contains             -> "...minerals and ballast substances..."
```

Both matched exactly. This is strong evidence that the reconstructed document order and token stream are the same prefix evaluated by the diagnostic.

## Document-by-document audit

The manual labels below concern source/document quality. They do **not** alter the frozen evaluation set.

| doc | cluster | source tokens | teacher-forced targets | share of 4,096 | manual classification | findings |
|---:|---:|---:|---:|---:|---|---|
| 0 | 4 | 190 | 190 | 4.64% | clean; minor boilerplate | Coherent Learning Unboxed show notes about Ohio State's WOW program. `Share Episode` / `Shownotes` are harmless page boilerplate. Names and program identity externally corroborate. |
| 1 | 12 | 638 | 639 | 15.60% | clean technical prose; minor editorial awkwardness | Coherent 2018 LED-driver/converter abstract. The lecture title and author are independently corroborated. Some non-native phrasing and a keyword-list tail, but no structural contamination. |
| 2 | 12 | 392 | 393 | 9.59% | degraded source-quality / translated SEO prose | Internally coherent LED-lighting topic but frequent grammatical defects and technical imprecision. It begins with the incorrect expansion `Lighting Emitting Diode`; standard usage is `light-emitting diode`. MICROLED/COB descriptions and broad efficiency claims are also loose. No packing/tokenizer corruption. |
| 3 | 12 | 702 | 703 | 17.16% | mostly clean article with format residue | The Polli Construction block-wall article is coherent and externally matches the source, including `the parg coat will crack`. Six private-use icon glyphs, a scraped comment, and an appended QA pair are source-record residue. `parg coat` is unusual but defensible construction wording and must not be classified as a pipeline error. |
| 4 | 17 | 170 | 171 | 4.17% | severe scrape/composite contamination | A single ClimbMix record splices at least two independently identifiable web sources: a Pleasant Hill Grain stone-burr paragraph and Double-lion stone-mill marketing copy, then degenerates into broken search-result-like text (`AFeb... Thewhat... residenzasanmichele.eu`) before an appended QA pair. The phrase `ballast substances` itself is present in the original Double-lion page and is not tokenizer corruption; the document surrounding it is badly contaminated. |
| 5 | 17 | 403 | 404 | 9.86% | coherent article with moderate scrape/metadata damage | Coconut-flour pancake recipe is readable, but contains health-marketing claims, a broken anchor concatenation (`True Food Detoxfor beginners`), a missing URL (`Learn more at  or`), Laura/Lauren name inconsistency, a decorative separator, and two appended QA pairs. |
| 6 | 12 | 756 | 757 | 18.48% | partially truncated / scrape-derived article | Drain-tile article starts mid-explanation, contains awkward wording (`Drainage systems in Interior Wall Tiles`), ends on a dangling heading (`Material for drain tile`), and appends four QA pairs. Local paragraphs remain understandable, but this is not a clean document boundary/extraction. |
| 7 | 6 | 889 | 839 | 20.48% | coherent human science blog with factual/technical caveats | Authentic-looking comparative-genomics/BLAST explainer plus author bio. Language is mostly coherent, with typos (`resides` for `residues`, `mentioned early`). The BLAST analogy introduces real technical inaccuracies: protein BLAST commonly uses word size 3, but word size is configurable and nucleotide BLAST uses other defaults; extension is score-driven, not literally performed in fixed groups of three; BLAST does not first assign a query to a family and then search only a family-specific box. No evidence of source concatenation or Small-LLM pipeline corruption. |

### Manual quality buckets by evaluated target share

These buckets are intentionally coarse and subjective; they are useful for interpreting this 4,096-target diagnostic, not for estimating corpus-wide quality from eight documents.

```text
clean / minor editorial issues                829 / 4096 = 20.24%   (docs 0, 1)
usable with notable format/technical caveats 1542 / 4096 = 37.65%   (docs 3, 7)
degraded but locally coherent                1554 / 4096 = 37.94%   (docs 2, 5, 6)
severe composite/scrape contamination         171 / 4096 =  4.17%   (doc 4)
```

The strongest negative finding is therefore **not** that most text is random corruption. It is that this tiny prefix contains a substantial amount of mediocre web prose, truncation/anchor damage, QA/metadata tails, and one clearly composite scrape failure. Exact-token probabilities on those spans need source-quality context before being interpreted as behavioral model failures.

## No evidence of Small-LLM tokenizer or packing corruption

No inspected anomaly required a tokenizer, decoder, sequence-packer, or shard-corruption explanation.

The source contract reuses existing GPT-2 IDs verbatim, document assignment is a deterministic hash of permanent source identity, and the validation packer appends an EOD boundary before concatenating the next document. The reconstructed pinned-source prefix matched both live teacher-forced oracle positions exactly.

Consequently:

- `parg` + target `" coat"` comes from the source record itself;
- `" ball"` beginning `ballast` comes from the source record itself;
- private-use glyphs, broken anchors, QA suffixes, truncations, and composite fragments are already present inside ClimbMix source records;
- document-to-document packing preserves an EOD boundary and is not responsible for the internal composite text in document 4.

## More important diagnostic-design finding: the default 4,096 targets are not representative

The 4,096 targets occupy only two sequences out of the 16-block frozen validation prefix:

```text
16 blocks * 64 sequences/block * 2048 targets/sequence = 2,097,152 frozen targets
4,096 / 2,097,152 = 0.1953125%
```

Only four accepted clusters occur in the default teacher-forced slice. Using the project's approved programming-cluster-excluded mixture weights:

| cluster | intended mixture share | default 4,096-target share |
|---:|---:|---:|
| 4 | 3.64% | 4.64% |
| 6 | 20.17% | 20.48% |
| 12 | 22.26% | **60.84%** |
| 17 | 7.61% | 14.04% |

The other 15 accepted clusters receive zero targets in the default diagnostic even though together they represent about **46.32%** of the approved mixture. Cluster 12 alone is sampled at about 2.73x its intended mixture share.

Therefore the teacher-forced summary's `loss` and `perplexity` are valid deterministic statistics for this fixed prefix, but they should **not** be interpreted as a broad validation-loss/perplexity estimate of the model. Their best use is a stable checkpoint-to-checkpoint confidence microscope plus raw outlier inspection.

## Interpretation of the two motivating examples

### `parg` -> `" coat"`

The exact source article contains `the parg coat will crack`. The model's extremely confident `"ing"` alternative is linguistically plausible (`parging`), but the ground-truth continuation is source-authentic and defensible. This is a real exact-token disagreement, not corrupted validation data.

### `minerals and` -> `" ball"` (`ballast`)

The phrase `minerals and ballast substances` exists verbatim in the original Double-lion product copy. The unusual wording is therefore source-authentic. However, the ClimbMix record containing it is a severe composite scrape and the underlying marketing page itself is low-editorial-quality/non-native English. The extreme token-level penalty should be retained for frozen comparability but should not be cited alone as evidence that the model misunderstood the semantic continuation.

## Recommended follow-up (not yet an adopted decision)

Keep the current frozen diagnostic for historical checkpoint comparability, but add a second teacher-forced view that samples across the frozen validation inventory rather than taking only its first two sequences. A stratified fixed-position design across the 16 frozen blocks can preserve determinism while greatly improving cluster/document coverage. Representative outliers should also carry source-document identity/boundary metadata so manual review can immediately distinguish a model failure from a noisy source span.

Do not silently clean or delete the current held-out documents after seeing model results: that would invalidate the frozen comparison. If a curated-quality diagnostic is desired, create it as an explicitly separate frozen suite.

## External corroboration used during review

- Learning Unboxed episode 122 / Ohio State WOW identity.
- Polli Construction, `How to Support a Block Wall` (source contains `the parg coat will crack`).
- Pleasant Hill Grain, `Buying Guide for Grain Mills` (source of the stone-burr paragraph in composite document 4).
- DIYTrade / Zhengzhou Double-lion product page (source contains `minerals and ballast substances`).
- U.S. Department of Energy LED material (standard `light-emitting diode` definition).
- NCBI BLAST documentation (configurable word-size/seed-and-extension algorithm and statistical scoring).
- NVIDIA CLIMB project description (ClimbMix derives from filtered/reclustered Nemotron-CC + smollm-corpus rather than manual per-document curation).

A separate 2026 independent ClimbMix analysis (`castorini/cmass`) reports that duplicated high-frequency documents are disproportionately boilerplate, single-word/navigation material, and scraping failures. That observation is consistent with the kinds of residue found here, but it is not used to infer a corpus-wide contamination rate from this eight-document audit.
