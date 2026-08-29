---
status: evidence
observed_at: 2026-08-24
suite: eval_core_v1-full-plus-greedy32-plus-sampled-t1-k20-p09
---

# 100M/10B mid-run evaluation at approximately 6.1B tokens

This record captures the user-supplied mid-run qualification of the still-training `100m-10b-deep-decay-from-step15500` trajectory and compares it with the completed 100M/2B pretraining endpoint. It is an intermediate measurement, not the final 10B result.

## Checkpoints and protocol identity

Intrinsic full eval:

```text
run:             100m-10b-deep-decay-from-step15500
checkpoint:      step-00046250
global step:     46,250
consumed tokens: 6,062,080,000
architecture:    gdn2_hybrid, d_model=512, d_ff=1408, 20 layers, context 2048
eval manifest:   aa7b6157e5f420dd53a99552685eaed01962ee45c23cbe438e1321a886422792
```

The frozen 100M/2B comparator is `step-00015267` at 2,001,000,448 consumed targets and uses the same `eval_core_v1` manifest, so the intrinsic metrics are directly comparable.

Prompt behavior was measured at the nearby `step-00046500` checkpoint (6,094,848,000 consumed tokens) with the same two protocols used for the 100M/2B behavioral qualification:

```text
greedy-32: temperature=0.0, top_p=1.0, top_k=0, seed=17, max_new_tokens=32
sampled:   temperature=1.0, top_p=0.9, top_k=20, seed=17
```

## Intrinsic comparison: 100M/2B -> 100M/~6.06B

| metric | 100M / 2B | 100M / ~6.06B | change |
|---|---:|---:|---:|
| loss | 3.338815 | 3.167205 | -0.171610 (-5.14%) |
| perplexity | 28.185701 | 23.741041 | -4.444660 (-15.77%) |
| bits / decoded byte | 1.042155 | 0.988590 | -0.053565 (-5.14%) |
| top-1 accuracy | 0.398875 | 0.414504 | +0.015629 (+1.56 pp) |
| top-5 accuracy | 0.618154 | 0.638356 | +0.020202 (+2.02 pp) |
| top-10 accuracy | 0.692041 | 0.712083 | +0.020042 (+2.00 pp) |
| calibration ECE | 0.010091 | 0.007624 | -0.002467 |
| cluster macro loss | 3.349121 | 3.174749 | -0.174372 (-5.21%) |
| source-mixture-weighted cluster loss | 3.042600 | 2.914806 | -0.127794 (-4.20%) |
| worst cluster loss | 3.949742 | 3.721464 | -0.228278 (-5.78%) |

The document-bootstrap 95% loss intervals do not overlap:

```text
100M / 2B:      3.303364 .. 3.374237
100M / ~6.06B:  3.134310 .. 3.199184
```

Thus the extra exposure produces a real and statistically clear intrinsic gain. The scale of that gain is nevertheless modest compared with the earlier capacity jump: 20M->100M at fixed 2B reduced loss by 14.27%, whereas 100M/2B->100M/~6.06B reduces it by 5.14% after roughly tripling token exposure.

For another reference point, the earlier 20M/500M->20M/2B fourfold data increase reduced loss by only 2.81%. The 100M model therefore still benefits from more data, but the returns are not close to the effect obtained by increasing capacity from 20M to 100M.

## Matched factual-QA behavior

Strict direct-answer reading of the 12 QA prompts under greedy-32:

```text
100M / 2B:      2 / 12  (Pacific Ocean, photosynthesis)
100M / ~6.1B:   3 / 12  (Paris, heart, photosynthesis)
```

The approximately-6.1B checkpoint still misses Jupiter, Mars, Shakespeare, 0 C, Pacific Ocean under this greedy sample, yen, Portuguese, 366, and 56. Ten of twelve QA generations stop before the 32-token cap, exactly matching the 100M/2B count of 10/12.

Strict direct-answer reading under the matched T=1/top-k20/top-p0.9 sampled protocol:

```text
100M / 2B:      4 / 12  (Jupiter, approximately 0 C, Pacific Ocean, photosynthesis)
100M / ~6.1B:   4 / 12  (Paris, Pacific Ocean, Portuguese, photosynthesis)
```

The correct-fact set changes, but the count does not improve. With only one stochastic sample per prompt, this is not a robust task-accuracy estimate.

## Free-form continuation quality

The flat strict-QA counts should not be interpreted as flat generation quality. Historical project journal evidence preserves the same `story_opening` prompt at the completed 100M/2B endpoint under the matched T=1/top-k20/top-p0.9 sampling policy:

```text
The rain had stopped before dawn, leaving the streets covered in ...
```

At 100M/2B, the sampled continuation starts with `iced coffee` and then drifts rapidly into grammatically broken, weakly connected parent/children discourse. The accompanying deterministic output starts awkwardly with `ices` and falls into repeated `The rain was still wet, and the sun was shining` loops.

At the ~6.1B checkpoint, the matched sampled continuation instead starts with the locally plausible `iced water` and develops a recognizable children/teachers/school-playground scene. It is still repetitive and not fully coherent at paragraph scale, but sentence formation, local semantic plausibility, entity continuity, and discourse stability are visibly stronger than in the preserved 100M/2B sample. The current greedy-32 continuation also opens with `iced water`, although it then repeats the prompt; because the older deterministic record was not the same 32-token-capped protocol, the sampled comparison is the stronger apples-to-apples evidence.

This is a meaningful qualitative improvement that the strict factual-QA score does not measure. The 100M/~6.1B checkpoint appears substantially better at producing locally coherent prose even though elementary fact retrieval and structured instruction-like behavior remain weak.

### User-confirmed midway observation

After reviewing the preserved 100M/2B output side by side with the ~6.1B output, the user explicitly agreed that the midway run shows a **significant improvement in text quality apart from strict-answer accuracy**. The rain-opening example is the canonical concrete observation for this midway checkpoint: the 2B sampled model begins with the semantically odd `iced coffee` and quickly degrades into broken parent/children discourse, whereas the ~6.1B model begins with the natural `iced water` and maintains a recognizable school/playground scene across multiple sentences. Treat this as a qualitative observation about prose coherence and semantic continuity, not as a claim of improved factual QA.

## Interpretation

Measured conclusion:

> By approximately 6.1B training tokens, the 100M trajectory is clearly better than its 2B-token parent as a language model. Intrinsic loss improves by about 5% and perplexity by about 16%. Strict factual-QA accuracy is nearly flat on the tiny frozen probes, but preserved matched free-form samples show a substantial improvement in local coherence, semantic plausibility, and discourse continuity.

The right distinction is therefore not simply `intrinsic progress versus no behavioral progress`. It is more specific: **free-form language generation quality has improved materially, while factual QA and structured prompt-following have improved much less**. The remaining weaknesses include repetition, schema drift, and elementary factual reliability.

Because this is an intermediate checkpoint before the completed 10B endpoint, it should not be used to pre-judge the final result. The final endpoint should repeat the same frozen full eval, greedy-32, and matched sampled protocols. Future qualitative comparisons should explicitly score continuation coherence separately from strict QA so this dimension is not hidden by answer accuracy.

## Sources

- User-supplied `100m-10b-latest-eval-core-full.json` (`step-00046250`).
- User-supplied `100m-10b-latest-greedy32.json` (`step-00046500`).
- User-supplied `100m-10b-latest-sampled.json` (`step-00046500`).
- Historical `journals/journal10.md`, preserving 100M/2B deterministic and T=1/top-k20/top-p0.9 `story_opening` continuations.
- `100m_2b_behavioral_qualification_2026-08-13.md`.
- `20m_500m_20m_2b_100m_2b_full_eval_2026-08-13.md`.
