---
status: evidence
observed_at: 2026-08-13
suite: adr-0025-greedy-32-plus-sampled-t1-k20-p09
---

# 20M/500M -> 20M/2B -> 100M/2B behavioral qualification

This record captures the six user-supplied result bundles used to close the missing behavioral measurement for the completed pretraining endpoints. It keeps the canonical ADR-0025 deterministic comparison separate from the supplementary sampled comparison authorized by ADR 0059.

## Canonical ADR-0025 comparison

All three prompt-only runs use the full 18-case suite with:

```text
temperature: 0.0
top_p: 1.0
top_k: 0
seed: 17
samples_per_prompt: 1
max_new_tokens: 32
```

Checkpoints:

- 20M/500M: `step-00015264`
- 20M/2B: `step-00061066`
- 100M/2B: `step-00015267`

Strict direct-answer reading of the 12 QA cases:

```text
20M/500M: 0 / 12
20M/2B:   2 / 12  (heart, photosynthesis)
100M/2B:  2 / 12  (Pacific Ocean, photosynthesis)
```

The strict QA count therefore does not improve from 20M/2B to 100M/2B. However generation shape changes materially. QA cases that stop before the 32-token cap are:

```text
20M/500M: 1 / 12
20M/2B:   3 / 12
100M/2B: 10 / 12
```

Across all 18 prompts the same counts are 1/18, 3/18, and 10/18 respectively. This is an exact property of the emitted lengths and should not be silently equated with EOS when a token trace is unavailable.

The 100M greedy QA outputs are substantially less repetitive and often shorter/directer, but factual coverage remains sparse. It still misses France/Paris, Jupiter, Mars, Shakespeare, 0 C exactly, yen, Portuguese, 366, and 56; it answers Pacific Ocean and photosynthesis correctly. Open continuations and structured patterns still show looping or schema failure, including repeated dialogue/list patterns.

## Supplementary T=1 / top-k=20 / top-p=0.9 full comparison

All three full-eval bundles use:

```text
temperature: 1.0
top_k: 20
top_p: 0.9
seed: 17
samples_per_prompt: 1
```

The intrinsic metrics reproduce the prior frozen scaling result because decoding settings affect prompt generation, not teacher-forced eval_core_v1 scoring:

| metric | 20M/500M | 20M/2B | 100M/2B |
|---|---:|---:|---:|
| loss | 4.007289 | 3.894576 | 3.338815 |
| perplexity | 54.997550 | 49.135214 | 28.185701 |
| top-1 | 0.343950 | 0.355129 | 0.398875 |
| top-5 | 0.547491 | 0.561084 | 0.618154 |
| top-10 | 0.620226 | 0.634112 | 0.692041 |

Strict direct-answer reading of the 12 sampled QA cases gives approximately:

```text
20M/500M: 0 / 12
20M/2B:   1 / 12  (heart)
100M/2B:  4 / 12  (Jupiter, ~0 C, Pacific Ocean, photosynthesis)
```

The 100M France-capital sample contains `Paris` but prefixes it with hallucinated text (`L'Oemee Paris`), so it is not counted as a clean strict answer. Under a looser contains-the-fact criterion it would be 5/12. This is separate from the earlier T=0.8 sampled run where the first answer was cleanly `Paris`.

The sampled 100M model therefore exposes more correct factual continuations than either 20M endpoint under the same stochastic policy, even though one sample per prompt is too small to estimate robust task accuracy. It still fails many elementary facts (Mars, Shakespeare, yen, Portuguese, 366, 56) and remains behaviorally weak in absolute terms.

Exact EOS-token counts in the sampled QA token traces are 7/12, 6/12, and 7/12 for 20M/500M, 20M/2B, and 100M/2B respectively; the sampled comparison does not show a monotonic EOS-rate gain. The main sampled behavioral gain is factual accessibility/coherence, not termination rate.

## Interpretation

The exact ADR-0025 measurement is now complete. It shows a mixed behavioral picture:

- canonical greedy strict-QA accuracy ties 20M/2B at 2/12;
- 100M greedy outputs stop before the 32-token cap much more often and are markedly less repetitive on QA;
- the supplementary T=1/k20/p0.9 run exposes substantially more correct factual answers at 100M than at either 20M endpoint;
- intrinsic capability improves very strongly and uniformly at 100M, as recorded in the existing three-way eval_core_v1 evidence;
- absolute base-model behavior is still poor and should not be described as robust factual QA or instruction following.

ADR 0050 deliberately leaves the launch trigger qualitative (`material behavioral/capability improvement`). This evidence completes the measurement needed for that decision but does not itself rewrite ADR 0050 or declare the fresh 100M/10B scientific launch authorized. Record that as an explicit gate decision if the user accepts the evidence as sufficient.

Related records:

- ADR 0025 — canonical greedy full prompt protocol
- ADR 0050 — fresh 100M/10B launch and ~5B continuation gate
- ADR 0059 — supplementary T=1/top-k20/top-p0.9 comparison
- `20m_500m_20m_2b_100m_2b_full_eval_2026-08-13.md` — intrinsic scaling comparison
