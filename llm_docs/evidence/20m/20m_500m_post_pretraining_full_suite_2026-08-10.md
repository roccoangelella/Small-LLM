# 20M / 500M Post-Pretraining Full Qualitative Suite

_Date: 2026-08-10 (Europe/Rome)_

## Evidence source

This record summarizes the completed console output supplied by the user for the canonical full post-pretraining qualitative suite. The suite also reported that it saved machine-readable output to:

```text
artifacts/20m_500m_full_suite.json
```

The JSON artifact itself and its hash were not supplied in this evidence handoff, so this document records the visible run metadata and qualitative outputs rather than claiming byte-level archival of that file.

## Checkpoint identity

```text
repository: roccoangelella/small-llm-20m-qualification
run ID: 20m-500m-dataset-001
pointer: best
checkpoint ID: step-00015264
checkpoint prefix: run/20m-500m-dataset-001/checkpoints/step-00015264/last
global step: 15264
consumed training target tokens: 500,156,416
pointer metric: -3.4395406044446504
architecture: gdn2_hybrid
d_model: 256
d_ff: 704
layers: 8
max sequence length: 2,048
device: cuda
precision: fp16
```

The pointer metric is retained exactly as emitted by the suite. This evidence note does not relabel it as a final evaluation result; the frozen evaluation/scorecard remains authoritative for final quantitative model-quality comparison.

## Canonical generation protocol

The run used the frozen deterministic full-suite settings:

```text
temperature: 0.0
top_p: 1.0
top_k: 0
base seed: 17
samples per prompt: 1
max new tokens: 32
trace top tokens: 0
full suite: yes
questions only: no
```

Prompt-specific seeds followed the suite's deterministic `base_seed + case_index * 1000` rule.

## Suite coverage

The full suite contained 18 cases:

```text
continuation cases: 4
structured cases: 2
general-knowledge question cases: 12
```

The cases covered story continuation, science explanation, encyclopedia-style continuation, dialogue, structured relation completion, sentiment-pattern completion, and twelve simple factual/arithmetic questions.

## Observed outputs

### Continuation behavior

The model produces grammatical-looking fragments and maintains some local lexical/topic cues, but semantic continuation is poor and repetition begins quickly.

Representative observations:

- `story_opening` begins with the unrelated token/word `ichthyophthirius`, then repeats variants of “The rain ... natural disaster.”
- `science_explanation` fails the liquid-to-vapor relation and produces self-referential statements such as water vapor being a liquid used to make water.
- `encyclopedia_style` drifts from the Roman Republic into vernal equinoxes and generic Roman Empire statements rather than completing the historical relation.
- `dialogue` preserves speaker formatting but repeats nearly the same sentence for Alice and Ben and does not track the window-closing context.

Interpretation: the checkpoint has learned substantial surface-form and local-language regularity, but long-range semantic conditioning remains weak in these probes.

### Structured behavior

- `list_pattern` fails the expected `Germany | Berlin` relation and collapses into repeated `Rome |` tokens.
- `sentiment_pattern` emits `negative` as the first classification token, then immediately degenerates into repeated “the effect of the effect ...” text.

Interpretation: the model can recognize and continue a visible text schema, but relation binding and stable structured continuation are unreliable.

### General-knowledge / arithmetic behavior

Under a strict direct-answer reading, **0 / 12 question probes contain the expected answer**.

Case-level notes:

| Case | Expected concept | Observed behavior |
|---|---|---|
| capital of France | Paris | tautologically repeats “capital of France” without naming Paris |
| largest planet | Jupiter | answers with “the solar system” |
| Red Planet | Mars | incorrectly associates it with the world's oceans |
| Hamlet author | William Shakespeare | repeats that “the author” wrote Hamlet without naming Shakespeare |
| water freezing point | 0 °C | gives a 60-degree Fahrenheit range |
| largest ocean | Pacific Ocean | tautologically repeats “largest ocean on Earth” without naming the Pacific |
| Japan currency | yen | says Japan is the largest country in the world |
| main language of Brazil | Portuguese | answers English |
| organ that pumps blood | heart | answers “The human body” |
| light-to-chemical-energy process | photosynthesis | paraphrases the question without naming photosynthesis |
| days in a leap year | 366 | answers 365.25 days |
| 7 × 8 | 56 | outputs “8 multiplied by 8” |

Several Q/A cases then generate another `Question: ... Answer:` record, showing that the model has learned the **surface pattern** of Q/A text more strongly than the requested factual mapping.

## Cross-case qualitative findings

### What is clearly learned

- fluent-looking English fragments and common syntactic templates;
- punctuation, paragraph breaks, dialogue labels, and Q/A formatting;
- local lexical associations around the prompt topic;
- continuation of structured textual patterns;
- deterministic CUDA/FP16 generation through the qualified GDN-2 execution path.

### What remains weak at 500M tokens

- factual retrieval in simple direct questions;
- binding an entity to the requested relation or attribute;
- maintaining semantic consistency beyond a short local span;
- resisting tautological restatement of the prompt;
- resisting phrase/token loops under greedy decoding;
- dialogue-state tracking;
- simple structured relation completion;
- arithmetic completion in the supplied one-step probe.

## Interpretation

This result should not be summarized as “the model learned nothing.” The checkpoint is plainly modeling English surface structure and prompt formats. However, it also should not be described as having acquired dependable general knowledge or robust semantic generation.

The important distinction is:

> At approximately 500M consumed training tokens, the 20M GDN-2 hybrid shows clear language-model learning signal and strong format imitation, but the canonical greedy full-suite probe still exposes severe semantic drift, repetition, tautological answering, and no reliable direct factual retrieval across the twelve simple Q/A cases.

The result is also useful because it separates validation/next-token progress from open-ended generation quality. A substantially improved held-out next-token metric does not by itself imply that greedy free generation will produce correct factual answers or stable multi-sentence continuations at this scale.

Do not infer that the GDN-2 architecture itself is the sole cause of these weaknesses from this single scale point. Capacity, data exposure, objective, decoding, and lack of post-training all remain confounded. The fresh 20M / 2B experiment is therefore a useful next longitudinal point: rerun this exact frozen suite there before attributing the remaining failure pattern to model capacity or architecture.

## Comparison cautions

- This 500M run uses deterministic greedy decoding with a 32-token cap. Earlier qualitative evidence used different sampling settings and longer native prompt budgets, so detailed repetition rates and continuation lengths are not directly comparable across those runs.
- The 12-question direct-answer count is a narrow qualitative diagnostic, not a standardized benchmark score.
- The suite probes a pretrained base model, not an instruction-tuned or preference-aligned model. Instruction-following expectations should be evaluated separately after post-training.
- Final quantitative comparison should use the frozen evaluation stack (`eval_core_v1`, held-out metrics, and teacher-forced confidence/rank diagnostics) in addition to this qualitative evidence.

## Longitudinal use

Retain this checkpoint as the 500M qualitative baseline for the fixed 20M scaling series. For the next comparable full-suite run, keep the canonical decoding settings unchanged and compare at minimum:

- direct factual-answer count;
- structured relation completion;
- incidence and onset of greedy repetition;
- semantic coherence of the four continuation cases;
- whether the model answers the requested relation rather than restating the question.

The corresponding frozen full-suite protocol is recorded in the current post-pretraining prompt-test decision/runbook.