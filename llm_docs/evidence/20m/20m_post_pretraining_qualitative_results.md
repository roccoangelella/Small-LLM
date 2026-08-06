# 20M Post-Pretraining Qualitative Results

_Last updated: 2026-08-05 14:22 Europe/Rome_

## Result identity

The first qualitative post-pretraining suite was run against the completed 20M qualification checkpoint:

```text
repository: roccoangelella/small-llm-20m-qualification
run ID: 20m-qualification-dataset-001
pointer: latest
checkpoint ID: step-00000306
checkpoint prefix: run/20m-qualification-dataset-001/checkpoints/step-00000306/last
remote metric: null
```

The run used the standard qualitative suite with one seeded sample per prompt and the default sampling policy:

```text
temperature: 0.8
top-p: 0.95
top-k: 50
base seed: 17
```

The retained JSON artifact was 9,488 bytes with SHA-256:

```text
dbf32cdfd2f41e718aabf471cd07db6b5fe4c97700a39413c265eefaa8971303
```

## Quantitative summary of the qualitative suite

```text
prompt cases: 18
continuation cases: 4
structured cases: 2
simple-question cases: 12
total prompt tokens: 302
total generated tokens: 985
mean generated tokens per case: 54.72
median generated tokens per case: 48
cases reaching their generation budget: 13 / 18
cases stopping early on EOS: 5 / 18
simple questions with the expected direct answer present: 0 / 12
```

One science-continuation case emitted EOS immediately after one generated token. Other early stops occurred in the dialogue, Hamlet, water-freezing, and photosynthesis cases.

## Qualitative observations

The checkpoint is clearly no longer random. It learned several properties expected from a causal English language model:

- valid GPT-2-tokenized output and stable autoregressive generation;
- recognizable English word order and sentence fragments;
- punctuation, paragraph breaks, quotation marks, and question/answer formatting;
- local topic associations such as oceans with marine life, plants with light, and repeated scientific or technical vocabulary;
- continuation of the `Question: ... Answer:` surface pattern, even when it did not answer the supplied question;
- nontrivial EOS behavior rather than always filling the entire requested budget.

The main limitations were also consistent across prompts:

- rapid topic drift away from the supplied entity or relation;
- heavy attraction to generic high-frequency phrases such as “the same time,” “the most important,” “the world,” and “the first”;
- repetition, malformed clauses, and weak agreement over more than a short local span;
- failure to complete simple structured relations such as `Germany | Berlin`;
- no reliable retrieval of elementary factual answers in the twelve one-sample Q/A probes;
- weak dialogue-state tracking and no dependable instruction-following behavior.

## Interpretation

The user's initial qualitative verdict was “not that bad.” The project interpretation is similarly positive but narrow:

> The completed 20M qualification run demonstrates that the model, tokenizer, data path, optimizer, checkpointing system, and generation path jointly learned non-random English next-token structure.

This is a meaningful engineering and learning-signal success for the smoke-scale run. It is not evidence that the checkpoint is a useful general-knowledge model, a chatbot, or a reliable question-answering system.

The weak factual and coherent-generation performance is not surprising at this training scale. The model has 20,637,592 parameters and saw 10,006,528 planned training target tokens, approximately 0.485 training tokens per parameter in one pass. This run was designed as an engineering qualification, not a compute-optimal capability run. These outputs therefore must not be used to reject the architecture before the controlled approximately-100M comparison.

## Decision and next use

The 20M checkpoint is accepted as having passed the qualitative learning-signal smoke test.

For later architecture and scale comparisons, retain this exact prompt set and sampling configuration as a longitudinal probe, but add:

1. deterministic greedy output;
2. multiple fixed sampling seeds;
3. held-out loss and perplexity;
4. structured benchmark scoring;
5. explicit repetition and EOS statistics.

Future qualitative comparisons should distinguish three separate questions:

- did the model learn fluent local English structure;
- did it retain and retrieve factual knowledge;
- did it learn instruction-following behavior through a later post-training stage.

The current checkpoint provides evidence only for the first question, and only at smoke-model quality.