# S0 Budget and Scalability Decision

_Last updated: 2026-08-06 Europe/Rome_

## Frozen decision

For the approximately-20M-parameter checkpoint pretrained on the approximately-100M-token corpus, the first S0 supervised-fine-tuning experiment uses a finite maximum horizon of:

```text
4,000,000 loss-bearing target tokens
```

The authorized candidate mixture remains:

```text
85% filtered Smol-SmolTalk instruction targets
15% frozen ClimbMix replay targets
```

Within the instruction portion, the current proposed source-level allocation is:

```text
75.0% smol-magpie-ultra-short
10.0% smol-contraints
 7.5% smollm-rewrite-30k
 7.5% smol-summarize-20k
```

The source allocation remains subject to the pinned-revision audit and is not silently promoted to frozen by this file.

## Interpretation of the 4M horizon

The 4M value is specific to the 20M-model / 100M-pretraining-token qualification experiment. It is not a universal SFT budget and must not become a hard-coded implementation limit.

The production implementation must support mandatory evaluation and selection points within the finite stream, provisionally:

```text
0.5M, 1M, 2M, and 4M cumulative loss-bearing target tokens
```

The final selected S0 checkpoint may precede the 4M endpoint when instruction performance saturates, base-model retention degrades, or generation quality worsens.

## Scalability contract

The SFT dataset and trainer modules must be geometry- and budget-scalable for later, larger, and more heavily pretrained models. In particular:

- token budgets, source shares, replay shares, shard sizes, checkpoint cadence, validation cadence, and effective loss-bearing targets per optimizer update must be configuration and manifest values rather than constants;
- the implementation must support finite streams larger than 4M targets without changing dataset or trainer mathematics;
- counters, scheduling, checkpointing, and resume must use committed loss-bearing target tokens and immutable data identities;
- source quotas must be measured by loss-bearing target tokens, not row counts or serialized byte size;
- no implicit repetition or oversampling is allowed merely to reach a configured quota;
- the dataset builder must be able to produce larger unique-token streams from the same pinned source family when future experiments authorize them;
- the trainer must reconstruct the same model geometry from any supported native base checkpoint rather than contain 20M-specific assumptions.

## Implementation status

This decision authorizes design and implementation of a reusable SFT data/training surface. It does not record that the implementation or its qualification tests already exist.

The model architecture is expected to remain identical to the selected base checkpoint during SFT. Post-training changes the serialized data, target mask, data mixture, optimizer state, learning-rate schedule, evaluation, and checkpoint metadata; it does not silently alter the decoder geometry or tokenizer vocabulary.
