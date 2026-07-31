# Dataset and Tokenization Interface

_Last updated: 2026-07-31_

## Tokenizer contract

The model consumes the GPT-2 byte-level BPE IDs already embedded in the pinned Nemotron-ClimbMix records.

- tokenizer ID: `gpt2`;
- semantic vocabulary size: 50,257;
- EOD token: `<|endoftext|>`, ID 50256;
- cache type: explicit little-endian `uint16`;
- accepted records are not detokenized and retokenized.

The model may allocate an internally padded embedding/output matrix of 50,304 rows for hardware alignment. The additional rows are implementation padding, not tokenizer vocabulary. They must not become valid dataset IDs, targets, or sampled outputs.

## Source and content policy

The initial pretraining source is the pinned `nvidia/Nemotron-ClimbMix` revision recorded in `LLM_PROJECT_MEMORY.md`.

Clusters 1–10 and 12–20 are accepted. Cluster 11, the explicit software/programming cluster, is excluded. The result is described as programming-cluster-excluded, not guaranteed code-free.

The training mixture is the exact empirical source-token distribution of the pinned release conditioned on cluster 11 being excluded.

## Sequence geometry

For context length `L = 2048`, every stored training sequence has 2,049 IDs:

```text
stored: [t0, ..., t2048]
input:  [t0, ..., t2047]
target: [t1, ..., t2048]
```

Stride is 2,048. Consecutive sequences overlap by one physically duplicated token so every intended next-token transition is retained.

## Model-facing batch contract

The trainer consumer must provide at minimum:

- `input_ids` with shape `[batch, 2048]`;
- `target_ids` with shape `[batch, 2048]`;
- any required loss mask for padding or invalid positions;
- deterministic block and sequence identifiers for resume and audit;
- source/provenance counters needed by joint checkpointing.

The base causal mask belongs to full-attention layers. GDN-2 consumes the sequence causally through its recurrent/chunkwise implementation.

## Durability boundary

A prepared block becomes trainer-visible only after the corresponding local immutable shard is durable. Joint checkpoints must bind the exact model state to the last consumed and durable dataset positions.
