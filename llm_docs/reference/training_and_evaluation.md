# Training, evaluation, and checkpointing

_Last reviewed: 2026-08-13_

## Reproducibility principle

Scaling or architecture claims must bind exact model geometry, tokenizer, dataset/source identity, data order, optimizer/update geometry, schedule, consumed targets, checkpoint identity, and evaluation protocol. Execution topology may differ only when explicitly qualified to preserve the scientific batch semantics.

## Instrumentation

Production training records at least:

- exact model/config identity and source commit;
- optimizer step and consumed loss-bearing targets;
- train and held-out validation loss/perplexity;
- gradient norm/clipping and FP16 scaler/overflow state;
- scheduled LR and optimizer-group state;
- peak GPU memory, step time, and throughput;
- dataset wait/readiness state where relevant;
- checkpoint/publication results;
- W&B stable run identity.

## Checkpoint boundary

Checkpoint only after a successful complete optimizer block. Model state and dataset cursor must describe the same acknowledged boundary. Exact resume restores model, optimizer, scheduler, scaler, counters, RNG, and dataset position; source/config drift fails closed.

Remote model storage follows ADR 0055:

```text
run/<run_id>/...       live two-phase exact-resume state
models/<run_id>/...    stable completed artifacts
```

Stable artifacts require native `local_manifest.json` verification. Live two-phase checkpoints additionally use their publication manifest and pointer protocol.

Dataset durability is separate. New dataset shards use HF Storage Buckets under ADR 0054; historical `drive_*` schema fields remain readable compatibility names.

## Frozen intrinsic evaluation

`eval_core_v1` is the canonical project intrinsic evaluation corpus. A full bundle records:

- NLL/loss and perplexity;
- bits per decoded target byte;
- top-1/5/10 next-token accuracy;
- ECE calibration and bins;
- per-cluster loss/perplexity;
- cluster macro and exact source-mixture-weighted loss;
- worst cluster;
- sequence-position buckets;
- document-bootstrap 95% intervals;
- wall time, throughput, and peak allocated VRAM.

Compare bundles only when their `eval_manifest_sha256` is identical. The current 20M/500M, 20M/2B, and 100M/2B scorecard uses manifest `aa7b6157e5f420dd53a99552685eaed01962ee45c23cbe438e1321a886422792`.

## Frozen qualitative comparison

ADR 0025, not the Python sampler defaults, is authoritative for canonical post-pretraining qualitative comparison:

```text
temperature: 0
top_p: 1
top_k: 0
seed: 17
samples_per_prompt: 1
questions_only: false
max_new_tokens: 32
trace_top_tokens: 0
```

The live-run version historically uses the validation-selected `best` pointer. Stable `models/...` artifacts preserve the terminal stable model rather than a validation-best history, so stable-model evaluation must record that endpoint-selection difference explicitly rather than pretending a `best` pointer exists.

A sampled decoding run is useful supplementary evidence but must never be merged into the canonical greedy score. Example: the sampled 100M/2B run answered `Paris`, while the greedy endpoint run answered `France`.

## Full evaluator versus exact qualitative protocol

`trainer.eval_suite` combines `eval_core_v1` metrics with qualitative prompts, but its prompt runner currently uses native per-case generation budgets and does not expose ADR 0025's global `max_new_tokens=32`. Therefore:

- use its intrinsic metrics for canonical full scale comparison;
- prompt outputs from runs with identical flags are mutually comparable;
- run `trainer.post_pretraining_prompt_suite` or the stable-model wrapper for the exact ADR-0025 qualitative protocol until the full evaluator gains an equivalent global cap.

## Teacher-forced confidence diagnostic

The teacher-forced held-out diagnostic is deterministic and operates on raw next-token logits; temperature/top-k/top-p are irrelevant. It records true-token probability/rank, top-k membership, entropy, and representative high-confidence errors over the frozen validation sample. It complements, but does not replace, full `eval_core_v1`.

## Current scale interpretation

The 2026-08-13 three-way full evaluation shows 20M improves from 500M→2B but unevenly, while 100M/2B improves every retained cluster and every context-position bucket relative to 20M/2B. This is evidence that 20M is capacity-constrained by 2B. Behavioral capability remains a separate gate for ADR 0050.

Evidence: [`../evidence/scaling/20m_500m_20m_2b_100m_2b_full_eval_2026-08-13.md`](../evidence/scaling/20m_500m_20m_2b_100m_2b_full_eval_2026-08-13.md).
