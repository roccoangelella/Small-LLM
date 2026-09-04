# ADR 0147 — Batch evaluation v2 execution without changing the benchmarks

Date: 2026-09-04
Status: accepted and wired

## Decision

Keep the evaluation-v2 benchmark definitions, cases, scoring rules, seeds and sampled decoding contract unchanged, but replace the dominant serial execution paths with evaluation-only batching.

The 2026-09-04 100M/2B full pretraining qualification showed that the original L20 adapter was operationally too slow on Kaggle T4 hardware: it scored each conditional-likelihood candidate with a separate model forward. The same implementation pattern would also make SFT Behavior v2 unnecessarily expensive because its 720 cases are evaluated greedily and, in full mode, under sampled robustness seeds 17/18/19 for both parent and tuned models.

## Wired execution contract

### L20 conditional likelihood

- Preserve all six tasks and all full-suite examples.
- Preserve zero-shot conditional-likelihood scoring and the existing lm-evaluation-harness metric contract.
- Encode every request exactly as before.
- Length-bucket independent requests, restore outputs to the original harness request order, and score multiple candidates per model forward.
- Cap batches at 16 requests and 8,192 padded input tokens. Long requests therefore shrink the effective batch automatically.
- Emit explicit request-progress messages so long external benchmark phases no longer appear frozen.

### Generated evaluation views

- Preserve native per-case generation budgets.
- Preserve greedy decoding exactly as a protocol: temperature=0, top_p=1, top_k=0.
- Preserve canonical sampled decoding exactly as a protocol: temperature=1, top_p=1, top_k=0.
- Preserve every per-case RNG seed. Batched sampled generation owns one `torch.Generator` per request, so length bucketing does not replace the existing seed contract with a shared batch RNG stream.
- Batch up to 16 independent generation requests per forward and restore outputs to original case order.
- Apply the batched generation engine to Base Prompt v2 and SFT Behavior v2.
- Emit progress at coarse completion intervals.

## Multi-GPU boundary

Do not introduce `torch.nn.DataParallel` around the GDN model as part of this optimization. The repository's qualified dual-T4 execution architecture is DDP/`torchrun`; `DataParallel` would replicate the module during repeated forwards and is not an acceptable unqualified shortcut for autoregressive evaluation.

This ADR therefore targets the dominant serial-forward waste first. A future dual-rank evaluator may shard independent evaluation requests under the established DDP process model, but that must be qualified separately and must not change benchmark semantics.

## Validation requirements

Regression tests must establish that:

1. batched greedy generation matches the legacy one-request generation helper on a deterministic reference model;
2. batched sampled generation preserves per-request RNG behavior on the same reference model;
3. batched L20 likelihood scoring matches one-request scoring for the same encoded context/continuation pairs.

## Consequences

The v2 JSON schemas gain execution metadata describing length bucketing and batch limits, but benchmark identities and headline metrics are unchanged. Full-suite sample counts must not be reduced merely to recover runtime.
