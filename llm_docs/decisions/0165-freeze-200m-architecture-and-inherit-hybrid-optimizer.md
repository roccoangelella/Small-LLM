---
status: accepted
date: 2026-09-08
supersedes: null
owners: [Small-LLM]
---

# ADR 0165: freeze the 200M architecture and inherit the hybrid optimizer

## Context

ADR 0160 selects approximately 200M parameters trained on 100B targets. ADRs 0161-0164 freeze the widened learning-rate family, proportional phase timing, exact LR anchors, and the Modal H100 / Beam RTX4090 provider boundary. The remaining scientific architecture choice is how to obtain the approximately-200M model while preserving comparability with the completed approximately-100M family.

The approximately-100M model uses 20 decoder blocks, `d_model=512`, `d_ff=1408`, eight 64-dimensional attention/GDN heads, and the repeating `[GDN-2, GDN-2, GDN-2, gated full MHA]` pattern. The user accepted scaling this model by width rather than depth so the next experiment does not simultaneously alter the recurrent/attention depth allocation.

## Decision

Freeze the next model as the same 20-layer hybrid family widened to:

```text
semantic vocab:          50,257
padded vocab:            50,304
context:                 2,048
layers:                  20
d_model:                 768
d_ff:                    2,048
attention heads:         12
attention head_dim:      64
GDN key heads:           12
GDN value heads:         12
GDN key/value dim:       64
layer rhythm:            [GDN-2, GDN-2, GDN-2, gated full MHA] x 5
normalization:           RMSNorm
FFN:                     SwiGLU
dropout:                 0
tied embedding/LM head:  yes
```

Under the current implementation this geometry has **203,978,740 learned parameters** (tied embedding/output counted once) and is the model referred to operationally as `200M`.

Do not add layers for this scale step. The change is wider embedding/residual representations, wider FFNs, and more 64-dimensional heads while preserving depth and the 3:1 GDN/MHA rhythm.

Retain the existing production optimizer contract unchanged:

- hybrid whole-matrix Muon + AdamW routing;
- Muon momentum `0.95`;
- Muon update RMS target `0.18`;
- AdamW `beta1=0.9`, `beta2=0.95`, `eps=1e-8`;
- weight decay `0.1` with the existing no-decay exceptions;
- global gradient clipping at `1.0`;
- FP32 optimizer state with the qualified FP16 training path.

The widened LR schedule is governed separately by ADRs 0161-0163. Provider-specific microbatch slicing may differ for memory/throughput, but the global block remains 64 sequences / 131,072 target tokens per optimizer update.

## Consequences

- The scale experiment changes capacity primarily through width, not depth, improving comparability with the 100M family.
- Every fourth layer remains full attention; the 20-layer model therefore contains 15 GDN-2 mixers and 5 full-MHA mixers.
- The same 64-dimensional per-head geometry gives 12 heads at `d_model=768`.
- Existing checkpoints remain identity-stable because historical 20M/100M factories are not changed.
- A new explicit trainer/model preset is required; silently mapping `200M` onto the 100M `substantive` factory is forbidden.
- Modal and Beam must build exactly the same model/trainer configuration; only execution slicing may differ.

## Validation before long launch

1. verify the exact factory fields and parameter count;
2. verify provider profile symmetry for `200M` / `100B`;
3. verify the exact ADR-0163 WSqD command/state on both providers;
4. run CPU/static tests without allocating a production GPU;
5. run provider-specific live GPU smoke/throughput qualification before full dispatch.

## Links

- `llm_docs/decisions/0160-select-200m-100b-as-next-pretraining-run.md`
- `llm_docs/decisions/0161-broaden-200m-100b-lr-anchor-range-slightly.md`
- `llm_docs/decisions/0162-scale-100m-10b-lr-phase-anchors-proportionally-to-200m-100b.md`
- `llm_docs/decisions/0163-freeze-200m-100b-lr-anchors.md`
- `llm_docs/decisions/0164-restrict-200m-100b-to-modal-h100-and-beam-rtx4090.md`
- `llm_docs/reference/model_architecture.md`
- `llm_docs/reference/model_geometry.md`
- `llm_docs/reference/optimizer_strategy.md`
