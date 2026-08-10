---
status: accepted
date: 2026-08-08
supersedes: null
---

# 0018 — Integrate FLA GDN-2 as a checkpoint-compatible CUDA backend

## Context

The standalone Tesla T4 qualification of `fla-core==0.5.1` passed GDN-2 forward parity, normal-decay backward gradient parity, strong-decay execution, and throughput benchmarking. At batch 4 / context 2,048, FLA was about `20.83x` faster than the adaptive PyTorch backend in normal forward execution, about `162.54x` faster in strong-decay forward execution, and about `135.44x` faster in strong-decay forward+backward execution.

The evidence strongly indicates that the late-training slowdown is caused by the correctness-first adaptive implementation rather than by a need to restrict learned decay. The user approved integrating FLA and testing it against the existing Small-LLM layer/checkpoint path before deciding whether to start a fresh 500M run.

## Decision

Integrate FLA by replacing only the GDN-2 recurrence execution backend on the supported CUDA path.

Preserve unchanged:

- `GatedDeltaNet2` / `StableGatedDeltaNet2` learned projections and parameters;
- decay, erase, and write semantics;
- parameter names and state-dict keys;
- checkpoint model configuration;
- the adaptive PyTorch backend as the explicit correctness/reference fallback.

The active 64-token CUDA path uses `fla.ops.gdn2.chunk_gdn2` through a Small-LLM adapter. Backend selection happens at call time because models are commonly constructed on CPU and moved to CUDA afterwards.

On supported 64-token CUDA execution, FLA is mandatory: a missing FLA package or kernel failure is surfaced rather than silently reverting to the pathological adaptive training path. CPU and unsupported/non-64 test geometries continue to use the adaptive backend.

The qualified dependency is `fla-core==0.5.1`. FLA receives the already-normalized q/k tensors and already-computed log decay / erase / write gates, so `use_qk_l2norm_in_kernel=False` and `use_gate_in_kernel=False`; no learned gate semantics are moved into FLA.

## Integration gate

Before authorizing a fresh production training run, run the one-click integration probe:

```bash
python kaggle/run_gdn2_fla_layer_probe.py
```

It must verify:

1. full Small-LLM GDN-layer output parity;
2. full-layer parameter/input gradient parity;
3. the same parity with forced approximately `log_decay=-6`;
4. unchanged checkpoint parameter keys.

An existing trainer checkpoint may additionally be checked with:

```bash
python kaggle/run_gdn2_fla_layer_probe.py --checkpoint /path/to/checkpoint.pt
```

This strict-loads the same checkpoint into an adaptive-reference model and the integrated FLA model and compares short full-model logits.

## Consequences

- Decay clipping/bounding remains unnecessary unless later evidence contradicts the backend diagnosis.
- Existing checkpoints are expected to remain load-compatible because the backend adds no parameters.
- Exact bitwise continuation is not promised after switching execution backends; different floating-point operation ordering may cause normal trajectory divergence even when mathematical parity holds.
- No fresh 500M restart is authorized by this ADR. The user will inspect checkpoint behavior first and decide separately whether to start a clean run.
- ADR 0005 remains relevant as the correctness/reference fallback and as historical rationale for the adaptive backend; this ADR does not erase that implementation history.

## Links

- [`../../model/gdn2_fla.py`](../../model/gdn2_fla.py)
- [`../../model/gdn2_stable.py`](../../model/gdn2_stable.py)
- [`../../kaggle/run_gdn2_fla_layer_probe.py`](../../kaggle/run_gdn2_fla_layer_probe.py)
- [`../reference/gdn2_fla_backend.md`](../reference/gdn2_fla_backend.md)
- [`../archive/gdn2_fla_investigation/gdn2_fla_investigation_handoff.md`](../archive/gdn2_fla_investigation/gdn2_fla_investigation_handoff.md)
- [`../evidence/gdn2_fla_t4_full_probe_2026-08-08.md`](../evidence/gdn2_fla_t4_full_probe_2026-08-08.md)
- [`0016-qualify-fla-gdn2-before-changing-decay.md`](0016-qualify-fla-gdn2-before-changing-decay.md)
- [`0005-adapt-gdn2-chunks-to-decay-span.md`](0005-adapt-gdn2-chunks-to-decay-span.md)
