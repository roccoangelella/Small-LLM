---
status: accepted_evidence
date: 2026-08-08
---

# FLA GDN-2 full-layer integration qualification

## Purpose

After the standalone `fla-core==0.5.1` GDN-2 operator passed Tesla T4 forward/backward qualification, the next gate was to verify the actual Small-LLM GDN layer rather than only the bare recurrence operator.

The integration deliberately preserves the existing `StableGatedDeltaNet2` projections, learned parameters, decay/erase/write semantics, parameter names, and checkpoint keys. Only CUDA recurrence execution is delegated to `fla.ops.gdn2.chunk_gdn2`.

## User-run Kaggle result

The user ran:

```bash
python kaggle/run_gdn2_fla_layer_probe.py
```

and reported the following summary:

```text
Small-LLM integrated FLA GDN-2 layer qualification

SUMMARY
layer_forward_backward_parity: True
checkpoint_parity: None
INTEGRATION QUALIFIED for checkpoint evaluation; fresh-training authorization remains separate.
JSON report: /kaggle/working/gdn2_fla_layer_probe.json
```

The checkpoint-specific branch was not run in this invocation, hence `checkpoint_parity: None`.

## What this establishes

The integration-level probe passed full Small-LLM GDN-layer forward/backward parity, including the strong-decay integration case implemented by the probe. This advances FLA from a standalone-operator candidate to an integrated backend suitable for checkpoint evaluation/migration testing.

This result does not itself prove bitwise continuation of an existing training trajectory. FLA and the prior PyTorch backend use different floating-point operation orderings, so a resumed trajectory is expected to diverge numerically while preserving the same mathematical recurrence.

## Resume compatibility refinement

The active 20M/500M launcher historically passes `--gdn-chunk-size 32`, and this value is serialized in trainer/model configuration. Changing the CLI configuration to 64 would therefore cause strict checkpoint model-configuration restore to fail.

The checkpoint-compatible design instead keeps the saved/configured `gdn_chunk_size=32` unchanged while using FLA's fixed 64-token kernel internally on CUDA. Chunk size is an execution grouping, not a learned model parameter. CPU/reference execution retains the configured adaptive chunk size.

The 500M trainer continues to restore the latest verified remote checkpoint and its model, optimizer, scheduler, scaler, RNG state, consumed-token cursor, and WSD position. The intended migration changes only GDN-2 CUDA recurrence execution after restore.

## Related files

- `model/gdn2_fla.py`
- `model/gdn2_stable.py`
- `kaggle/run_gdn2_fla_layer_probe.py`
- `kaggle/run_20m_500m.py`
- `llm_docs/decisions/0018-integrate-fla-gdn2-as-checkpoint-compatible-cuda-backend.md`
